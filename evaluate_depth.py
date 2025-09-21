import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import datasets
from networks import *
from utils import *
from layers import disp_to_depth, compute_depth_errors
import warnings
import networks
import numpy as np
from layers import *
import pdb
import os
import requests
from tqdm import tqdm

def download_file(url, save_path):
    """下载文件并带进度条"""
    if os.path.exists(save_path):
        print(f"✔ Found {save_path}, skip downloading")
        return save_path
    
    print(f"⬇ Downloading {url} to {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    with open(save_path, "wb") as file, tqdm(
        desc=save_path, total=total, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
    return save_path

warnings.filterwarnings("ignore")

STEREO_SCALE_FACTOR = 5.4

def eval_args():
    parser = argparse.ArgumentParser(description='Evaluation Parser')

    parser.add_argument('--pretrained_path',
                        type=str,
                        help="path of model checkpoint to load")
    parser.add_argument("--backbone",
                        type=str,
                        help="backbone of depth encoder",
                        default="hrnet",
                        choices=["resnet", "litemono", "diffnet"])
    parser.add_argument("--model",
                        type=str,
                        help="backbone of depth encoder",
                        default="lite-mono",
                        choices=["lite-mono", "lite-mono-8m"])
    parser.add_argument("--num_layers",
                        type=int,
                        help="number of resnet layers",
                        default=18,
                        choices=[18, 34, 50, 101, 152])
    parser.add_argument("--batch_size",
                        type=int,
                        help="batch size",
                        default=4)
    parser.add_argument("--height",
                        type=int,
                        help="input image height",
                        default=192)
    parser.add_argument("--width",
                        type=int,
                        help="input image width",
                        default=640)
    parser.add_argument("--num_workers",
                        type=int,
                        help="number of dataloader workers",
                        default=2)
    parser.add_argument("--min_depth",
                        type=float,
                        help="minimum depth",
                        default=0.1)
    parser.add_argument("--max_depth",
                        type=float,
                        help="maximum depth",
                        default=100.0)
    parser.add_argument("--drop_path",
                        type=float,
                        help="minimum depth",
                        default=0.2)
    parser.add_argument("--post_process",
                        help="if set will perform the flipping post processing "
                            "from the original monodepth paper",
                        action="store_true")
    parser.add_argument("--use_stereo",
                        help="if set, uses stereo pair for training",
                        action="store_true")
    parser.add_argument("--test_cs_all",
                    help="if set, test_all_pixels_on_cs",
                    action="store_true")
    ## paths of test datasets
    parser.add_argument('--kitti_path',
                        type=str,
                        help="data path of KITTI, do not set if you do not want to evaluate on this dataset")
    parser.add_argument('--make3d_path',
                        type=str,
                        help="data path of Make3D, do not set if you do not want to evaluate on this dataset")
    parser.add_argument('--nyuv2_path',
                        type=str,
                        help="data path of NYU v2, do not set if you do not want to evaluate on this dataset")
    parser.add_argument('--cityscapes_path',
                        type=str,
                        help="data path of Cityscapes, do not set if you do not want to evaluate on this dataset")
    parser.add_argument('--waymo_path',
                        type=str,
                        help="data path of Waymo, do not set if you do not want to evaluate on this dataset")
    parser.add_argument('--nuscenes_path',
                        type=str,
                        help="data path of Waymo, do not set if you do not want to evaluate on this dataset")                        
    args = parser.parse_args()
    return args

def compute_errors(gt, pred):
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log10(gt) - torch.log10(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    l_mask = torch.tensor(l_mask).cuda()
    r_mask = torch.tensor(r_mask.copy()).cuda()
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def load_model(args):
    print("-> Loading weights from {}".format(args.pretrained_path))

    encoder_path = os.path.join(args.pretrained_path, "encoder_ref.pth")
    decoder_path = os.path.join(args.pretrained_path, "depth_ref.pth")

    # 如果本地不存在
    encoder_url = "https://github.com/zhihudqlwo/Codes-for-AdaDepth/releases/download/v1.0/encoder_ref.pth"
    decoder_url = "https://github.com/zhihudqlwo/Codes-for-AdaDepth/releases/download/v1.0/depth_ref.pth"

    encoder_path = download_file(encoder_url, encoder_path)
    decoder_path = download_file(decoder_url, decoder_path)


    # encoder_path = os.path.join(args.pretrained_path, "encoder.pth")
    # decoder_path = os.path.join(args.pretrained_path, "depth.pth") # for coarse depth
    encoder_dict = torch.load(encoder_path, map_location='cuda')
    decoder_dict = torch.load(decoder_path, map_location='cuda')

    # 初始化模型
    if args.backbone == "resnet":
        depth_encoder = networks.ResnetEncoder(args.num_layers, False)
        depth_decoder = networks.DepthDecoder(depth_encoder.num_ch_enc, scales=range(1))
    elif args.backbone == "litemono":
        depth_encoder = networks.LiteMono(model=args.model, drop_path_rate=args.drop_path)

        depth_decoder = networks.LiteMonoDepthDecoder(depth_encoder.num_ch_enc, scales=range(1))
    elif args.backbone == "diffnet":
        depth_encoder = networks.test_hr_encoder.hrnet18(True)
        depth_encoder.num_ch_enc = [ 64, 18, 36, 72, 144 ]
    
        depth_decoder = networks.HRDepthDecoder(
            depth_encoder.num_ch_enc, scales=range(1))

    def load_weights(model, state_dict):
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module

        model_state_dict = model.state_dict()


        new_state_dict = {}

        # 遍历加载的权重
        for k, v in state_dict.items():
            # 如果键完全匹配
            if k in model_state_dict:
                new_state_dict[k] = v
            elif k.startswith('module.depth_encoder.') and k[len('module.depth_encoder.'):] in model_state_dict:
                new_state_dict[k[len('module.depth_encoder.'):]] = v
            elif k.startswith('module.depth_decoder.') and k[len('module.depth_decoder.'):] in model_state_dict:
                new_state_dict[k[len('module.depth_decoder.'):]] = v
            elif 'module.' + k in model_state_dict:
                new_state_dict['module.' + k] = v
            else:
                # 如果仍然不匹配，打印警告
                print(f"Warning: Key '{k}' in state_dict does not match any key in the model.")

        # 加载匹配的权重
        model.load_state_dict(new_state_dict, strict=False)
        return model

def test_kitti(args, dataloader, depth_encoder, depth_decoder, eval_split='eigen'):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    gt_path = os.path.join(os.path.dirname(__file__), "splits", "kitti", eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]

    pred_disps = []
    for data in dataloader:
        input_color = data[("color", 0, 0)].cuda()
        if args.post_process:
            input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
        output = depth_decoder(depth_encoder(input_color))
        pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
        pred_disp = pred_disp[:, 0]
        if args.post_process:
            N = pred_disp.shape[0] // 2
            pred_disp = batch_post_process_disparity(pred_disp[:N], torch.flip(pred_disp[N:], [2]))
        pred_disps.append(pred_disp)
    pred_disps = torch.cat(pred_disps, dim=0)

    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):
        gt_depth = torch.from_numpy(gt_depths[i]).cuda()
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = pred_disps[i:i+1].unsqueeze(0)
        pred_disp = F.interpolate(pred_disp, (gt_height, gt_width), mode="bilinear", align_corners=True)
        pred_depth = 1 / pred_disp[0, 0, :]
        if eval_split == "eigen":
            mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)
            crop_mask = torch.zeros_like(mask)
            crop_mask[
                    int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1
            mask = mask * crop_mask
        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]
        if args.use_stereo:
            pred_depth *= STEREO_SCALE_FACTOR
        else:
            ratio = torch.median(gt_depth) / torch.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio  
        pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_depth_errors(gt_depth, pred_depth))

    if not args.use_stereo:
        ratios = torch.tensor(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios / med)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

    mean_errors = torch.tensor(errors).mean(0)

    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))

def test_cityscapes(args, dataloader, depth_encoder, depth_decoder):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    gt_path = os.path.join(os.path.dirname(__file__), "splits", "cityscapes", "gt_depths")
    pred_disps = []
    doj_masks = []
    # 对一个批次的深度图进行预测
    for data in dataloader:
        input_color = data[("color", 0, 0)].cuda()
        if args.post_process:
            input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
        output = depth_decoder(depth_encoder(input_color))
        pred_disp, _ = disp_to_depth(output[("disp", 0)], args.min_depth, args.max_depth)
        pred_disp = pred_disp[:, 0]
        if args.post_process:
            N = pred_disp.shape[0] // 2
            pred_disp = batch_post_process_disparity(pred_disp[:N], torch.flip(pred_disp[N:], [2]))
        pred_disps.append(pred_disp)

        doj_masks.append(data["doj_mask"].cuda()) # data["doj_mask"] [8, 1, 192, 512]

    #所有batch拼接的成果
    pred_disps = torch.cat(pred_disps, dim=0)
    doj_masks = torch.cat(doj_masks, dim=0)

    errors = []
    errors_obj = []
    ratios = []
    ratios_doj = []
    for i in range(pred_disps.shape[0]):
        gt_depth = np.load(os.path.join(gt_path, str(i).zfill(3) + '_depth.npy'))
        gt_height, gt_width = gt_depth.shape[:2]
        # crop ground truth to remove ego car -> this has happened in the dataloader for inputs
        gt_height = int(round(gt_height * 0.75))
        gt_depth = torch.from_numpy(gt_depth[:gt_height]).cuda()
        pred_disp = pred_disps[i:i+1].unsqueeze(0)
        pred_disp = F.interpolate(pred_disp, (gt_height, gt_width), mode="bilinear", align_corners=True)
        pred_depth_raw = 1 / pred_disp[0, 0, :]

        # when evaluating cityscapes, we centre crop to the middle 50% of the image.
        # Bottom 25% has already been removed - so crop the sides and the top here
        # if test_cs_all:
        #     gt_depth = gt_depth
        #     pred_depth = pred_depth
        # else:
        gt_depth_raw = gt_depth[256:, 192:1856]
        pred_depth = pred_depth_raw[256:, 192:1856]
        pred_depth_obj = pred_depth_raw[256:, 192:1856]      # 512,1664

        mask = (gt_depth_raw > MIN_DEPTH) & (gt_depth_raw < MAX_DEPTH)
        # print(mask.size())
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth_raw[mask]

        # pred_depth_obj = pred_depth_obj[mask]
       
        if args.use_stereo:
            pred_depth *= STEREO_SCALE_FACTOR
        else:
            ratio = torch.median(gt_depth) / torch.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio  
            # print(pred_depth.size())

        pred_depth = torch.clamp(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_depth_errors(gt_depth, pred_depth))

        #-----------加入动态掩码---------------
        doj_mask = doj_masks[i]
        doj_mask.unsqueeze(0) 
        doj_mask_resize = F.interpolate(doj_mask[None], [gt_height, gt_width]) 
        doj_mask_resize = doj_mask_resize[0][0] 
        doj_mask_crop = doj_mask_resize[256:, 192:1856]
        doj_mask = mask * doj_mask_crop.bool()           

        if doj_mask.sum() == 0: continue

        pred_depth_doj = pred_depth_obj[doj_mask]
        gt_depth_doj = gt_depth_raw[doj_mask]

        if args.use_stereo:
            pred_depth_doj *= STEREO_SCALE_FACTOR
        else:
            ratio_doj = torch.median(gt_depth_doj) / torch.median(pred_depth_doj) 
            ratios_doj.append(ratio_doj)
            pred_depth_doj *= ratio
        pred_depth_doj = torch.clamp(pred_depth_doj, MIN_DEPTH, MAX_DEPTH)
        errors_obj.append(compute_depth_errors(gt_depth_doj, pred_depth_doj))    


    if not args.use_stereo:
        ratios = torch.tensor(ratios)
        med = torch.median(ratios)
        std = torch.std(ratios / med)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, std))

    mean_errors = torch.tensor(errors).mean(0)
    mean_errors_obj = torch.tensor(errors_obj).mean(0)

    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors.tolist()))  
    print(("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("{: 8.3f} | " * 7 + "\n").format(*mean_errors_obj.tolist()))  

def main(args):

    depth_encoder, depth_decoder = load_model(args)
    input_resolution = (args.height, args.width)
    
    print(" Evaluated at resolution {} * {}".format(input_resolution[0], input_resolution[1]))
    if args.post_process:
        print(" Post-process is used")
    else:
        print(" No post-process")
    if args.use_stereo:
        print(" Stereo evaluation - disabling median scaling")
        print(" Scaling by {} \n".format(STEREO_SCALE_FACTOR))
    else:
        print(" Mono evaluation - using median scaling \n")

    splits_dir = os.path.join(os.path.dirname(__file__), "splits")

    if args.kitti_path:
        ## evaluate on eigen split
        print(" Evaluate on KITTI with eigen split:")
        filenames = readlines(os.path.join(splits_dir, "kitti", "eigen", "test_files.txt")) 
        dataset = datasets.KITTIRAWDataset(args.kitti_path, filenames, input_resolution[0], input_resolution[1], [0], 1, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_kitti(args, dataloader, depth_encoder, depth_decoder, "eigen")

        ## evaluate on eigen_benchmark split
        print(" Evaluate on KITTI with eigen_benchmark split (improved groundtruth):")
        filenames = readlines(os.path.join(splits_dir, "kitti", "eigen_benchmark", "test_files.txt")) 
        dataset = datasets.KITTIRAWDataset(args.kitti_path, filenames, input_resolution[0], input_resolution[1], [0], 1, is_train=False, img_ext='.png')
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_kitti(args, dataloader, depth_encoder, depth_decoder, "eigen_benchmark")

    if args.cityscapes_path:
        print(" Evaluate on Cisyscapes:")
        filenames = readlines(os.path.join(splits_dir, "cityscapes", "test_files.txt"))
        dataset = datasets.CityscapesDataset(args.cityscapes_path, filenames, input_resolution[0], input_resolution[1], [0], 1, is_train=False, load_mask=True)
        dataloader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, drop_last=False)
        with torch.no_grad():
            test_cityscapes(args, dataloader, depth_encoder, depth_decoder)  

if __name__ == '__main__':
    args = eval_args()
    main(args)
