# Codes-for-AdaDepth
This repository provides the implementation of our method for depth estimation of dynamic objects, achieving state-of-the-art performance on datasets like Cityscapes and Waymo Open. Our approach leverages intrinsic image information for robust and efficient depth estimation, requiring only 30 epochs for training without offline mask generation.

# AdaDepth

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **AdaDepth: Exploiting Inherent Scene Information for Self-Supervised Depth Estimation in Dynamic Scenes**
>
> [under review]

This code is for non-commercial use; please see the [license file](LICENSE) for terms.

If you find our work useful in your research please consider citing our paper:

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [Single-GPU](#-single-gpu-training)
  - [Multi-GPU](#-multi-gpu-training)
- [Evaluation](#evaluation)
  - [Depth](#-depth)

## ⚙️ Installation

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch=1.31.1 torchvision=0.14.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
```
We ran our experiments with PyTorch 1.13.1, CUDA 11.7, Python 3.9.12 and Ubuntu 20.04.

<!-- We recommend using a [conda environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) to avoid dependency conflicts.

We also recommend using `pillow-simd` instead of `pillow` for faster image preprocessing in the dataloaders. -->

## 💾 Data-preparation

### 💾 KITTI Dataset
🔹 You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```
wget -i kitti_archives_to_download.txt -P data_dir/kitti_raw
cd data_dir/kitti_raw/
unzip "*.zip"
rm *.zip
cd ../..
```
Please refer to [Monodepth2](https://github.com/nianticlabs/monodepth2) for `kitti_archives_to_download`([link](https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/splits/kitti_archives_to_download.txt)). Once downloaded and unzipped to `data_dir/kitti/raw`, you should be able to obtain following directoy sturcture.
```
data_dir/kitti_raw/
  |-- 2011_09_26
    |-- calib_cam_to_cam.txt
    |-- calib_imu_to_velo.txt
    |-- calib_velo_to_cam.txt
    |-- 2011_09_26_drive_0001_sync
      |-- image_00
      |-- image_01
      |-- image_02
      |-- image_03
      |-- oxts
      |-- velodyne_points
    |-- 2011_09_26_drive_0002_sync
    |-- ...
  |-- 2011_09_28
  |-- 2011_09_29
  |-- 2011_09_30
  |-- 2011_10_03
```
There are roughly 57G, 22G, 5.5G, 47G, and 40G in `2011_09_26/`, `2011_09_28/`, `2011_09_29/`, `2011_09_30/`, and `2011_10_03/`, respectively.

### 💾 Waymo Open Dataset

🔹 Please refer to the [official website](https://waymo.com/open/) for downloading the Waymo Open Dataset. Once downloaded and unzipped, you should be able to obtain following directoy sturcture.
```
</PATH/TO/waymo_records>
  |-- train
    |-- segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord
    |-- segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord
    |-- ...
    |-- segment-990914685337955114_980_000_1000_000_with_camera_labels.tfrecord
    |-- segment-9985243312780923024_3049_720_3069_720_with_camera_labels.tfrecord
  |-- val
    |-- segment-10203656353524179475_7625_000_7645_000_with_camera_labels.tfrecord
    |-- segment-1024360143612057520_3580_000_3600_000_with_camera_labels.tfrecord
    |-- ...
    |-- segment-9579041874842301407_1300_000_1320_000_with_camera_labels.tfrecord
    |-- segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord
```
`waymo_records/train/` should have 798 files with suffix `*_with_camera_labels.tfrecord` totalling roughly 760G and `waymo_records/val/` should have 202 files with suffix `*_with_camera_labels.tfrecord` totalling roughly 192G.

🔹 ` Please see the “Prepare_data” section of [Dynamo-Depth](https://dynamo-depth.github.io) for more details.

## 💾 Cityscapes Dataset

From [Cityscapes official website](https://www.cityscapes-dataset.com/) download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip` into the `CS_RAW` folder.

Preprocess the Cityscapes dataset using the `prepare_train_data.py`(from SfMLearner) script with following command:

```bash
cd CS_RAW
unzip leftImg8bit_sequence_trainvaltest.zip
unzip camera_trainvaltest.zip
cd ..

python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir CS_RAW \
    --dataset_name cityscapes \
    --dump_root CS \
    --seq_length 3 \
    --num_threads 8
```

Once preprocess the Cityscapes dataset, you should be able to obtain the following directory structure.
```
</PATH/TO/cityscapes_preprocesses>
  ├── aachen
  │   ├── tubingen_000038_000010.png
  │   ├── tubingen_000038_000011_cam.txt
  │   ├── tubingen_000038_000011.png
  │   ├── tubingen_000038_000012_cam.txt
  │   ├── ...
  ├── bochum
  ├── bremen
  ├── cologne
  ├── darmstadt
  ├── dusseldorf
  ├── erfurt
  ├── hamburg
  ├── hanover
  ├── jena
  ├── krefeld
  ├── monchengladbach
  ├── strasbourg
  ├── stuttgart
  ├── tubingen
  ├── ulm
  ├── weimar
  └── zurich
```

Download cityscapes depth ground truth(provided by manydepth) for evaluation:
```bash
cd ..
cd splits/cityscapes/
wget https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip
unzip gt_depths_cityscapes.zip
cd ../..


