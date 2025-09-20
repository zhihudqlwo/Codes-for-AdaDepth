
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py \
--pretrained_path folders-to-models \
--backbone litemono \
--drop_path 0.2 \
--model lite-mono-8m \
--batch_size 8 \
--width 512 \
--height 192 \
--cityscapes_path .../cityscapes \