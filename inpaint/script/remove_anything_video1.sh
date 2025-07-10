python remove_anything_video_new.py \
  --input_video /coc/flash7/yliu3735/workspace/inpaint/cropped_video.mp4 \
  --coords_type key_in \
  --point_coords 530 1037 \
  --point_labels 1 \
  --dilate_kernel_size 5 \
  --output_dir ./results \
  --sam_ckpt weights/sam2.1_hiera_base_plus.pt \
  --sam_cfg segment_anything2/sam2/configs/sam2.1/sam2.1_hiera_b.yaml \
  --lama_config lama/configs/prediction/default.yaml \
  --lama_ckpt ./pretrained_models/big-lama \
  --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
  --vi_ckpt ./pretrained_models/sttn.pth \
  --mask_idx 2 \
  --fps 25 \
  --uv /coc/flash7/yliu3735/workspace/inpaint/uv.txt \
  # --segment-only


# <<for large model
# python remove_anything_video_new.py \
#   --input_video /coc/flash7/yliu3735/workspace/inpaint/result_video.mp4 \
#   --coords_type key_in \
#   --point_coords 534 808 \
#   --point_labels 1 \
#   --dilate_kernel_size 5 \
#   --output_dir ./results \
#   --sam_ckpt segment_anything2/checkpoints/sam2.1_hiera_large.pt \
#   --sam_cfg /coc/flash7/yliu3735/workspace/inpaint/segment_anything2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml \
#   --lama_config lama/configs/prediction/default.yaml \
#   --lama_ckpt ./pretrained_models/big-lama \
#   --tracker_ckpt vitb_384_mae_ce_32x4_ep300 \
#   --vi_ckpt ./pretrained_models/sttn.pth \
#   --mask_idx 1 \
#   --fps 25
