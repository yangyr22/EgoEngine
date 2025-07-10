python segment_anything.py \
  --input_video /coc/flash7/yliu3735/workspace/inpaint/marked_video.mp4 \
  --sam_cfg segment_anything2/sam2/configs/sam2.1/sam2.1_hiera_b.yaml \
  --sam_ckpt weights/sam2.1_hiera_base_plus.pt \
  --point_coords 1185 700 \
  --point_labels 1 \
  --output_dir /coc/flash7/yliu3735/workspace/inpaint/segment_results \
  --mask_idx 2