python depth_anything_video.py \
    --input_video /coc/flash7/yliu3735/workspace/inpaint/example/video/paragliding/transparent.mp4 \
    --output_path /coc/flash7/yliu3735/workspace/inpaint/results/depth_video.mp4 \
    --depth_encoder vitl \
    --depth_ckpt /coc/flash7/yliu3735/workspace/inpaint/weights/depth_anything_v2_vitl.pth \
    --device cuda \
    --resize_factor 1.0