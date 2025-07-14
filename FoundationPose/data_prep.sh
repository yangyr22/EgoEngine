python ../inpaint/depth_anything2/run_video.py \
    --video-path /coc/flash7/yliu3735/workspace/inpaint/temp_data/result_Trial/cropped_video.mp4 \
    --outdir /coc/flash7/yliu3735/workspace/inpaint/temp_data/result_Trial/cropped_video.mp4 \
    --grayscale \
    --pred-only

python ../inpaint/depth_prep.py \
    -b /coc/flash7/yliu3735/workspace/inpaint/temp_data/result_Trial \

python ../inpaint/sam2_segment.py \
        --input_img /path/to/image.png \
        --point_coords 750 500 \
        --point_labels 1 \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_cfg /.../segment_anything2/configs/sam2.1/sam2.1_hiera_base_plus.yaml \
        --sam_ckpt /.../segment_anything2/checkpoints/sam2.1_hiera_base_plus.pt

####################################TODO####################################