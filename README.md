# Things we have done now and where to find them:
  ## Retargeting
  run `dex_retargeting/utils/online_retargeting.py`
  ## Vrs data preperation(hand estimation)
  1. run command `aria_mps single -i path/to/vrs` 
  2. run `inpaint/hand_tracking_prep.py` (under env sam2)
  ## Inpaint
  run `inpaint/script/remove_anything_video1.sh` (under sam2)
  ## Object estimation
  1. get depth: run `inpaint/depth_anything2/run_video.py` and `inpaint/depth_prep.py`(under sam2)
  2. get segment: run `inpaint/sam2_segment.py` (under sam2)
  3. reconstruct data with `FoundationPose/video_to_frames.py` and download mesh
  4. run `FoundationPose/run_demo.py` (under foundationpose)
  ### TODO ###
  make this into a whole file.
