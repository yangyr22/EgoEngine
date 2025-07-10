import cv2

# 源视频（要被缩放的）
src_path = "/coc/flash7/yliu3735/workspace/inpaint/example/video/paragliding/original_video.mp4"
# 参考视频（分辨率将作为基准）
ref_path = "/coc/flash7/yliu3735/workspace/inpaint/example/video/paragliding/original_video1.mp4"
# 输出路径
out_path = "/coc/flash7/yliu3735/workspace/inpaint/example/video/paragliding/original_video_resized.mp4"

# 打开参考视频并获取目标尺寸
ref_cap = cv2.VideoCapture(ref_path)
ret, ref_frame = ref_cap.read()
ref_cap.release()

if not ret:
    raise RuntimeError("Failed to read reference video.")

target_height, target_width = ref_frame.shape[:2]

# 打开源视频
src_cap = cv2.VideoCapture(src_path)
fps = src_cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (target_width, target_height))

while True:
    ret, frame = src_cap.read()
    if not ret:
        break
    resized = cv2.resize(frame, (target_width, target_height))
    out.write(resized)

src_cap.release()
out.release()

print(f"✅ Done. Saved to: {out_path}")
