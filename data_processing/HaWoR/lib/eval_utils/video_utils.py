import cv2
import os
import subprocess

def make_video_grid_2x2(out_path, vid_paths, overwrite=False):
    """
    将四个视频以原始分辨率拼接成 2x2 网格。

    :param out_path: 输出视频路径。
    :param vid_paths: 输入视频路径的列表（长度必须为 4）。
    :param overwrite: 如果为 True，覆盖已存在的输出文件。
    """
    if os.path.isfile(out_path) and not overwrite:
        print(f"{out_path} already exists, skipping.")
        return

    if any(not os.path.isfile(v) for v in vid_paths):
        print("Not all inputs exist!", vid_paths)
        return

    # 确保视频路径长度为 4
    if len(vid_paths) != 4:
        print("Error: Exactly 4 video paths are required!")
        return

    # 获取视频路径
    v1, v2, v3, v4 = vid_paths

    # ffmpeg 拼接命令，直接拼接不调整大小
    cmd = (
        f"ffmpeg -i {v1} -i {v2} -i {v3} -i {v4} "
        f"-filter_complex '[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]' "
        f"-map '[v]' {out_path} -y"
    )

    print(cmd)
    subprocess.call(cmd, shell=True, stdin=subprocess.PIPE)

def create_video_from_images(image_list, output_path, fps=15, target_resolution=(540, 540)):
    """
    将图片列表合成为 MP4 视频。

    :param image_list: 图片路径的列表。
    :param output_path: 输出视频的文件路径（如 output.mp4）。
    :param fps: 视频的帧率（默认 15 FPS）。
    """
    # if not image_list:
    #     print("图片列表为空！")
    #     return

    # 读取第一张图片以获取宽度和高度
    first_image = cv2.imread(image_list[0])
    if first_image is None:
        print(f"无法读取图片: {image_list[0]}")
        return

    height, width, _ = first_image.shape
    if height != width:
        if height < width:
            vis_w = target_resolution[0]
            vis_h = int(target_resolution[0] / width * height)
        elif height > width:
            vis_h = target_resolution[0]
            vis_w = int(target_resolution[0] / height * width)
    else:
        vis_h = target_resolution[0]
        vis_w = target_resolution[0]
    target_resolution = (vis_w, vis_h)

    # 定义视频编码器和输出参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, target_resolution)

    # 遍历图片列表并写入视频
    for image_path in image_list:
        frame = cv2.imread(image_path)
        frame_resized = cv2.resize(frame, target_resolution)
        if frame is None:
            print(f"无法读取图片: {image_path}")
            continue
        video_writer.write(frame_resized)

    # 释放视频写入器
    video_writer.release()
    print(f"视频已保存至: {output_path}")