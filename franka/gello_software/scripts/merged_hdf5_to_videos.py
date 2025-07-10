import h5py
import os
import imageio
import argparse


def create_video_for_image_obs(demos, image_obs_name, video_folder):
    video_fn = os.path.join(video_folder, f"{image_obs_name}.mp4")
    writer = imageio.get_writer(video_fn, fps=20)

    #Iterate over all demos
    for demo in demos:
        demo_images = demos[demo][f"obs/{image_obs_name}"][:]
        # Write the video for this camera
        for i in range(demo_images.shape[0]):
            writer.append_data(demo_images[i])

    writer.close()


def hdf_to_videos(demo_fn, video_folder, task_name):
    ### With New Setup

    output_folder_name = task_name
    video_folder = os.path.join(video_folder,output_folder_name)
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    print(f"Saving videos to : {video_folder}")
    print("==================================================================================================")

    demo_file = h5py.File(demo_fn)
    demos = demo_file['data']

    # First, find all image obs names
    image_obs_list = []
    for demo in demos:
        demo = demos[demo]
        obs = demo['obs']
        for mod in obs:
            if "image" in mod or "rgb" in mod:
                image_obs_list.append(mod)
        break

    # Create video for each camera
    for image_obs in image_obs_list:
        print(f"Processing camera: {image_obs}")
        create_video_for_image_obs(demos, image_obs, video_folder)


    ##### With ImageIo

    # shoulderview_left_video_fn = os.path.join(video_folder, "shoulderview_left_all_demos.mp4")
    # shoulderview_right_video_fn = os.path.join(video_folder, "shoulderview_right_all_demos.mp4")
    # wrist_video_fn = os.path.join(video_folder, "wrist_all_demos.mp4")
    # left_video_fn = os.path.join(video_folder, "left_realsense_all_demos.mp4")
    # right_video_fn = os.path.join(video_folder, "right_realsense_all_demos.mp4")
    #
    # shoulderview_left_writer = imageio.get_writer(shoulderview_left_video_fn, fps=20)
    # shoulderview_right_writer = imageio.get_writer(shoulderview_right_video_fn, fps=20)
    # wrist_writer = imageio.get_writer(wrist_video_fn, fps=20)
    # left_writer = imageio.get_writer(left_video_fn, fps=20)
    # right_writer = imageio.get_writer(right_video_fn, fps=20)
    #
    #
    # for j, demo in enumerate(demos):
    #     print(f"Processing demo: {j}")
    #     shoulderview_left_images = demos[demo]['obs/shoulderview_left_image']
    #     shoulderview_right_images = demos[demo]['obs/shoulderview_right_image']
    #     wrist_images = demos[demo]['obs/wrist_image']
    #
    #     #left_images = demos[demo]['obs/left_image']
    #     #right_images = demos[demo]['obs/right_image']
    #     for i in range(shoulderview_left_images.shape[0]):
    #         shoulderview_left_writer.append_data(shoulderview_left_images[i])
    #         shoulderview_right_writer.append_data(shoulderview_right_images[i])
    #         wrist_writer.append_data(wrist_images[i])
    #
    #         #left_writer.append_data(left_images[i])
    #         #right_writer.append_data(right_images[i])
    #
    # shoulderview_left_writer.close()
    # shoulderview_right_writer.close()
    # wrist_writer.close()
    #
    # left_writer.close()
    # right_writer.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # location of folder to save hdf5 files in
    parser.add_argument(
        "--demo",
        default="",
        type=str,
        required=True,
    )

    # location of source folder with hdf5s to merge / postprocess (optional)
    # if not provided, a source.hdf5 folder must be present in @save_dir,
    # corresponding to the merged source demonstration file.
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        required=True
    )


    parser.add_argument(
        "--task_name",
        type=str,
        default=""
    )

    args = parser.parse_args()

    if args.task_name == "":
        demo_name = args.demo.split('/')[-1]
        ind = demo_name.find("_demo")
        args.task_name = demo_name[0:ind]

    hdf_to_videos(args.demo, args.save_dir, args.task_name)
