"""
Example usage:
    python scripts/postprocess_demos.py --demo_dir=/media/robot/0e230ee7-e486-472e-8972-20b4e9e4cf0f/bc_data/rmv2/gello/demos/coffee --save_dir=/media/robot/0e230ee7-e486-472e-8972-20b4e9e4cf0f/bc_data/rmv2/gello/merged_demos/merged_coffee_demos

    python scripts/postprocess_demos.py --source_path=/media/robot/0e230ee7-e486-472e-8972-20b4e9e4cf0f/bc_data/rmv2/gello/merged_demos/merged_coffee_demos/tmp/source.hdf5 --save_dir=/media/robot/0e230ee7-e486-472e-8972-20b4e9e4cf0f/bc_data/rmv2/gello/merged_demos/merged_coffee_demos/
    
"""

import argparse
from glob import glob
import os
import h5py
import json
import numpy as np
from collections import defaultdict
from PIL import Image

# TODO: maybe RESIZE_SHAPE should be set elsewhere
RESIZE_SHAPE = (128, 128)
#RESIZE_SHAPE = (672, 376)


def center_crop(im, t_h, t_w):
    assert(im.shape[-3] >= t_h and im.shape[-2] >= t_w)
    assert(im.shape[-1] in [1, 3])
    crop_h = int((im.shape[-3] - t_h) / 2)
    crop_w = int((im.shape[-2] - t_w) / 2)
    return im[..., crop_h:crop_h + t_h, crop_w:crop_w + t_w, :]

def resize_rgb_images(images, output_shape):
    """
    @param images: batch of images of shape (b,h,w,c)
    @param output_shape: tuple of (h',w') to resize to
    @return: resized batch of images after center cropping
    """
    image_list = []

    for i in range(images.shape[0]):
        crop_size = min(images[i].shape[0], images[i].shape[1])
        cropped_image = center_crop(images[i], crop_size,crop_size)
        resized_image = Image.fromarray(cropped_image).resize(output_shape, Image.BILINEAR)
        image_list.append(resized_image)

    images = np.stack(image_list, axis=0)
    return images


def merge_hdf5s(demo_dir, save_dir, args):
    """
    Helper function to merge all source demonstration hdf5s
    into one hdf5 in order.
    """
    source_hdf5s = glob(os.path.join(demo_dir, "*.hdf5"))

    all_hdf5s = []
    for source_hdf5_path in source_hdf5s:
        all_hdf5s.append(source_hdf5_path)

    # write demos in order to new file
    merged_hdf5_path = os.path.join(save_dir, f"{args.task_name}_source.hdf5")
    f_new = h5py.File(merged_hdf5_path, "w")
    f_new_grp = f_new.create_group("data")

    for i, source_hdf5_path in enumerate(all_hdf5s):
        print(f"Copying source demo: {i}")
        print('try read: ', source_hdf5_path)
        with h5py.File(source_hdf5_path, "r") as f:
            # copy this episode over under a different name
            demo_str = "demo_{}".format(i)
            f.copy("data/demo_0", f_new_grp, name=demo_str)
        

    f_new.close()


def postprocess_hdf5(src_hdf5_path, dst_hdf5_path):
    """
    Helper function to postprocess the collected (merged) demonstrations by
    aggregating chunks of data into single datasets per demonstration,
    """

    with h5py.File(src_hdf5_path, "r") as f_src:

        # ensure we traverse demos in order
        src_demos = list(f_src["data"].keys())
        inds = np.argsort([int(elem[5:]) for elem in src_demos])
        src_demos = [src_demos[i] for i in inds]

        with h5py.File(dst_hdf5_path, "w") as f_dst:

            f_dst_grp = f_dst.create_group("data")
            #First add env metadata
            # f_dst_grp.attrs["env_args"] = json.dumps(ENV_ARGS)

            num_written = 0
            total_samples = 0
            written_env_args = False

            for demo_ind, src_demo_key in enumerate(src_demos):
                print(f"Postprocessing demo: {demo_ind}")

                if not written_env_args:
                    written_env_args = True
                    f_dst_grp.attrs["env_args"] = f_src["data"][src_demo_key].attrs['env_args']


                # source group for this demonstration i.e. demo_{0,1,etc}
                src_grp = f_src["data"][src_demo_key]

                #destination group for this demo
                dst_grp = f_dst_grp.create_group("demo_{}".format(num_written))

                # first, we will merge the chunks of the various data in this trajectory

                # TODO: currently relying on Python to maintain the link to the dst_grp object so that when it is
                # changed in the function below, it also changes in the f_dst_grp object. This is very sketchy and
                # probably should be changed

                dst_grp = merge_chunks_and_create_dest_grp(src_grp, dst_grp)

                # f_dst_grp["demo_{}".format(num_written)] = dst_grp
                total_samples += dst_grp.attrs["num_samples"]
                num_written += 1

            f_dst_grp.attrs["total"] = total_samples


def merge_chunks_and_create_dest_grp(src_grp, dst_grp):
    """
    Merge all the chunks of a source demo and return it
    demo: a src demo
    """
    obs_key_to_merged_obs = defaultdict(list)
    actions_merged = []
    absolute_actions_merged = []
    ordered_chunk_names = sorted([chunk for chunk in src_grp.keys()], key=lambda x: int(x.split("_")[1]))
    control_enabled_merged = []

    for chunk in ordered_chunk_names:
        # Merging delta actions
        actions = src_grp[chunk]["action"]
        # actions = np.array(actions); actions[:,-1] = actions[:,-1]*2 + 1 ### TEMPORARY FIX, REMOVE LATER
        actions_merged.append(actions)
        # Maybe merging absolute actions
        if "action_absolute" in src_grp[chunk]:
            absolute_actions = src_grp[chunk]["action_absolute"]
            # absolute_actions = np.array(absolute_actions); absolute_actions[:,-1] = absolute_actions[:,-1]*2 + 1 ### TEMPORARY FIX, REMOVE LATER
            absolute_actions_merged.append(absolute_actions)
        # Merging obs
        all_obs = src_grp[chunk]["obs"]
        for obs in all_obs:
            obs_key_to_merged_obs[obs].append(all_obs[obs])

        # merged control enabled
        if "control_enabled" in src_grp[chunk]:
            control_enabled = src_grp[chunk]['control_enabled']
            control_enabled_merged.append(control_enabled)

    actions_merged = np.concatenate(actions_merged, axis=0)
    dst_grp.create_dataset("actions", data=np.array(actions_merged))
    if len(absolute_actions_merged) > 0:
        absolute_actions_merged = np.concatenate(absolute_actions_merged, axis=0)
        dst_grp.create_dataset("absolute_actions", data=np.array(absolute_actions_merged))

    for obs in obs_key_to_merged_obs:
        merged_obs = np.concatenate(obs_key_to_merged_obs[obs], axis=0)
        if "image" in obs or "rgb" in obs:
            merged_obs = resize_rgb_images(merged_obs, RESIZE_SHAPE)
        dst_grp.create_dataset(f"obs/{obs}", data=np.array(merged_obs))

    # Add control enabled array
    if len(control_enabled_merged) > 0:
        control_enabled_merged = np.concatenate(control_enabled_merged, axis=0)
        dst_grp.create_dataset("control_enabled", data=np.array(control_enabled_merged))

    delta_actions = np.zeros(actions_merged.shape)
    delta_actions[1:, :] = actions_merged[1:, :] - actions_merged[0:-1, :]
    delta_actions[:, -1] = actions_merged[:, -1]
    dst_grp.create_dataset("delta_actions", data=np.array(delta_actions))
    dst_grp.attrs["num_samples"] = actions_merged.shape[0]

    # assume sparse rewards for now, that occur at the end of a demonstration
    rewards = np.zeros(actions_merged.shape[0])
    rewards[-1] = 1.
    dones = np.zeros(actions_merged.shape[0])
    dones[-1] = 1.
    dones = dones.astype(int)
    dst_grp.create_dataset("rewards", data=np.array(rewards))
    dst_grp.create_dataset("dones", data=np.array(dones))

    # Add joint position obs without gripper obs
    joint_positions = dst_grp['obs/joint_positions'][:]
    dst_grp.create_dataset("obs/joint_position", data=np.array(np.copy(joint_positions[:])))



    return dst_grp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # location of folder to save hdf5 files in
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
        required=True,
    )

    # location of source folder with hdf5s to merge / postprocess (optional)
    # if not provided, a source.hdf5 folder must be present in @save_dir,
    # corresponding to the merged source demonstration file.
    parser.add_argument(
        "--demo_dir",
        type=str,
        default="",
    )

    parser.add_argument(
        "--source_path",
        type=str,
        default=None
    )

    # Optional, if not provided, the lowest level subdirectory of demo_dir will be the task name
    parser.add_argument(
        "--task_name",
        type=str,
        default=""
    )

    args = parser.parse_args()

    if args.task_name == "":
        args.task_name = args.demo_dir.split("/")[-1]

    if args.source_path is None:
        # create save dir if it doesn't exist
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        if args.demo_dir is not None:
            # merge hdf5s into one source hdf5 first (with sorted demos) and put in save_dir
            merge_hdf5s(demo_dir=args.demo_dir, save_dir=args.save_dir, args=args)

            # paths to source and postprocessed hdf5s
        source_hdf5_path = os.path.join(args.save_dir, f"{args.task_name}_source.hdf5")
    else:
        source_hdf5_path = args.source_path
    assert os.path.exists(source_hdf5_path), "Source demo does not exist"
    dest_hdf5_path = os.path.join(args.save_dir, f"{args.task_name}_demo.hdf5")

    postprocess_hdf5(src_hdf5_path=source_hdf5_path, dst_hdf5_path=dest_hdf5_path)