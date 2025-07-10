import h5py
import numpy as np

demo_fn = "/media/aloha/Data/robomimic-v2/Demos/Merged/can/move_to_skynet/left_cam_high_75_demos_demo.hdf5"
processed_fn = "/media/aloha/Data/robomimic-v2/Demos/Merged/can/move_to_skynet/left_cam_high_75_demos_filtered.hdf5"
original_dataset = h5py.File(demo_fn, 'r')['data']

with h5py.File(processed_fn, 'w') as dataset:
    grp = dataset.create_group('data')
    grp.attrs['env_args'] = original_dataset.attrs['env_args']
    total = 0
    for demo in original_dataset:
        print(f"Processing: {demo}")
        grp.create_group(demo)
        proc_demo_grp = grp[demo]

        orig_demo = original_dataset[demo]
        control_enabled = orig_demo['control_enabled'][:]

        enabled_count = control_enabled.sum()
        total += enabled_count
        for name, item in orig_demo.items():
            if isinstance(item, h5py.Dataset):
                proc_demo_grp.create_dataset(name, data=np.array(item[control_enabled]))
            elif isinstance(item, h5py.Group):
                item_group = proc_demo_grp.create_group(name)
                for name, dataset in item.items():
                    item_group.create_dataset(name, data=np.array(dataset[control_enabled]))
        proc_demo_grp.attrs['num_samples'] = enabled_count
    grp.attrs['total'] = total



