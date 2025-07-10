**Private Key Password for ssh:** robomimic-v2

# Startup
- Turn on robot
- Login to https://172.16.0.6/desk/
  - Username: rl2
  - Password: 123456789
- Click two bars in the top right -> activate FCI
- Click Unlock button under "End-Effector"
- Robot light should turn green




## Data Collection:
  - Check Headset:
    - Make sure white light is on
    - If program is not running, start "Quest Teleop" in headset
  - Setup for data collection:
    - `cd /home/aloha/nadun/`
    - `source robomimic-env/bin/activate`
    - `cd /home/aloha/nadun/gello_software/experiments`
  - Start collection
    - `python run_rl2_env.py --task=<name of task>`
  - Manually moving the robot:
    - Go to https://172.16.0.6/desk/
    - Hit two bars in the top right to close the menu
    - In bottom right, switch from 'execution' to 'programming'
    - Click the two buttons on the wrist of the robot to manually move it.
    - Press 'B' on the controller to start the reset process
    - Switch from 'programming' to 'execution' on the web interface
    - **NOTE: DO NOT SWITCH FROM PROGRAMMING TO EXECUTION BEORE PRESSING 'B'**

## Postprocessing and uploading Demos

- Postprocessing
  - Make sure the environment from above is activated
  - `cd /home/aloha/nadun/gello_software/scripts`
  - `python postprocess_demos.py --demo_dir=<path to where demos were saved for task> --save_dir=<where to save postprocessed demos> --task_name=<name of task>`
- Check quality of demos (optional):
  - `cd /home/aloha/nadun/gello_software/scripts`
  - `python merged_hdf5_to_videos.py --demo=<path to demo> --save_dir=<where to save the videos> --task_name=<name of task> (optional)`
- Upload the postprocessed file (ends with "demo")
  - `rsync -ah -P <path to file> nkra3@sky1.cc.gatech.edu:/coc/flash7/nkra3/Droid/robomimic-dev/datasets/kitchen/`
  - `rsync -ah -P <path to file> kwang601@sky1.cc.gatech.edu:/coc/flash8/kwang601/robomimic-dev/datasets/kitchen/`

## Testing Models:

- Download latest models:
  - `cd /home/aloha/nadun/gello_software/rl2_experiment`
  - `./download_models.sh`
- Setup for testing:
  - `cd /home/aloha/nadun/`
  - `source robomimic-env/bin/activate`
  - `cd robomimic-dev/robomimic/scripts`
- Run a model:
  - `python run_trained_agent_rl2.py --agent=<path_to_agent>`

## Shutdown:
- Always shutdown robot and laptop after data collection/model testing
- Robot:
  - Go to https://172.16.0.6/desk/
  - Click two bars in the top right
  - Click "Shutdown"
  - Wait for "Finished Shutting Down" message
  - Turn off controller