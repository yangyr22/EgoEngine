# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Obtained from  https://github.com/polymetis/polymetis/python/polymetis/robot_client/robotiq_gripper/third_party/robotiq_2finger_grippers/robotiq_2f_gripper.py
# Which was modified from: https://github.com/Danfoa/robotiq_2finger_grippers/blob/master/robotiq_2f_gripper_control/src/robotiq_2f_gripper_control/robotiq_2f_gripper.py

import serial
from serial.serialutil import SerialException

# from pymodbus.client.sync import ModbusSerialClient
# from .robotiq_modbus_rtu import comModbusRtu
# import comModBusRtu 
from gello.pymodbus_robotiq import comModBusRtu 

from math import ceil

import numpy as np
import array

import time

ACTION_REQ_IDX = 7
POS_INDEX = 10
SPEED_INDEX = 11
FORCE_INDEX = 12


class Robotiq2FingerGripper:
    def __init__(self, device_id=0, stroke=0.085, comport="/dev/ttyUSB0", baud=115200):

        self.client = comModBusRtu.communication()

        connected = self.client.connectToDevice(device=comport)
        if not connected:
            raise Exception(
                "Communication with gripper %d on serial port: %s and baud rate: %d not achieved"
                % (device_id, comport, baud)
            )

        self.init_success = True
        self.device_id = device_id + 9
        self.stroke = stroke
        self.initialize_communication_variables()

        self.message = []

        print("Activating gripper...")
        self.activate_emergency_release()
        self.sendCommand()
        time.sleep(0.5)
        self.deactivate_emergency_release()
        self.sendCommand()
        time.sleep(0.5)
        self.activate_gripper()
        self.sendCommand()


    def _update_cmd(self):

        # Initiate command as an empty list
        self.message = []
        # Build the command with each output variable
        self.message.append(self.rACT + (self.rGTO << 3) + (self.rATR << 4))
        self.message.append(0)
        self.message.append(0)
        self.message.append(self.rPR)
        self.message.append(self.rSP)
        self.message.append(self.rFR)

    def sendCommand(self):
        """Send the command to the Gripper."""
        return self.client.sendCommand(self.message)

    def getStatus(self):
        """Request the status from the gripper and return it in the Robotiq2FGripper_robot_input msg type."""

        # Acquire status from the Gripper
        status = self.client.getStatus(6)

        # Check if read was successful
        if status is None:
            return False

        # Assign the values to their respective variables
        self.gACT = (status[0] >> 0) & 0x01
        self.gGTO = (status[0] >> 3) & 0x01
        self.gSTA = (status[0] >> 4) & 0x03
        self.gOBJ = (status[0] >> 6) & 0x03
        self.gFLT = status[2]
        self.gPR = status[3]
        self.gPO = status[4]
        self.gCU = status[5]

        return True

    def initialize_communication_variables(self):
        # Out
        self.rPR = 0
        self.rSP = 255
        self.rFR = 150
        self.rARD = 1
        self.rATR = 0
        self.rGTO = 0
        self.rACT = 0
        # In
        self.gSTA = 0
        self.gACT = 0
        self.gGTO = 0
        self.gOBJ = 0
        self.gFLT = 0
        self.gPO = 0
        self.gPR = 0
        self.gCU = 0

        self._update_cmd()
        self._max_force = 100.0  # [%]

    def shutdown(self):
        self.client.close()

    def activate_gripper(self):
        self.rACT = 1
        self.rPR = 0
        self.rSP = 255
        self.rFR = 150
        self._update_cmd()

    def deactivate_gripper(self):
        self.rACT = 0
        self._update_cmd()

    def activate_emergency_release(self, open_gripper=True):
        self.rATR = 1
        self.rARD = 1

        if open_gripper:
            self.rARD = 0
        self._update_cmd()

    def deactivate_emergency_release(self):
        self.rATR = 0
        self._update_cmd()

    def goto(self, pos, vel, force):
        """
        Sets the command to send the gripper to a position with a desired speed and force
        Args:
            pos: in range [1, 100], 100 being closed
            vel: in range [0, 0.085]
            force: in range [1, 100]
        """
        self.rACT = 1
        self.rGTO = 1
        self.rPR = int(np.clip(( -255.0) / (self.stroke * pos) + 255.0, 0, 255))
        self.rSP = int(np.clip(255.0 / (0.1 - 0.013) * vel - 0.013, 0, 255))
        self.rFR = int(np.clip(255.0 / (self._max_force) * force, 0, 255))
        self._update_cmd()

    def grasp(self, command):
        if command < 0: #fully open
            pos = 1
        elif command >= 0: # fully closed
            pos = 10000
        vel = 0.085 # max speed
        force = 40 # 75% force # TODO(VS) might wanna reduce this; 75 destroyed the coffee pod
        self.goto(pos, vel, force)
        self.sendCommand()

    def stop(self):
        self.rACT = 1
        self.rGTO = 0
        self._update_cmd()

    def is_ready(self):
        return self.gSTA == 3 and self.gACT == 1

    def is_reset(self):
        return self.gSTA == 0 or self.gACT == 0

    def is_moving(self):
        return self.gGTO == 1 and self.gOBJ == 0

    def is_stopped(self):
        return self.gOBJ != 0

    def object_detected(self):
        return self.gOBJ == 1 or self.gOBJ == 2

    def get_fault_status(self):
        return self.gFLT

    def get_pos(self):
        po = float(self.gPO)
        return np.clip(self.stroke / (3.0 - 230.0) * (po - 230.0), 0, self.stroke)
    
    def get_gripper_act(self):
        assert self.getStatus()
        pos = self.get_pos()
        return 1 - (pos / self.stroke)

    def get_req_pos(self):
        pr = float(self.gPR)
        return np.clip(self.stroke / (3.0 - 230.0) * (pr - 230.0), 0, self.stroke)

    def get_current(self):
        return self.gCU * 0.1
    

if __name__ == "__main__":
    gripper = Robotiq2FingerGripper()
    if not gripper.init_success:
        raise Exception(f"Unable to init gripper")
    if not gripper.getStatus():
        raise Exception("Failed to contact gripper")




    status = gripper.getStatus()
    fault = gripper.get_fault_status()

    # if (
    #     gripper.is_ready()
    #     and gripper.sendCommand()
    #     and gripper.getStatus()
    # ):
    #     print("Activated.")
    # else:
    #     raise Exception(f"Unable to activate!")

    time.sleep(2)
    while True:
        # vel = 0.085
        # force = 100
        # pos = 100
        # gripper.goto(pos, vel, force)
        # gripper.sendCommand()
        gripper.grasp(1)
        time.sleep(2)
        gripper.grasp(-1)
        time.sleep(2)