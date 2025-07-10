import time
import serial

import numpy as np
from serial.tools import list_ports

################################
# PPP (Point-to-Point) Helpers #
################################

def ppp_stuff(input_bytes):
    FRAME_CHAR = 0x7E
    ESC_CHAR = 0x7D
    ESC_MASK = 0x20

    working_buf = bytearray(input_bytes)

    # Escape all 0x7D
    indices = [i for i, b in enumerate(working_buf) if b == ESC_CHAR]
    for idx in reversed(indices):
        working_buf[idx] ^= ESC_MASK
        working_buf.insert(idx, ESC_CHAR)

    # Escape all 0x7E
    indices = [i for i, b in enumerate(working_buf) if b == FRAME_CHAR]
    for idx in reversed(indices):
        working_buf[idx] ^= ESC_MASK
        working_buf.insert(idx, ESC_CHAR)

    # Add frame delimiters
    working_buf.insert(0, FRAME_CHAR)
    working_buf.append(FRAME_CHAR)
    return bytes(working_buf)


def ppp_unstuff(input_bytes):
    FRAME_CHAR = 0x7E
    ESC_CHAR = 0x7D
    ESC_MASK = 0x20

    # Must start and end with FRAME_CHAR
    if not input_bytes or input_bytes[0] != FRAME_CHAR or input_bytes[-1] != FRAME_CHAR:
        return bytearray()

    working_input = bytearray(input_bytes[1:-1])  # remove the two 0x7E

    # Unescape
    i = 0
    while i < len(working_input):
        if working_input[i] == ESC_CHAR:
            i += 1
            if i < len(working_input):
                working_input[i] ^= ESC_MASK
        i += 1

    return working_input


def compute_checksum(data_bytes):
    """Compute the 8-bit 2's-complement negative of the sum of the bytes."""
    return (-sum(data_bytes)) & 0xFF


########################################
# Ability Hand Plain Python Controller #
########################################

class AbilityHandController:
    """
    A simple, non-ROS Python controller to command the Ability Hand over serial.
    Uses PPP stuffing if desired, similar to the 'gello' style Robotiq driver.
    """

    def __init__(
        self,
        comport=None,
        baud_rate=460800,
        stuff_data=True,
        hand_address=0x50,
        reply_mode=0x10,
    ):
        """
        :param comport:  If None, we will auto-search for a USB/COM port.
        :param baud_rate: Serial baud rate. Default from example = 460800.
        :param stuff_data: Whether to use PPP byte stuffing for sending commands.
        :param hand_address: Ability Hand address byte (0x50 default).
        :param reply_mode: Format header for your TX messages (0x10 default).
        """
        self.comport = comport
        self.baud_rate = baud_rate
        self.stuff_data = stuff_data
        self.hand_address = hand_address
        self.reply_mode = reply_mode

        self.serial_port = None
        self._setup_serial()

        self.latest_positions = [15.0] * 6  # e.g. 6 DOF placeholder

    def _setup_serial(self):
        """Open or auto-search for a serial port that can talk to the Ability Hand."""
        if self.comport:
            port_candidate = self.comport
            print(f"[AbilityHandController] Using specified port: {port_candidate}")
        else:
            # Try auto-detect
            print("[AbilityHandController] Searching for a USB serial port...")
            port_candidate = None
            for p in list_ports.comports():
                if p.device:
                    port_candidate = p.device
                    break

            if not port_candidate:
                raise IOError("[AbilityHandController] No available USB/COM port found!")

        print(f"[AbilityHandController] Connecting to {port_candidate} at {self.baud_rate} baud...")
        try:
            self.serial_port = serial.Serial(port_candidate, self.baud_rate, timeout=0.1)
            print("[AbilityHandController] Successfully connected.")
        except serial.SerialException as e:
            raise IOError(f"[AbilityHandController] Failed to open {port_candidate}: {e}")

    def close(self):
        """Shutdown the serial port when done."""
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("[AbilityHandController] Serial port closed.")

    def generate_tx(self, positions):
        """
        Build a TX packet (un-stuffed) for the given finger positions.

        `positions` is a list or tuple of 6 floats, each in degrees.
        Example typical range: 0..90 for fingers, -/+ for the thumb rotator.
        """
        tx_buffer = bytearray()
        # Byte 0: Address
        tx_buffer.append(self.hand_address)
        # Byte 1: reply mode
        tx_buffer.append(self.reply_mode)

        # Next: 6 fingers, each 2 bytes (low, high) in signed 16-bit
        # Scale from [0..150 deg] => [-32767..32767], per your original code
        for pos_deg in positions:
            pos_fixed = int(pos_deg * 32767.0 / 150.0)
            lo = pos_fixed & 0xFF
            hi = (pos_fixed >> 8) & 0xFF
            tx_buffer.append(lo)
            tx_buffer.append(hi)

        # Finally: Checksum
        cksum = compute_checksum(tx_buffer)
        tx_buffer.append(cksum)

        return tx_buffer

    def send_positions(self, positions):
        """
        Send the desired finger positions to the Ability Hand.
        `positions`: list of 6 floats in degrees.
        """
        if not self.serial_port or not self.serial_port.is_open:
            print("[AbilityHandController] Serial port not open! Cannot send command.")
            return

        tx_msg = self.generate_tx(positions)
        if self.stuff_data:
            tx_msg = ppp_stuff(tx_msg)

        self.serial_port.write(tx_msg)
        self.latest_positions = positions

        '''
        # Read more bytes than just 10 (the response can be 38 or 72+ if stuffed)
        incoming = self.serial_port.read(200)  # <-- CHANGED: read up to 200 bytes

        if incoming:
            pass
        else:
            print("[AbilityHandController] No response received from hand.")
        # --------------------- END CHANGES ---------------------
        '''

    def open_hand(self):
        positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # all open
        self.send_positions(positions)

    def close_hand(self):
        positions = [90.0, 90.0, 90.0, 90.0, 50.0, -50.0]
        self.send_positions(positions)

    def grasp(self, command):
        positions = np.degrees(command).tolist()
        # print(positions)
        self.send_positions(positions)

    def get_gripper_act(self):
        return np.radians(self.latest_positions).tolist()


#####################
# Usage (Example)   #
#####################
if __name__ == "__main__":
    hand = AbilityHandController(
        comport=None,      # or "/dev/ttyUSB0"
        baud_rate=460800,
        stuff_data=True,   # enable PPP stuffing
        hand_address=0x50,
        reply_mode=0x10
    )

    try:
        print("Opening the hand...")
        hand.open_hand()
        time.sleep(2.0)

        print("Closing the hand...")
        hand.close_hand()
        time.sleep(2.0)

        custom_positions = [45.0, 45.0, 45.0, 0.0, 10.0, -10.0]
        print(f"Sending custom positions: {custom_positions}")
        hand.send_positions(custom_positions)
        time.sleep(2.0)

    finally:
        hand.close()
