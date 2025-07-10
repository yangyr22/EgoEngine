import time
import math
from PCANBasic import PCANBasic, PCAN_USBBUS1, PCAN_BAUD_500K, PCAN_MESSAGE_STANDARD, TPCANMsg


###########################
# Allegro Hand Controller #
###########################

class AllegroHandPCANController:
    def __init__(self, pcan_channel=PCAN_USBBUS1, bitrate=PCAN_BAUD_500K):
        """
        Initialize the PCAN controller for the Allegro Hand.
        
        :param pcan_channel: The PCAN channel (default PCAN_USBBUS1).
        :param bitrate: The CAN bitrate (default 500Kbps).
        """
        self.pcan = PCANBasic()
        self.pcan_channel = pcan_channel

        # Initialize the PCAN-USB interface
        status = self.pcan.Initialize(self.pcan_channel, bitrate)
        if status != PCAN_ERROR_OK:
            raise RuntimeError(f"[AllegroHandPCAN] Failed to initialize PCAN: {status}")

        print(f"[AllegroHandPCAN] Connected to PCAN channel {self.pcan_channel} at {bitrate}")

    def send_positions(self, positions):
        """
        Send desired joint positions to the Allegro Hand via PCAN.
        
        :param positions: List of 16 joint angles in degrees.
        """
        if len(positions) != 16:
            print("[AllegroHandPCAN] Error: Expected 16 joint angles.")
            return

        # Convert angles to integer format expected by the hand
        joint_bytes = bytearray()
        for pos in positions:
            pos_fixed = int(pos * 32767.0 / 150.0)  # Scale [-32767, 32767]
            joint_bytes.append(pos_fixed & 0xFF)  # Low byte
            joint_bytes.append((pos_fixed >> 8) & 0xFF)  # High byte

        # Split into CAN frames (each frame supports 8 bytes max)
        frames = [joint_bytes[i:i+8] for i in range(0, len(joint_bytes), 8)]

        # Send each frame with an incrementing CAN ID (if needed)
        for i, frame in enumerate(frames):
            can_msg = TPCANMsg()
            can_msg.ID = 0x123 + i  # Adjust CAN ID if necessary
            can_msg.LEN = len(frame)
            can_msg.MSGTYPE = PCAN_MESSAGE_STANDARD
            can_msg.DATA = (c_ubyte * 8)(*frame)

            status = self.pcan.Write(self.pcan_channel, can_msg)
            if status != PCAN_ERROR_OK:
                print("[AllegroHandPCAN] ERROR: Failed to send CAN message.")

    def close(self):
        """Close the PCAN connection."""
        self.pcan.Uninitialize(self.pcan_channel)
        print("[AllegroHandPCAN] PCAN connection closed.")

#####################################
# Continuous Joint Movement Control #
#####################################

def continuous_joint_movement():
    """
    Continuously move the Allegro Hand joints using sinusoidal motion.
    """
    hand = AllegroHandPCANController(pcan_channel=PCAN_USBBUS1)

    # Define joint movement parameters (amplitude, frequency, phase)
    movement_params = [
        (0.5, 0.2, 0), (0.3, 0.25, math.pi / 4), (0.4, 0.3, math.pi / 2), (0.2, 0.35, 3 * math.pi / 4),
        (0.5, 0.2, math.pi), (0.3, 0.25, 5 * math.pi / 4), (0.4, 0.3, 3 * math.pi / 2), (0.2, 0.35, 7 * math.pi / 4),
        (0.5, 0.2, 0), (0.3, 0.25, math.pi / 4), (0.4, 0.3, math.pi / 2), (0.2, 0.35, 3 * math.pi / 4),
        (0.5, 0.2, math.pi), (0.3, 0.25, 5 * math.pi / 4), (0.4, 0.3, 3 * math.pi / 2), (0.2, 0.35, 7 * math.pi / 4)
    ]

    start_time = time.time()
    update_rate = 1.0 / 50  # 50 Hz update rate

    try:
        while True:
            current_time = time.time() - start_time
            desired_positions = [
                amp * math.sin(2 * math.pi * freq * current_time + phase)
                for amp, freq, phase in movement_params
            ]

            hand.send_positions(desired_positions)
            time.sleep(update_rate)

    except KeyboardInterrupt:
        print("Stopped by user.")
        hand.close()

if __name__ == "__main__":
    continuous_joint_movement()
