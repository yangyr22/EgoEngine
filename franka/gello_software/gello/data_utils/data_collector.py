import os
import threading
import time
from collections import defaultdict
import datetime
import h5py
import json
import numpy as np

class DataCollector():

    def __init__(self, save_dir, flush_freq = 200):

        self.total_tasks = 0
        self.transition_count = 0
        self.chunk_count = 0
        self.flush_freq =flush_freq

        # hdf5 file stuff
        self.f = None
        self.f_grp = None
        self.ep_data_grp = None

        self.buffer = None

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir

        self._async_flusher = AsyncFlusher(self._flush_buffer_to_disk)
        self._async_flusher.start()


    def start_episode(self, env_args):
        # First we will create the file for this episode
        dt_time = datetime.datetime.now()
        self.hdf5_path = os.path.join(self.save_dir,
                                 dt_time.strftime("%m%d_%H%M%S") + ".hdf5"
                                 )

        self.maybe_acquire_flush_lock()
        self.f = h5py.File(self.hdf5_path, "w")
        self.f_grp = self.f.create_group("data")
        self.ep_data_grp = self.f_grp.create_group("demo_0")
        self.ep_data_grp.attrs["env_args"] = json.dumps(env_args)
        self.maybe_release_flush_lock()
        print(f"Data Collector: recording data to {self.hdf5_path}")

        # Reset the buffer and increment counters
        self.buffer = None
        self.total_tasks += 1

    def maybe_acquire_flush_lock(self):
        self._async_flusher.flush_lock.acquire()

    def maybe_release_flush_lock(self):
        self._async_flusher.flush_lock.release()


    def collect(self, data):
        if self.buffer is None:
            self.buffer = defaultdict(list)
        self.transition_count += 1
        self.buffer['obs'].append(data['obs'])
        self.buffer['act_abs'].append(data['act_abs'])
        self.buffer['act_delta'].append(data['act_delta'])
        self.buffer['control_enabled'].append(data['control_enabled'])

        if self.transition_count % self.flush_freq == 0:
            self._async_flusher.flush(self.buffer)
            self.buffer = None


    def _flush_buffer_to_disk(self, buffer):
        print('*'*50)
        print("{}: flushing {} transitions to disk".format(self.__class__.__name__, self.transition_count))
        print('*'*50)


        obs_buffer = buffer["obs"]
        act_delta_buffer = buffer["act_delta"]
        act_abs_buffer = buffer["act_abs"]
        enabled_buffer = buffer["control_enabled"]

        # Flush obs
        for k in obs_buffer[0]:
            obs_chunk_to_flush = np.stack([obs_buffer[i][k] for i in range(len(obs_buffer))], 0)
            self.ep_data_grp.create_dataset(f"chunk_{self.chunk_count}/obs/{k}", data=obs_chunk_to_flush)

        # Flush actions
        act_chunk_to_flush = np.stack(act_delta_buffer)
        self.ep_data_grp.create_dataset(f"chunk_{self.chunk_count}/action", data=act_chunk_to_flush)
        act_chunk_to_flush = np.stack(act_abs_buffer)
        self.ep_data_grp.create_dataset(f"chunk_{self.chunk_count}/action_absolute", data=act_chunk_to_flush)

        # Flush teleop state (i.e. is the user actively controlling)
        enabled_chunk_to_flush = np.stack(enabled_buffer)
        self.ep_data_grp.create_dataset(f"chunk_{self.chunk_count}/control_enabled", data=enabled_chunk_to_flush)

        self.chunk_count += 1

        # Reset the buffer
        # self.buffer - defaultdict(list)

    def end_episode(self, success=True):
        """
        Cleanup and end the episode
        Args:
            success:

        Returns:

        """
        print("End episode called")
        # First flush any remaining data
        if self.buffer is not None:
            self._async_flusher.flush(self.buffer)

        # Wait for flush to finish
        # TODO this is a hack using time.sleep to allow the flushing thread to flush the buffer to the disk. Fix this with synchro and conditions
        time.sleep(0.05)
        self.maybe_acquire_flush_lock()
        self.buffer = defaultdict(list)



        if self.f is not None:
            fname = self.f.name
            self.f.close()
            # TODO: maybe set from config whether or not to delete unsuccessful demos
            if not success:
                # delete this unsuccessful demonstration
                print("DataCollector: removing unsuccessful demonstration at {}".format(self.hdf5_path))
                os.remove(self.hdf5_path)

        # Reset vars
        self.f = None
        self.f_grp = None
        self.ep_data_grp = None

        self.transition_count = 0
        self.chunk_count = 0

        self.maybe_release_flush_lock()
        if not self._async_flusher.get_running_status():
            raise Exception("Async flusher is not running, stop server!")


    def close(self):
        """
        Close the data collector.
        """
        if self.f is not None:
            self.f.close()

        self._async_flusher.stop()
        self._async_flusher.join()



class AsyncFlusher(threading.Thread):
    def __init__(self, flush_function):
        super(AsyncFlusher, self).__init__()
        self._should_flush_condition = threading.Condition()
        self._running = True
        self._running_lock = threading.Lock()

        self._buffer_to_flush = None
        self._flush_function = flush_function

        self.flush_lock = threading.Lock()

    def flush(self, buffer_to_flush):
        """
        NOTE: this function should never get blocked!!
        """
        with self.flush_lock:
            self._buffer_to_flush = buffer_to_flush
        self._should_flush_condition.acquire()
        self._should_flush_condition.notify()
        self._should_flush_condition.release()

    def _wait_on_flush(self):
        """
        NOTE: we don't need a loop to check state here since only 1 thread is being notified.
        """
        self._should_flush_condition.acquire()
        self._should_flush_condition.wait()
        self._should_flush_condition.release()

    def _set_running_status(self, status):
        """
        Set the status of the server. Use a lock to guard this.
        """
        with self._running_lock:
            self._running = status

    def get_running_status(self):
        """
        Check if the server is still running. Use a lock to guard this.
        :return: True if server is still running.
        """
        with self._running_lock:
            status = self._running
        return status

    def run(self):
        """
        The main loop for this thread.
        """
        print("\n{}: thread starting ...".format(self.__class__.__name__))
        self._set_running_status(True)
        # loop until connection is terminated by the main data collector
        while True:
            # wait until signaled to collect some shit
            self._wait_on_flush()

            if not self.get_running_status():
                print("\n{}: Getting running status, stop the thread!".format(self.__class__.__name__))
                break

            if self._buffer_to_flush is not None:
                with self.flush_lock:
                    try:
                        self._flush_function(self._buffer_to_flush)
                        self._buffer_to_flush = None
                    except:
                        self._set_running_status(False)

    def stop(self):
        """
        Called by main thread when it is time to terminate.
        """
        print("\n{}: Stopping!".format(self.__class__.__name__))
        self._set_running_status(False)
        # important: make sure main loop gets unblocked so it can terminate gracefully
        self.flush(buffer_to_flush=None)


