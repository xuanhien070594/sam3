"""Test script for LCM listener and automated object tracking."""

import atexit
import lcm
from lcm_sys.lcm_types.lcm_rb_pose.lcmt_list_of_strings import lcmt_list_of_strings

import psutil
import subprocess
import time


active_processes = []


class ObjectsToTrackSubscriber:
  def __init__(self):    
    self.objects_to_track = []
    self.lc = lcm.LCM()
    self.subscription = self.lc.subscribe("OBJECTS_TO_TRACK", self.callback_)
    self.addressed_latest_msg = False
  
  def callback_(self, channel, data):
    msg = lcmt_list_of_strings.decode(data)
    self.objects_to_track = msg.strings
    self.addressed_latest_msg = False

  def run(self):
    self.lc.handle_timeout(100)

  def get_objects_to_track(self):
    self.addressed_latest_msg = True
    return self.objects_to_track
  

def kill_process(process):
  try:
    parent = psutil.Process(process.pid)
    for child in parent.children(recursive=True):
      child.kill()
    parent.kill()
  except psutil.NoSuchProcess:
    pass

def kill_all_processes():
  for process in active_processes:
    kill_process(process)
  active_processes.clear()

atexit.register(kill_all_processes)


if __name__ == "__main__":
  subscriber = ObjectsToTrackSubscriber()
  while True:
    subscriber.run()
    if not subscriber.addressed_latest_msg:
      object_names = subscriber.get_objects_to_track()
      print(f'Starting to track objects: {object_names}')
      kill_all_processes()
      if len(object_names) > 0:
        f = open("memory_output.txt", "w")
        process = subprocess.Popen(
          ["python", "camera_memory.py"], stdout=f, stderr=f)
        active_processes.append(process)
      time.sleep(0.5)
      for i, name in enumerate(object_names):
        f = open(f"tracking_output_{i+1}.txt", "w")
        process = subprocess.Popen(
          ["python", "fpTracking_share3.py", f"--object_name={name}"],
          stdout=f, stderr=f)
        active_processes.append(process)

    for i, process in enumerate(active_processes):
      if process.poll() is not None:
        extra_info = subscriber.get_objects_to_track()[i-1] if i > 0 else \
          "Camera memory"
        print(f"Subprocess {i} has exited: {extra_info}")
        print("Terminating all subprocesses.")
        kill_all_processes()
        break

