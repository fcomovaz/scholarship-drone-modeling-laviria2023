# mujoco library is required to run this code
import mujoco
from mujoco.glfw import glfw

# for file handling
import os

# for time handling
import time

# for mathematical operations
import numpy as np
from math import radians

# for statistical operations
from sklearn.metrics import mean_squared_error as mse

# for visualization
import matplotlib.pyplot as plt

# for creating gif
from PIL import Image


def create_gif(frames, filename, duration=1):
    # frames: List of PIL Image objects
    # filename: Name of the output GIF file
    # duration: Duration between frames in milliseconds (default: 100 ms)

    # Save the frames as a GIF
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,  # Loop forever
    )


xml_path = "model.xml"
# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath
model = mujoco.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

duration = 10  # (seconds)
framerate = 30  # (Hz)
timestep = 0.1e-3 # (seconds)

cam = mujoco.MjvCamera()  # Abstract camera
opt = mujoco.MjvOption()

# initialize visualization data structures
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)

frames = []
results = []
times = []

# get the time in seconds
start_time = time.time()

# change timestep
model.opt.timestep = timestep

mujoco.mj_resetData(model, data)

# start model parameters
initial_angle = radians(10)
swing = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "swing")
data.joint("swing").qpos = initial_angle
# data.qpos[swing] = initial_angle

while data.time < duration:
    mujoco.mj_step(model, data)

    if len(frames) < data.time * framerate:
        # visualize all model
        renderer.update_scene(data, scene_option=scene_option)

        pixels = renderer.render()
        frames.append(pixels)

    results.append(data.qpos[swing])
    times.append(data.time)


# Example usage
frames = [Image.fromarray(frame) for frame in frames]
output_filename = "output.gif"
create_gif(frames, output_filename, 1 / 120)

# calculate the total time
end_time = time.time()
total_time = end_time - start_time
print(f"Real Time: {total_time}")
print(f"Simu Time: {duration}")
print(f"Time Perc: {total_time/duration*100}%")

# plot the results
plt.plot(times, results)
plt.xlabel("time (s)")
plt.ylabel("swing angle (rad)")
plt.title("Swing angle vs time")
plt.grid()
plt.show()
