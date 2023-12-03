# ===========================================
# ========= MUJOCO TEMPLATE SCRIPT ==========
# ===========================================
# mujoco library is required to run this code
import mujoco as mj
from mujoco.glfw import glfw

# for file handling
import os

# for mathematical operations
import numpy as np
from math import radians

# for statistical operations
from sklearn.metrics import mean_squared_error as mse

# for visualization
import matplotlib.pyplot as plt

# for creating gif
from PIL import Image

# for pid
import PID


# ===========================================
# ============ REQUIRED FUNCTIONS ===========
# ===========================================
def create_gif(frames, filename, duration=10, loopit=0):
    """
    Create gif from an array of frames.

    :param frames: List of PIL Image objects
    :param filename: Name of the output GIF file
    :param duration: duration between frames in milliseconds (10ms default)

    """

    frames = [Image.fromarray(frame) for frame in frames]

    # Save the frames as a GIF
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loopit,  # Loop forever
    )


# ===========================================
# ========= MODEL & DATA PARAMETERS =========
# ===========================================
# name of the xml file
xml_path = "model.xml"
# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath
# MuJoCo model
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)  # data from the model
renderer = mj.Renderer(model, 400, 600)  # renderer of the model
model.opt.integrator = mj.mjtIntegrator.mjINT_RK4
model.opt.solver = mj.mjtSolver.mjSOL_PGS


# ===========================================
# ========== SIMULATION PARAMETERS ==========
# ===========================================
duration = 100  # (seconds)
framerate = 5  # (Hz)
frames = []  # frames of the simulation
gif_creation = True  # create a gif of the simulation


# ===========================================
# ========= INITIALIZE SIMULATION ===========
# ===========================================
# reset the data from the model
mj.mj_resetData(model, data)
# enable joint visualization option:
scene_option = mj.MjvOption()
mj.mjv_defaultOption(scene_option)
scene_option.flags[mj.mjtVisFlag.mjVIS_JOINT] = True


# ===========================================
# ========= START CUSTOM PARAMETERS =========
# ===========================================
# initialization of the model
myPIDz = PID.PID(1, 7, .6)
z_goal = 0.8
last_error = 0
last_time = 0
ball = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "bola")



# ===========================================
# ======= CUSTOM RETURNED PARAMETERS ========
# ===========================================
# variables where you're gonna save the data
z = []
t = []
print(len(data.qvel))

# ===========================================
# ============ START SIMULATION =============
# ===========================================
while data.time < duration:
    # load the model to evaluate
    mj.mj_step(model, data)

    # =======================================
    # ====== CUSTOM CODE FOR SIMULATION =====
    # =======================================
    # get the current position
    z_current = data.geom_xpos[ball][2]

    # get the time
    time = data.time
    dt = time - last_time
    last_time = time

    # compute the control
    control_z = myPIDz.compute(z_current, z_goal, dt)

    # apply the control to the ball as velocities
    data.qvel[2] = control_z

    z.append(z_current)
    t.append(time)

    # check if the gif is true
    if gif_creation:
        # render the scene to create a gif
        if len(frames) < data.time * framerate:
            # visualize all model
            renderer.update_scene(data, scene_option=scene_option)
            # create and append the pixels of the scene
            pixels = renderer.render()
            frames.append(pixels)


# ===========================================
# =============== GIF CREATION ==============
# ===========================================
# check if the gif is true
if gif_creation:
    output_filename = "output.gif"
    create_gif(frames, output_filename, 1 // framerate)


# plot
plt.plot(t, z)
plt.show()
