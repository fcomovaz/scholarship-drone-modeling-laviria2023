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

# for PID controller
from PID import *


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


import numpy as np
import matplotlib.pyplot as plt


def generate_trajectory(num_points, x_limits, y_limits, num_control_points=4):
    """
    Generate a closed smooth curve using Bézier curves.
    :param num_control_points: Number of control points for each Bézier curve
    :param num_points: Number of points to generate for each Bézier curve
    :return: A numpy array of shape (num_points, 2) containing the points of the curve
    """
    # Generate random control points within the specified limits
    control_points = np.random.uniform(
        low=[x_limits[0], y_limits[0]],
        high=[x_limits[1], y_limits[1]],
        size=(num_control_points, 2),
    )

    # Create a Bézier curve
    t = np.linspace(0, 1, num_points)
    curve_points = np.zeros((num_points, 2))
    for i in range(num_points):
        for j in range(num_control_points):
            curve_points[i] += (
                control_points[j]
                * (t[i] ** j)
                * ((1 - t[i]) ** (num_control_points - j - 1))
            )

    return curve_points


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


# ===========================================
# ========== SIMULATION PARAMETERS ==========
# ===========================================
duration = 5  # (seconds)
framerate = 30  # (Hz)
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
ball = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "ball")
model.opt.viscosity = 1
# pid controller
kp, ki, kd = 5, 0, 0.1
my_pid_x = PID(kp, ki, kd)
my_pid_y = PID(kp, ki, kd)
last_time = 0
last_error_x = 0
last_error_y = 0
integral_x = 0
# for path tracking
num_points = 500
x_limits = (-2, 1.5)
y_limits = (-1, 2)
trajectory = generate_trajectory(num_points, x_limits, y_limits)
# place the ball in the initial position
data.qpos[0] = trajectory[0, 0]
data.qpos[1] = trajectory[0, 1]


# ===========================================
# ======= CUSTOM RETURNED PARAMETERS ========
# ===========================================
# variables where you're gonna save the data
position_x = []
position_y = []
times = []
last_time = 0
not_done = True
mse_error = []

# print the initial position of the ball

# ===========================================
# ============ START SIMULATION =============
# ===========================================
for i in range(num_points):
    print(f"Progress: {i}/{num_points}", end="\r")
    # load the model to evaluate
    mj.mj_step(model, data)

    # =======================================
    # ====== CUSTOM CODE FOR SIMULATION =====
    # =======================================
    # track the trajectory
    goal_x, goal_y = trajectory[i]

    # get the position of the ball
    pos_x = data.geom_xpos[ball][0]
    pos_y = data.geom_xpos[ball][1]

    # make the PID for each point until get the goal
    euclidean_distance = np.sqrt((goal_x - pos_x) ** 2 + (goal_y - pos_y) ** 2)

    # if the point is reached then go to the next one
    mse_error.append([pos_x, pos_y])
    # if euclidean_distance < 1e-3:
    #     # print(f"Point ({goal_x}, {goal_y}) reached at ({pos_x}, {pos_y})")
    #     # mse_error.append([pos_x, pos_y])
    #     # i += 1
    # # if the last point is reached then stop the simulation
    # if i == num_points:
    #     not_done = False

    # get the error
    error_x = goal_x - pos_x
    error_y = goal_y - pos_y

    # get the time differences if is 0 then 1
    dt = data.time - last_time if data.time - last_time != 0 else 0

    # compute PID
    control_x = my_pid_x.compute(pos_x, goal_x, dt)
    control_y = my_pid_y.compute(pos_y, goal_y, dt)

    # update velocities
    data.qvel[0] = control_x
    data.qvel[1] = control_y

    # update the time
    last_time = data.time

    # append results
    position_x.append(pos_x)
    position_y.append(pos_y)
    times.append(data.time)

    # here goes frame append
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


# print the mse
print("MSE: ", mse(trajectory, mse_error))

# x_values, y_values = [i[0] for i in mse_error], [i[1] for i in mse_error]

# plot 2 subfigures, trajectory and trajectory made by the ball
fig, axs = plt.subplots(2, 1)
axs[0].plot(trajectory[:, 0], trajectory[:, 1])
axs[0].scatter(trajectory[0, 0], trajectory[0, 1], c="g", label="Initial point")
axs[0].scatter(trajectory[-1, 0], trajectory[-1, 1], c="b", label="Final point")
axs[0].set_title("Trajectory")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].grid(True)
axs[1].plot(position_x, position_y)
axs[1].scatter(position_x[0], position_y[0], c="g", label="Initial point")
axs[1].scatter(position_x[-1], position_y[-1], c="b", label="Final point")
axs[1].set_title("Trajectory made by the ball")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].grid(True)
# show the legend
axs[0].legend()
axs[1].legend()
plt.show()
