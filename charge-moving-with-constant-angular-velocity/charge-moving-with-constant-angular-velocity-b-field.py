import argparse
import time
import sys
sys.path.append("..")

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as mpla

from utils import *
from charge import *
from fields import *
from circle import *

charge = __import__("charge-moving-with-constant-angular-velocity")

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--animation",
    help="generate animation",
    action="store_true")
parser.add_argument("-f", "--frequency",
    help="number of rotations the charge makes during one second",
    type=float)
parser.add_argument("-r", "--radius",
    help="radius of the circle",
    type=float)
args = parser.parse_args()

range = 0.5e0 # 50 cm
npoints = 200 + 1

fps = 25
duration = 30 # in seconds
frames = fps * duration

if args.frequency is None or args.frequency <= 0.0:
    args.frequency = 1.0e8

if args.radius is None or args.radius > range:
    args.radius = 0.045

circle = Circle(args.radius, args.frequency)

if (circle.max_linear_velocity() >= c):
    print(f"Max charge's velocity {circle.max_linear_velocity():.2e} cannot be bigger than speed of light")
    sys.exit()

# Generate values for t
t_values = np.linspace(0.0, np.reciprocal(args.frequency), frames)

# Calculate the circle curve points
# circle_points = circle.parametric_equation(t_values)

dt = 10 * np.reciprocal(args.frequency) / frames
q = charge.ChargeMovingWithConstantAngularVelocity(circle.parametric_equation, dt)
fields = Fields([q])

X, Y, Z = np.meshgrid(
    np.linspace(-range, +range, npoints),
    np.linspace(-range, +range, npoints),
    np.array([0]))
grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis = 1)

fig, ax = plt.subplots(figsize=(8, 8))

t = dt * 4
scatter = ax.scatter(q.position(t)[0], q.position(t)[1], s=3, c="red", marker="o")

annotate_coords = coords(-0.45, +0.45, 0.00, -0.03)
frequency = ax.annotate(0, xy=annotate_coords.get(), color="white")
frequency.set_text(f"f = {args.frequency:.2e} Hz")

annotate_coords.inc()
radius = ax.annotate(0, xy=annotate_coords.get(), color="white")
radius.set_text(f"r = {args.radius:.2e} m")

annotate_coords.inc()
max_velocity = ax.annotate(0, xy=annotate_coords.get(), color="white")
max_velocity.set_text(f"v_max = {circle.max_linear_velocity():.2e} m/s")

if args.animation:
    annotate_coords.inc()
    time = ax.annotate(0, xy=annotate_coords.get(), color="white")
    time.set_text(f"t = {t:.2e} s")

B_field = fields.B(grid, t)
B_field_magnitude = np.array([np.linalg.norm(B) for B in B_field]).reshape((npoints, npoints))
print("B min: " + str(np.min(B_field_magnitude)))
print("B max: " + str(np.max(B_field_magnitude)))

im = ax.imshow(B_field_magnitude, origin="lower",
                extent=[-range, range, -range, range])
im.set_norm(mpl.colors.LogNorm(vmin=1e-20, vmax=1e-14))

plt.xticks([-range, -range/2, 0, range/2, range])
plt.yticks([-range, -range/2, 0, range/2, range])
plt.xlabel("$x$ [m]")
plt.ylabel("$y$ [m]")
colorbar = fig.colorbar(im, fraction=0.046, pad = 0.04)
colorbar.ax.set_ylabel("$|\\mathbf{B}|$ [T]", rotation=270, labelpad=12)

# trajectory_plot = ax.plot(circle_points[:, 0], circle_points[:, 1], linewidth=1, color="red")

def update(frame):
    text = "\rProcessing frame {0}/{1}".format(frame + 1, frames)
    sys.stdout.write(text)
    sys.stdout.flush()

    t = frame * dt
    scatter.set_offsets([q.position(t)[0], q.position(t)[1]])
    time.set_text(f"t = {t:.2e} s")

    B_field = fields.B(grid, t)
    B_field_magnitude = np.array([np.linalg.norm(B) for B in B_field]).reshape((npoints, npoints))
    im.set_data(B_field_magnitude)

if args.animation:
    animation = mpla.FuncAnimation(fig, update, interval = 1000 / fps, frames = frames)
    animation.save(
        f"charge-moving-with-constant-angular-velocity-b-field-{args.frequency:.2e}Hz.mp4",
        writer = mpla.FFMpegWriter(fps = fps))
else:
    plt.savefig(
        f"charge-moving-with-constant-angular-velocity-b-field-{args.frequency:.2e}Hz.png",
        format = "png", bbox_inches="tight")
    plt.show()
