import argparse
import time
import sys
sys.path.append("..")

import numpy as np
import scipy as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as mpla

from utils import *
from charge import *
from fields import *

charge = __import__("charge-moving-with-constant-velocity")

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--animation",
    help="generate animation",
    action="store_true")
parser.add_argument("-v", "--velocity",
    help="velocity of the particle expressed as a fraction of the speed of light",
    type=float)
args = parser.parse_args()

range = 0.5e0 # 50 cm
npoints = 200 + 1

fps = 25
duration = 10 # in seconds
frames = fps * duration

if args.animation:
    r0x = -0.425e0 # -42.5 cm
else:
    r0x = +0.025e0 # +2.5 cm

if args.velocity is None or args.velocity <= 0.0 or args.velocity >= 1.0:
    args.velocity = 0.01

r0 = np.array([r0x, 0.025, 0.0])
vx = args.velocity * c
v = np.array([vx, 0, 0])
q = charge.ChargeMovingWithConstantVelocity(r0, v)
fields = Fields([q])

X, Y, Z = np.meshgrid(
    np.linspace(-range, +range, npoints),
    np.linspace(-range, +range, npoints),
    np.array([0]))
grid = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis = 1)

fig, ax = plt.subplots(figsize=(8, 8))

t = 0
scatter = ax.scatter(q.position(t)[0], q.position(t)[1], s=3, c="red", marker="o")

annotate_coords = coords(-0.45, +0.45, 0.00, -0.03)
velocity = ax.annotate(0, xy=annotate_coords.get(), color="white")
velocity.set_text(f"v = {args.velocity:.2f} * c")

if args.animation:
    annotate_coords.inc()
    time = ax.annotate(0, xy=annotate_coords.get(), color="white")
    time.set_text(f"t = {t * 1e9:.3f} ns")

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
colorbar = fig.colorbar(im, fraction=0.046, pad=0.04)
colorbar.ax.set_ylabel("$|\\mathbf{B}|$ [T]", rotation=270, labelpad=12)

dt = (2 * abs(r0x) / frames) / vx
def update(frame):
    text = "\rProcessing frame {0}/{1}".format(frame + 1, frames)
    sys.stdout.write(text)
    sys.stdout.flush()

    t = frame * dt
    scatter.set_offsets([q.position(t)[0], q.position(t)[1]])
    time.set_text(f"t = {t * 1e9:.3f} ns")

    B_field = fields.B(grid, t)
    B_field_magnitude = np.array([np.linalg.norm(B) for B in B_field]).reshape((npoints, npoints))
    im.set_data(B_field_magnitude)

if args.animation:
    animation = mpla.FuncAnimation(fig, update, interval = 1000 / fps, frames = frames)
    animation.save(
        f"charge-moving-with-constant-velocity-b-field-v{args.velocity:.2f}c.mp4",
        writer = mpla.FFMpegWriter(fps = fps))
else:
    plt.savefig(
        f"charge-moving-with-constant-velocity-b-field-v{args.velocity:.2f}c.png",
        format = "png", bbox_inches="tight")
    plt.show()
