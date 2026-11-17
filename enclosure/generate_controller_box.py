"""
Controller box (low-voltage side only) for Ilyasa's PSM2 build.

Holds: ESP32 DevKit (plugged into a 400-point half breadboard), 16x2 LCD with
I2C backpack (mounted behind a window in the lid), active buzzer (under a
grille in the lid). Cable entries for the 8-core cable from the mains box,
the USB power cable, and an optional DHT22 cable.

NOTHING ON THE 240V SIDE GOES IN THIS BOX. The PZEM, relay, fuse and mains
wiring stay in the bought ABS enclosure. Only low voltage (5V / 3.3V) here.

Every dimension below is a PARAMETER. The defaults are the common/typical
sizes for these modules, NOT measured from Ilyasa's actual parts. Measure
with calipers and edit the values marked MEASURE before printing, then run:

    pip install manifold3d trimesh
    python generate_controller_box.py

It writes controller_box_base.stl and controller_box_lid.stl next to itself.

Print notes:
  - Base prints open side up, lid prints face down (already oriented that
    way in the STL). No supports needed for either.
  - PETG or PLA both fine (low voltage only). 0.2 mm layers, 3 walls, 20% infill.
  - Lid screws: 4x M3 x 10 mm self-tapping (or machine screws with heat-set
    inserts, set POST_HOLE_D = 4.2 for typical M3 inserts).
  - LCD screws: 4x M3 x 6 mm self-tapping (or M2.5 if M3 is tight in the
    LCD's holes).
"""

from pathlib import Path

import numpy as np
import trimesh
from manifold3d import Manifold

SEG = 64  # circle smoothness

# ---------------- Box shell ----------------
INNER_L = 120.0   # inside length (x)
INNER_W = 85.0    # inside width (y)
INNER_H = 60.0    # inside height (floor top to lid underside)
WALL = 2.4        # 6 perimeters at 0.4 mm nozzle
FLOOR = 2.0
LID_T = 2.4
LIP_H = 3.0       # alignment lip on the lid that drops inside the walls
LIP_T = 1.6
FIT = 0.4         # clearance between lip and walls

# Corner screw posts (lid screws into these)
POST_D = 7.0
POST_INSET = 3.5          # post centre distance from inner walls
POST_HOLE_D = 2.6         # M3 self-tapping pilot. Use 4.2 for M3 heat-set inserts
POST_HOLE_DEPTH = 12.0
LID_SCREW_D = 3.4         # clearance hole in lid

# ---------------- Breadboard (floor) ----------------
# MEASURE: common 400-point half breadboard is about 82.5 x 54.5 x 8.5 mm.
# If the BOM breadboard is a full 830-point board (about 165 mm long) it will
# not fit this box. Use a 400-point one.
BB_L = 82.5
BB_W = 55.0
BB_RIM_H = 3.0
BB_RIM_T = 1.2
BB_FIT = 0.5
# Breadboard butts against the -x wall so the ESP32's USB end sits near the
# USB slot. Centred in y.

# ---------------- USB slot (-x wall) ----------------
# MEASURE: height of the ESP32 USB connector centre above the floor once the
# ESP32 is pushed into the breadboard. Typical: 8.5 (breadboard) + 2.5
# (header spacer) + 1.6 (PCB) + 1.5 (half connector) = about 14 mm.
USB_Z = 14.0
USB_Y = 0.0
USB_W = 14.0
USB_H = 10.0

# ---------------- Cable entries (+x wall) ----------------
CABLE8_D = 8.0     # MEASURE: 8-core cable outer diameter + about 1 mm
CABLE8_Y = 12.0
CABLE8_Z = 14.0
DHT_HOLE = True
DHT_HOLE_D = 5.0
DHT_Y = -22.0
DHT_Z = 14.0

# Zip-tie strain relief anchor for the 8-core cable (inside, on the floor)
ANCHOR_X_FROM_WALL = 14.0

# ---------------- Vents (both long walls) ----------------
VENT_W = 2.5
VENT_H = 20.0
VENT_Z0 = 25.0      # above floor
VENT_X = [-40 + 8 * i for i in range(11)]

# ---------------- LCD (lid) ----------------
# MEASURE all four of these on the actual LCD.
LCD_PCB_L = 80.0
LCD_PCB_W = 36.0
LCD_HOLE_DX = 75.0   # centre-to-centre, along the long side
LCD_HOLE_DY = 31.0
LCD_BEZEL_L = 71.2   # the metal frame around the display
LCD_BEZEL_W = 24.2
LCD_BEZEL_H = 7.0    # MEASURE: bezel height above the PCB front face
LCD_BEZEL_OFFSET_Y = 0.0  # MEASURE: bezel centre vs PCB centre (usually ~0)
LCD_WINDOW_FIT = 0.6
LCD_STANDOFF_D = 6.0
LCD_SCREW_PILOT = 2.5
# Placement in the lid (installed view, looking down at the lid).
# LCD sits right of centre to leave room on the left for the I2C backpack's
# 4-pin header and its jumper wires, which stick out past the PCB edge.
LCD_CX = 8.0
LCD_CY = 12.0

# ---------------- Buzzer (lid) ----------------
BUZZER_D = 12.0     # MEASURE
BUZZER_RING_H = 8.0
BUZZER_CX = 30.0
BUZZER_CY = -25.0
GRILLE_HOLE_D = 2.0

OUT_DIR = Path(__file__).resolve().parent


def cube(sx, sy, sz, x=0.0, y=0.0, z=0.0):
    """Axis-aligned box with its min corner at (x, y, z)."""
    return Manifold.cube((sx, sy, sz)).translate((x, y, z))


def ccube(sx, sy, sz, cx, cy, z0):
    """Box centred in x/y at (cx, cy), from z0 upward."""
    return cube(sx, sy, sz, cx - sx / 2, cy - sy / 2, z0)


def cyl(d, h, cx=0.0, cy=0.0, z0=0.0):
    return Manifold.cylinder(h, d / 2, d / 2, SEG).translate((cx, cy, z0))


def xcyl(d, length, y, z, x0):
    """Cylinder along +x starting at x0 (for holes through x walls)."""
    return Manifold.cylinder(length, d / 2, d / 2, SEG).rotate((0, 90, 0)).translate((x0, y, z))


def post_centres():
    px = INNER_L / 2 - POST_INSET
    py = INNER_W / 2 - POST_INSET
    return [(sx * px, sy * py) for sx in (-1, 1) for sy in (-1, 1)]


def build_base():
    ox, oy = INNER_L / 2 + WALL, INNER_W / 2 + WALL
    total_h = FLOOR + INNER_H
    body = cube(2 * ox, 2 * oy, total_h, -ox, -oy, 0)
    body = body - cube(INNER_L, INNER_W, INNER_H + 1, -INNER_L / 2, -INNER_W / 2, FLOOR)

    # Corner posts, fused into the corner walls
    for cx, cy in post_centres():
        sx, sy = np.sign(cx), np.sign(cy)
        post = cyl(POST_D, INNER_H, cx, cy, FLOOR)
        fill = cube(POST_INSET + 0.01, POST_INSET + 0.01, INNER_H,
                    cx if sx > 0 else cx - POST_INSET,
                    cy if sy > 0 else cy - POST_INSET, FLOOR)
        body = body + post + fill
        body = body - cyl(POST_HOLE_D, POST_HOLE_DEPTH + 1, cx, cy, total_h - POST_HOLE_DEPTH)

    # Breadboard rim (3 sides, open against the -x wall)
    bb_x0 = -INNER_L / 2
    bb_x1 = bb_x0 + BB_L + BB_FIT
    bb_y = BB_W / 2 + BB_FIT
    rim = cube(BB_RIM_T, 2 * bb_y + 2 * BB_RIM_T, BB_RIM_H, bb_x1, -bb_y - BB_RIM_T, FLOOR)
    rim = rim + cube(bb_x1 - bb_x0 + BB_RIM_T, BB_RIM_T, BB_RIM_H, bb_x0, bb_y, FLOOR)
    rim = rim + cube(bb_x1 - bb_x0 + BB_RIM_T, BB_RIM_T, BB_RIM_H, bb_x0, -bb_y - BB_RIM_T, FLOOR)
    body = body + rim

    # Zip-tie anchor: block with a tunnel along y
    ax = INNER_L / 2 - ANCHOR_X_FROM_WALL
    anchor = ccube(8, 10, 5.5, ax, CABLE8_Y, FLOOR)
    anchor = anchor - ccube(4.5, 12, 2.6, ax, CABLE8_Y, FLOOR - 0.01)
    body = body + anchor

    # USB slot, -x wall
    body = body - ccube(WALL + 2, USB_W, USB_H, -ox + (WALL + 2) / 2 - 1, USB_Y, FLOOR + USB_Z - USB_H / 2)

    # 8-core cable hole and DHT22 hole, +x wall
    body = body - xcyl(CABLE8_D, WALL + 2, CABLE8_Y, FLOOR + CABLE8_Z, INNER_L / 2 - 1)
    if DHT_HOLE:
        body = body - xcyl(DHT_HOLE_D, WALL + 2, DHT_Y, FLOOR + DHT_Z, INNER_L / 2 - 1)

    # Vents on both long walls
    for vx in VENT_X:
        for sy in (-1, 1):
            y0 = sy * (INNER_W / 2 + WALL / 2)
            body = body - ccube(VENT_W, WALL + 2, VENT_H, vx, y0, FLOOR + VENT_Z0)

    return body


def build_lid_installed():
    """Lid in installed orientation: plate top face at z = LID_T, features
    hang down into the box (negative z)."""
    ox, oy = INNER_L / 2 + WALL, INNER_W / 2 + WALL
    lid = cube(2 * ox, 2 * oy, LID_T, -ox, -oy, 0)

    # Alignment lip, notched around the corner posts
    lo_x, lo_y = INNER_L / 2 - FIT, INNER_W / 2 - FIT
    lip = cube(2 * lo_x, 2 * lo_y, LIP_H, -lo_x, -lo_y, -LIP_H)
    lip = lip - cube(2 * (lo_x - LIP_T), 2 * (lo_y - LIP_T), LIP_H + 2, -(lo_x - LIP_T), -(lo_y - LIP_T), -LIP_H - 1)
    for cx, cy in post_centres():
        notch = POST_D / 2 + POST_INSET + 1.5
        lip = lip - ccube(2 * notch, 2 * notch, LIP_H + 2, cx + np.sign(cx) * POST_INSET / 2,
                          cy + np.sign(cy) * POST_INSET / 2, -LIP_H - 1)
    lid = lid + lip

    # Lid screw holes
    for cx, cy in post_centres():
        lid = lid - cyl(LID_SCREW_D, LID_T + 2, cx, cy, -1)

    # LCD window (bezel pokes up through it, face flush with lid top)
    lid = lid - ccube(LCD_BEZEL_L + LCD_WINDOW_FIT, LCD_BEZEL_W + LCD_WINDOW_FIT, LID_T + 2,
                      LCD_CX, LCD_CY + LCD_BEZEL_OFFSET_Y, -1)

    # LCD standoffs under the lid
    standoff_h = max(LCD_BEZEL_H - LID_T, 1.0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            hx = LCD_CX + sx * LCD_HOLE_DX / 2
            hy = LCD_CY + sy * LCD_HOLE_DY / 2
            so = cyl(LCD_STANDOFF_D, standoff_h, hx, hy, -standoff_h)
            so = so - cyl(LCD_SCREW_PILOT, standoff_h + 1, hx, hy, -standoff_h - 0.5)
            lid = lid + so

    # Buzzer holder ring + grille
    ring = cyl(BUZZER_D + 3.2, BUZZER_RING_H, BUZZER_CX, BUZZER_CY, -BUZZER_RING_H)
    ring = ring - cyl(BUZZER_D + 0.3, BUZZER_RING_H + 1, BUZZER_CX, BUZZER_CY, -BUZZER_RING_H - 0.5)
    lid = lid + ring
    lid = lid - cyl(GRILLE_HOLE_D, LID_T + 2, BUZZER_CX, BUZZER_CY, -1)
    for k in range(6):
        a = np.deg2rad(60 * k)
        lid = lid - cyl(GRILLE_HOLE_D, LID_T + 2, BUZZER_CX + 3.5 * np.cos(a),
                        BUZZER_CY + 3.5 * np.sin(a), -1)
    return lid


def to_trimesh(m):
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=True)


def main():
    base = build_base()
    lid_installed = build_lid_installed()
    # Print orientation: face down on the bed
    lid_print = lid_installed.rotate((180, 0, 0))
    bb = lid_print.bounding_box()
    lid_print = lid_print.translate((0, 0, -bb[2]))

    base_tm = to_trimesh(base)
    lid_tm = to_trimesh(lid_print)
    base_tm.export(OUT_DIR / "controller_box_base.stl")
    lid_tm.export(OUT_DIR / "controller_box_lid.stl")

    for name, tm in (("base", base_tm), ("lid", lid_tm)):
        ext = tm.bounds[1] - tm.bounds[0]
        print(f"{name}: {ext[0]:.1f} x {ext[1]:.1f} x {ext[2]:.1f} mm, "
              f"watertight={tm.is_watertight}, volume={tm.volume / 1000:.1f} cm3")
    return base, lid_installed


if __name__ == "__main__":
    main()
