import math
from pathlib import Path

# Basic geometry helpers ----------------------------------------------------

def normal(a, b, c):
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return (nx / length, ny / length, nz / length)


def tri(a, b, c):
    return normal(a, b, c), (a, b, c)


def add_quad(triangles, a, b, c, d):
    triangles.append(tri(a, b, c))
    triangles.append(tri(a, c, d))


def add_polygon_cap(triangles, points, reverse=False):
    # fan around centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    cz = sum(p[2] for p in points) / len(points)
    center = (cx, cy, cz)
    ordered = points if not reverse else list(reversed(points))
    for i in range(len(ordered)):
        a = ordered[i]
        b = ordered[(i + 1) % len(ordered)]
        if reverse:
            triangles.append(tri(center, a, b))
        else:
            triangles.append(tri(center, b, a))


def add_box(triangles, min_corner, max_corner):
    x0, y0, z0 = min_corner
    x1, y1, z1 = max_corner
    v = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    # bottom, top, sides
    add_quad(triangles, v[0], v[1], v[2], v[3])  # bottom
    add_quad(triangles, v[4], v[5], v[6], v[7])  # top
    add_quad(triangles, v[0], v[4], v[5], v[1])  # front
    add_quad(triangles, v[1], v[5], v[6], v[2])  # right
    add_quad(triangles, v[2], v[6], v[7], v[3])  # back
    add_quad(triangles, v[3], v[7], v[4], v[0])  # left


def add_cylinder_y(triangles, center, radius, half_width, segments=24):
    cx, cy, cz = center
    top_y = cy + half_width
    bottom_y = cy - half_width
    ring_top = []
    ring_bottom = []
    for i in range(segments):
        theta = 2 * math.pi * i / segments
        x = cx + radius * math.cos(theta)
        z = cz + radius * math.sin(theta)
        ring_top.append((x, top_y, z))
        ring_bottom.append((x, bottom_y, z))

    for i in range(segments):
        a_top = ring_top[i]
        b_top = ring_top[(i + 1) % segments]
        a_bottom = ring_bottom[i]
        b_bottom = ring_bottom[(i + 1) % segments]
        add_quad(triangles, a_top, b_top, b_bottom, a_bottom)

    # caps
    add_polygon_cap(triangles, list(reversed(ring_top)))  # outward normal +y
    add_polygon_cap(triangles, ring_bottom, reverse=True)  # outward normal -y


# Build trailer body --------------------------------------------------------

BASE_WIDTH = 2.4
BASE_HEIGHT = 1.05
ROOF_HEIGHT = 1.45
TOP_STEPS = 14

cross_section = [(-BASE_WIDTH / 2, 0.0), (-BASE_WIDTH / 2, BASE_HEIGHT)]
for i in range(1, TOP_STEPS):
    angle = math.pi - i * math.pi / TOP_STEPS
    y = (BASE_WIDTH / 2) * math.cos(angle)
    z = BASE_HEIGHT + ROOF_HEIGHT * math.sin(angle)
    cross_section.append((y, z))
cross_section.append((BASE_WIDTH / 2, BASE_HEIGHT))
cross_section.append((BASE_WIDTH / 2, 0.0))
cross_section.append((-BASE_WIDTH / 2, 0.0))

sections = [
    (-0.55, 0.55, 0.62),
    (-0.15, 0.85, 0.88),
    (0.35, 0.98, 1.02),
    (1.8, 1.08, 1.05),
    (3.6, 1.1, 1.05),
    (5.4, 1.08, 1.03),
    (6.8, 1.0, 1.0),
    (7.3, 0.86, 0.95),
    (7.8, 0.62, 0.82),
]

rings = []
for x, width_scale, height_scale in sections:
    ring = []
    for y, z in cross_section:
        scaled_y = y * width_scale
        scaled_z = z * height_scale if z != 0.0 else 0.0
        ring.append((x, scaled_y, scaled_z))
    rings.append(ring)

triangles = []

for idx in range(len(rings) - 1):
    curr = rings[idx]
    nxt = rings[idx + 1]
    for i in range(len(curr) - 1):
        add_quad(triangles, curr[i], curr[i + 1], nxt[i + 1], nxt[i])

# caps
add_polygon_cap(triangles, rings[-1])  # rear cap
add_polygon_cap(triangles, rings[0], reverse=True)  # front cap

# Hitch tongue --------------------------------------------------------------
hitch_height = 0.25
hitch_width = 0.22
hitch_length = 0.9
hitch_base_x = sections[0][0]
hitch_tip_x = hitch_base_x - hitch_length
add_box(
    triangles,
    (hitch_tip_x, -hitch_width / 2, 0.12),
    (hitch_base_x, hitch_width / 2, 0.12 + hitch_height),
)
# Coupler
coupler_top = (hitch_tip_x - 0.12, 0.0, 0.38)
coupler_base = [
    (hitch_tip_x, -hitch_width / 2, 0.12 + hitch_height),
    (hitch_tip_x, hitch_width / 2, 0.12 + hitch_height),
    (hitch_tip_x, hitch_width / 2, 0.12),
    (hitch_tip_x, -hitch_width / 2, 0.12),
]
add_polygon_cap(triangles, coupler_base, reverse=True)
for i in range(len(coupler_base)):
    a = coupler_base[i]
    b = coupler_base[(i + 1) % len(coupler_base)]
    triangles.append(tri(a, b, coupler_top))

# Wheel wells ---------------------------------------------------------------
wheel_centers = [2.7, 4.35]
wheel_offset = BASE_WIDTH * sections[3][1] / 2 + 0.12
wheel_radius = 0.42
wheel_half_width = 0.11
for x in wheel_centers:
    add_cylinder_y(triangles, (x, wheel_offset, wheel_radius + 0.15), wheel_radius, wheel_half_width)
    add_cylinder_y(triangles, (x, -wheel_offset, wheel_radius + 0.15), wheel_radius, wheel_half_width)

# Window trims --------------------------------------------------------------
window_specs = [
    # (center_x, center_z, width, height)
    (1.6, 1.65, 1.8, 0.8),
    (3.9, 1.7, 1.6, 0.75),
    (5.8, 1.65, 1.2, 0.7),
]
trim_depth = 0.06
for cx, cz, width, height in window_specs:
    for side in (1, -1):
        y_center = side * (BASE_WIDTH * sections[3][1] / 2 - 0.04)
        y0 = y_center - trim_depth * side
        y1 = y_center + trim_depth * side
        z0 = cz - height / 2
        z1 = cz + height / 2
        x0 = cx - width / 2
        x1 = cx + width / 2
        add_box(triangles, (x0, min(y0, y1), z0), (x1, max(y0, y1), z1))

# Door outline --------------------------------------------------------------
door_bottom = 0.12
door_height = 1.85
door_width = 0.8
door_x0 = 2.1
for offset in (-0.015, 0.015):
    add_box(
        triangles,
        (door_x0, -BASE_WIDTH * 0.52 + offset, door_bottom),
        (door_x0 + door_width, -BASE_WIDTH * 0.52 + offset + 0.03, door_bottom + door_height),
    )

# Roof details --------------------------------------------------------------
add_box(
    triangles,
    (1.0, -0.25, max(p[2] for p in rings[3]) - 0.15),
    (2.6, 0.25, max(p[2] for p in rings[3]) - 0.02),
)
add_box(
    triangles,
    (4.0, -0.22, max(p[2] for p in rings[4]) - 0.18),
    (5.4, 0.22, max(p[2] for p in rings[4]) - 0.05),
)

# Stabiliser jacks ----------------------------------------------------------
jack_width = 0.18
jack_height = 0.25
jack_depth = 0.14
jack_positions = [(0.4, BASE_WIDTH * 0.55), (6.6, BASE_WIDTH * 0.55)]
for x, y_mag in jack_positions:
    for side in (1, -1):
        y0 = side * (y_mag + 0.08)
        y1 = y0 + side * jack_depth
        add_box(triangles, (x - jack_width / 2, min(y0, y1), 0), (x + jack_width / 2, max(y0, y1), jack_height))

# Write STL -----------------------------------------------------------------
out_path = Path(__file__).resolve().parents[1] / "src" / "rev_cam" / "static" / "models" / "airstream_trailer.stl"
with out_path.open("w", encoding="ascii") as fp:
    fp.write("solid airstream_trailer\n")
    for n, (a, b, c) in triangles:
        fp.write(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        fp.write("    outer loop\n")
        fp.write(f"      vertex {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}\n")
        fp.write(f"      vertex {b[0]:.6f} {b[1]:.6f} {b[2]:.6f}\n")
        fp.write(f"      vertex {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        fp.write("    endloop\n")
        fp.write("  endfacet\n")
    fp.write("endsolid airstream_trailer\n")

print(f"Wrote {len(triangles)} triangles to {out_path}")
