"""The test scene for planning, shared by the planner and the simulator. No Isaac or cuRobo imports.

Everything is in the training mount frame (origin at the base bottom on the
joint 1 axis, x forward, z up). A table at the base's height with a square hole
around the fixed base (the base cannot move, so its contact with the table does
not matter; the moving links must still miss the base itself), and a wall
standing on the table in front of the arm, between a left and a right target area.
"""
TABLE_TOP = 0.0
TABLE_THICKNESS = 0.02
TABLE_X, TABLE_Y = (-0.25, 0.55), (-0.45, 0.45)
BASE_HOLE = 0.07  # Half-width of the hole around the base; the base footprint is about +-0.045 m.
WALL_CENTER, WALL_DIMS = (0.27, 0.0, 0.08), (0.14, 0.03, 0.16)

# Targets: top-down grasps, alternating left and right of the wall.
# The jaw tips are 47 mm past the TCP, so straight-down grasps need the TCP at least ~5 cm above the table.
TARGET_X, TARGET_ABS_Y, TARGET_Z = (0.20, 0.32), (0.10, 0.20), (0.06, 0.10)
HOME_TCP = (0.18, 0.0, 0.25)  # Start above the wall, pointing down (highest reachable z is used).


def cuboids():
    """name -> (center xyz, dims xyz) of every obstacle."""
    z = TABLE_TOP - TABLE_THICKNESS / 2
    (x0, x1), (y0, y1), h = TABLE_X, TABLE_Y, BASE_HOLE
    box = lambda xa, xb, ya, yb: (((xa + xb) / 2, (ya + yb) / 2, z), (xb - xa, yb - ya, TABLE_THICKNESS))
    return {
        'table_front': box(h, x1, y0, y1),
        'table_back': box(x0, -h, y0, y1),
        'table_left': box(-h, h, h, y1),
        'table_right': box(-h, h, y0, -h),
        'wall': (WALL_CENTER, WALL_DIMS),
    }


def curobo_scene(margin=0.0):
    """The obstacles as a cuRobo scene dict (pose: x, y, z, qw, qx, qy, qz), each grown by margin (m) on every
    side. Growing the obstacles keeps the arm farther from them without inflating its own spheres, which would
    also tighten its self-collision checks."""
    return {'cuboid': {name: {'dims': [d + 2 * margin for d in dims], 'pose': [*center, 1., 0., 0., 0.]}
                       for name, (center, dims) in cuboids().items()}}
