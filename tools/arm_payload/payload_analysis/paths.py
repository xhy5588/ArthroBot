"""Tool folder and the ArthroBot arm model it screens.

The arm is ArthroBot's own build (``build/arm/arm.urdf`` and ``model_report.json``,
made from the CAD export by ``arthrobot_assets.arm.build``), mounted as in the arm
reach task (``arthrobot_tasks/arm_reach/mount.py``).
"""
import importlib.util
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent  # tools/arm_payload
REPO_DIR = PROJECT_DIR.parent.parent
OUTPUT_DIR = PROJECT_DIR / 'workspace_payload'

# The tool runs from a checkout; use its packages when ArthroBot is not installed.
if importlib.util.find_spec('arthrobot') is None:
    sys.path.insert(0, str(REPO_DIR / 'source'))


def arm_urdf() -> Path:
    """``build/arm/arm.urdf``, built from the CAD export if missing (as arm_planning does)."""
    from arthrobot import paths
    path = paths.BUILD_DIR / 'arm' / 'arm.urdf'
    if not path.exists():
        from arthrobot_assets.arm.build import write
        path, _ = write()
    return Path(path)


def arm_report() -> Path:
    """The arm build report (part masses and visual manifest) next to the URDF."""
    return arm_urdf().with_name('model_report.json')
