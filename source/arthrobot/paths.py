"""Repository locations.

ArthroBot is used from a source checkout (``pip install -e .``). Generated files
(URDF/USD builds, training runs) go under ``build/`` and ``logs/`` at the
repository root and are never written into ``source/``.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / 'source'
ASSETS_DIR = SOURCE_DIR / 'arthrobot_assets'
DATA_DIR = SOURCE_DIR / 'arthrobot' / 'data'
CHECKPOINTS_DIR = REPO_ROOT / 'checkpoints'
BUILD_DIR = REPO_ROOT / 'build'
LOGS_DIR = REPO_ROOT / 'logs'


def build_dir(name: str) -> Path:
    """Return (and create) ``build/<name>`` for generated robot files."""
    path = BUILD_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
