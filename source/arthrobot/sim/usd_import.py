"""Import a URDF into USD with Isaac Sim's URDF importer, rebuilding only when needed."""
import fcntl
import hashlib
from pathlib import Path
from typing import Callable


def import_urdf(urdf_path: Path, usd_path: Path, fix_base: bool, collision_from_visuals: bool) -> None:
    """Import ``urdf_path`` to ``usd_path``. Fixed joints are merged; mimic joints are not parsed
    (couplings are authored explicitly afterwards, see :mod:`arthrobot.sim.gripper_drive`)."""
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension('isaacsim.asset.importer.urdf')
    ok, config = omni.kit.commands.execute('URDFCreateImportConfig')
    assert ok
    config.set_merge_fixed_joints(True)
    config.set_fix_base(fix_base)
    config.set_import_inertia_tensor(True)
    config.set_distance_scale(1.)
    config.set_make_default_prim(True)
    config.set_create_physics_scene(False)
    config.set_collision_from_visuals(collision_from_visuals)
    config.set_convex_decomp(False)
    config.set_self_collision(True)
    config.set_parse_mimic(False)
    ok, _ = omni.kit.commands.execute('URDFParseAndImportFile', urdf_path=str(urdf_path),
                                      import_config=config, dest_path=str(usd_path))
    assert ok, f'URDF import failed: {urdf_path}'


def import_if_changed(urdf_path: Path, usd_path: Path, variant: str, configure: Callable[[Path], None],
                      rebuild: bool = False, **import_options) -> bool:
    """Import and configure ``usd_path`` unless it is already up to date; return True if rebuilt.

    ``configure(usd_path)`` authors the physics on a fresh import. A ``<usd>.revision``
    file records the URDF hash and ``variant`` label it was built from, and is written
    only after ``configure`` succeeds. A lock file serializes concurrent builders
    (for example a preview started next to a training run).
    """
    signature = hashlib.sha256(urdf_path.read_bytes()).hexdigest() + ':' + variant
    stamp = usd_path.with_suffix('.revision')
    usd_path.parent.mkdir(parents=True, exist_ok=True)
    with (usd_path.parent / '.import.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not rebuild and usd_path.exists() and stamp.exists() and stamp.read_text() == signature:
            return False
        stamp.unlink(missing_ok=True)
        import_urdf(urdf_path, usd_path, **import_options)
        configure(usd_path)
        stamp.write_text(signature)
        return True
