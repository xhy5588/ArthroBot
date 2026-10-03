"""Process-local Vulkan workaround for Isaac Sim 5.1 on NVIDIA 595.x drivers.

Isaac Sim 5.1's RTX renderer needs the reported maximum memory allocation capped
on these drivers (https://github.com/isaac-sim/IsaacSim/issues/568). This loads
the Khronos Vulkan Profiles layer for the current process only; the driver and
the global Vulkan configuration are not changed.

Call :func:`enable` before starting the Isaac Sim app, whenever it will render
(GUI, cameras or videos). It does nothing on other driver versions. Build the
layer once with ``scripts/setup_rtx_compat.sh``. For scripts that do not call it
(such as the official ``scripts/rsl_rl/play.py``), wrap the command:

    python -m arthrobot.sim.rtx_compat python scripts/rsl_rl/play.py --task ...

Environment variables:

- ``ARTHROBOT_VULKAN_PROFILES``: layer checkout (default ``~/.cache/arthrobot-vulkan-profiles``)
- ``ARTHROBOT_RTX_COMPAT_DISABLE=1``: skip the workaround
"""
import json
import os
from pathlib import Path
import subprocess

AFFECTED_DRIVER_PREFIX = '595.'
PROFILE_NAME = 'VP_LOCAL_arthrobot_compat'
MAX_ALLOCATION_BYTES = 4292870144   # 4 GiB minus 2 MiB


def layer_dir() -> Path:
    default = Path.home() / '.cache/arthrobot-vulkan-profiles'
    return Path(os.environ.get('ARTHROBOT_VULKAN_PROFILES', default))


def needs_workaround() -> bool:
    if os.environ.get('ARTHROBOT_RTX_COMPAT_DISABLE') == '1':
        return False
    try:
        versions = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                  capture_output=True, text=True, check=True).stdout.split()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return any(version.startswith(AFFECTED_DRIVER_PREFIX) for version in versions)


def enable() -> None:
    if not needs_workaround():
        return
    cache = layer_dir()
    library = cache / 'build/layer/libVkLayer_khronos_profiles.so'
    if not library.is_file():
        raise RuntimeError(f'Build the Vulkan profiles layer first (scripts/setup_rtx_compat.sh); missing {library}')
    profile_dir, manifest_dir = cache / 'profile', cache / 'xdg/vulkan/implicit_layer.d'
    profile_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        '$schema': 'https://schema.khronos.org/vulkan/profiles-0.8-latest.json#',
        'capabilities': {'allocation_limit': {'properties': {
            'VkPhysicalDeviceMaintenance3Properties': {'maxMemoryAllocationSize': MAX_ALLOCATION_BYTES}}}},
        'profiles': {PROFILE_NAME: {
            'version': 1, 'api-version': '1.1.0', 'label': 'ArthroBot renderer compatibility',
            'description': 'Finite allocation limit for Isaac Sim 5.1 on NVIDIA 595.x.',
            'contributors': {'local': {}},
            'history': [{'revision': 1, 'date': '2026-09-16', 'author': 'local',
                         'comment': 'Local compatibility profile.'}],
            'capabilities': ['allocation_limit']}},
    }
    (profile_dir / 'arthrobot.json').write_text(json.dumps(profile, indent=2) + '\n')
    manifest = {'file_format_version': '1.2.1', 'layer': {
        'name': 'VK_LAYER_KHRONOS_profiles', 'type': 'GLOBAL', 'library_path': str(library),
        'api_version': '1.4.341', 'implementation_version': '1', 'description': 'Khronos Profiles',
        'enable_environment': {'ARTHROBOT_RTX_PROFILE': '1'},
        'disable_environment': {'ARTHROBOT_RTX_PROFILE_DISABLE': '1'}}}
    (manifest_dir / 'VkLayer_KHRONOS_profiles.json').write_text(json.dumps(manifest, indent=2) + '\n')
    os.environ.update({
        'ARTHROBOT_RTX_PROFILE': '1',
        'VK_KHRONOS_PROFILES_PROFILE_NAME': PROFILE_NAME,
        'VK_KHRONOS_PROFILES_PROFILE_DIRS': str(profile_dir),
        'VK_KHRONOS_PROFILES_SIMULATE_CAPABILITIES': 'SIMULATE_PROPERTIES_BIT',
        'VK_KHRONOS_PROFILES_DEBUG_REPORTS': 'DEBUG_REPORT_ERROR_BIT',
    })
    for key, path in (('VK_ADD_IMPLICIT_LAYER_PATH', manifest_dir), ('XDG_DATA_DIRS', cache / 'xdg')):
        prior = os.environ.get(key, '/usr/local/share:/usr/share' if key == 'XDG_DATA_DIRS' else '')
        os.environ[key] = str(path) + (':' + prior if prior else '')
    print('RTX_COMPAT: enabled the process-local Vulkan allocation cap', flush=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        sys.exit('usage: python -m arthrobot.sim.rtx_compat <command> [arguments...]')
    enable()
    os.execvp(sys.argv[1], sys.argv[1:])
