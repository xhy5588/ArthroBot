#!/usr/bin/env bash
# Build the Khronos Vulkan Profiles layer used by arthrobot.sim.rtx_compat.
# Only needed for rendering (GUI, cameras, videos) with Isaac Sim 5.1 on NVIDIA 595.x drivers.
set -euo pipefail
LAYER_DIR="${ARTHROBOT_VULKAN_PROFILES:-${HOME}/.cache/arthrobot-vulkan-profiles}"
if [[ ! -d "$LAYER_DIR/.git" ]]; then
    git clone --depth 1 --branch vulkan-sdk-1.4.341.0 \
        https://github.com/KhronosGroup/Vulkan-Profiles.git "$LAYER_DIR"
fi
cmake -S "$LAYER_DIR" -B "$LAYER_DIR/build" -D CMAKE_BUILD_TYPE=Release -D UPDATE_DEPS=ON -D BUILD_TESTS=OFF
cmake --build "$LAYER_DIR/build" --target ProfilesLayer --parallel 4
