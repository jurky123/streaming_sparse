#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
d=external_baselines/Sparse-VideoGen/svg/kernels/3rdparty/flashinfer/3rdparty/cutlass
rev=f115c3f85467d5d9619119d1dbeb9c03c3d73864
if [[ ! -e "$d" ]]; then
    git init -q "$d"
    git -C "$d" fetch --depth=1 https://github.com/NVIDIA/cutlass.git "$rev"
    git -C "$d" checkout --detach -q FETCH_HEAD
fi
test "$(git -C "$d" rev-parse HEAD)" = "$rev"
test -f "$d/include/cutlass/version.h"
