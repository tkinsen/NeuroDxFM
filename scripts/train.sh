#!/usr/bin/env bash
set -euo pipefail
torchrun --standalone --nproc_per_node=8 -m neurodxfm.cli train --config configs/main.yaml
