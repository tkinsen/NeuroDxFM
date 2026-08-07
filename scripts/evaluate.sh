#!/usr/bin/env bash
set -euo pipefail
python -m neurodxfm.cli evaluate --config configs/main.yaml --weights "$1" --output "$2"
