#!/usr/bin/env bash
# Stop the PD split.  --free-arena also deletes the shared expert arena: it is a tmpfs file, so it keeps ~47 GiB of
# RAM after the engines stop (the next pd_up.sh reuses it and starts faster).  Free it before running anything else
# that needs the RAM.
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/pd.env"
docker stop -t 60 pd-router pd-dec pd-pre >/dev/null 2>&1
docker rm -f pd-router pd-dec pd-pre >/dev/null 2>&1
echo "engines stopped"
if [ "${1:-}" = "--free-arena" ]; then
  sudo -n rm -f "$ARENA" "$SLOTS"/*.bin 2>/dev/null && echo "arena removed"   # created by root inside the containers
fi
free -g | sed -n 2p
