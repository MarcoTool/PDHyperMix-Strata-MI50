#!/usr/bin/env bash
# Start the PD split: the prefill engine (it creates the shared expert arena), then the decode engine (it maps the
# same arena), then the router.  Refuses to start when RAM or /dev/shm cannot hold it (docs/LESSONS.md#1, #2), or when
# the two images come from different Strata commits (docs/LESSONS.md#3).  Build the images first: ./pd_build.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/pd.env"
mkdir -p "$WORK"; cp "$HERE/pd_router.py" "$WORK/"
python3 "$HERE/pd_cfg.py" "$HERE/pd.env" || exit 1

# ---- both images from one commit, or the decode engine refuses the prefill engine's sessions
c_pre=$(docker run --rm "$IMG_PRE" -c 'cat /opt/strata/engine/COMMIT' 2>/dev/null)
c_dec=$(docker run --rm "$IMG_DEC" -c 'cat /opt/strata/engine/COMMIT' 2>/dev/null)
if [ -z "$c_pre" ] || [ "$c_pre" != "$c_dec" ]; then
  echo "images built from different commits ($IMG_PRE: ${c_pre:-missing}, $IMG_DEC: ${c_dec:-missing}): run ./pd_build.sh"; exit 1
fi
# ---- tmpfs large enough for the arena + a snapshot + Strata's 4 GiB reserve
shm=$(df -BG --output=size /dev/shm | tail -1 | tr -dc 0-9)
want=$(echo "$SHM_SIZE" | tr -dc 0-9)
if [ "$shm" -lt "$want" ]; then
  echo "/dev/shm is ${shm}G, needs ${SHM_SIZE}: sudo mount -o remount,size=$SHM_SIZE /dev/shm"; exit 1
fi
# ---- RAM: the arena is new memory unless it already exists (a restart reuses it)
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
need=$ENGINE_GIB; [ -s "$ARENA" ] || need=$((ARENA_GIB + ENGINE_GIB))
if [ "$avail" -lt "$need" ]; then
  echo "MemAvailable ${avail} GiB, needs ${need} GiB: stop whatever else holds RAM (another Strata without --shared-expert-arena?)"; exit 1
fi
mkdir -p "$SLOTS"; chmod 777 "$SLOTS"
sudo -n sysctl -q -w kernel.numa_balancing=0 2>/dev/null || echo "note: could not set numa_balancing=0 (two NUMA nodes: decode is slower with it on)"

ready() {   # port container
  for i in $(seq 1 300); do
    curl -s -m 3 "http://127.0.0.1:$1/v1/models" 2>/dev/null | grep -q '"loaded"' && return 0
    docker ps -q --filter "name=^$2$" | grep -q . || return 1
    sleep 5
  done
  return 1
}
docker rm -f pd-pre pd-dec pd-router >/dev/null 2>&1

t0=$(date +%s)
docker run -d --name pd-pre $DOCKER_ARGS --gpus "device=$PRE_GPU" --network host --ipc host -v /dev/shm:/dev/shm \
  -e HOME=/tmp -v "$DATA":/data -v "$WORK":"$WORK" "$IMG_PRE" \
  -c "cd /opt/strata && exec .venv/bin/python serve/server.py --engine strata --config $WORK/pd-pre.json --host 127.0.0.1 --port $PORT_PRE --slot-save-path $SLOTS" >/dev/null
ready "$PORT_PRE" pd-pre || { echo "prefill engine failed"; tail -20 "$WORK/pd-pre.log"; exit 1; }
echo "prefill engine ready after $(( $(date +%s) - t0 )) s"

# the GPU device groups by the host's numbers: the image may not know the names (render is missing in the ROCm image)
gids=""; for g in video render; do id=$(getent group $g | cut -d: -f3); [ -n "$id" ] && gids="$gids --group-add $id"; done
t0=$(date +%s)
docker run -d --name pd-dec $DOCKER_ARGS --network host --ipc host -v /dev/shm:/dev/shm --device /dev/kfd --device /dev/dri \
  $gids -e HOME=/tmp/home -e HSA_OVERRIDE_GFX_VERSION=9.0.6 -e HIP_VISIBLE_DEVICES="$DEC_HIP_DEVICES" $DEC_ENV \
  -v "$DATA":/data -v "$WORK":"$WORK" "$IMG_DEC" \
  -lc "mkdir -p /tmp/home && cd /opt/strata && exec .venv/bin/python serve/server.py --engine strata --config $WORK/pd-dec.json --host 127.0.0.1 --port $PORT_DEC --slot-save-path $SLOTS" >/dev/null
ready "$PORT_DEC" pd-dec || { echo "decode engine failed"; tail -20 "$WORK/pd-dec.log"; exit 1; }
echo "decode engine ready after $(( $(date +%s) - t0 )) s"

docker run -d --name pd-router --network host -v "$WORK":"$WORK" -v /dev/shm:/dev/shm \
  -e PD_PORT="$PORT_ROUTER" -e PD_PRE_PORT="$PORT_PRE" -e PD_DEC_PORT="$PORT_DEC" -e PD_SLOTS="$SLOTS" -e PD_MIN_CHARS="$PD_MIN_CHARS" \
  -e PD_IMAGES="${PD_IMAGES:-decode}" \
  --entrypoint python3 "$IMG_ROUTER" "$WORK/pd_router.py" >/dev/null
sleep 2
curl -s -m 10 "http://127.0.0.1:$PORT_ROUTER/v1/models" | grep -q '"loaded"' && echo "router up on 127.0.0.1:$PORT_ROUTER" || echo "router not answering"
free -g | sed -n 2p
