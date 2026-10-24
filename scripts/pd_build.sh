#!/usr/bin/env bash
# Build both engine images from one Strata checkout, so they come from the same commit (the session handoff needs it:
# docs/LESSONS.md#3).  Reads STRATA_SRC, IMG_PRE, IMG_DEC, CUDA_ARCHITECTURES and BUILD_JOBS from pd.env.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/pd.env"
DOCKER="$HERE/../docker"
commit=$(git -C "$STRATA_SRC" rev-parse HEAD)
git -C "$STRATA_SRC" diff --quiet HEAD -- src include cmake CMakeLists.txt \
  || echo "note: $STRATA_SRC has uncommitted engine changes; both images get them, COMMIT says $commit"
echo "building from $STRATA_SRC at $commit"

docker build -f "$DOCKER/Dockerfile.cuda" -t "$IMG_PRE" --build-arg BUILD_JOBS="${BUILD_JOBS:-0}" \
  --build-arg CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" --build-arg STRATA_COMMIT="$commit" "$STRATA_SRC"
docker build -f "$DOCKER/Dockerfile.gfx906" -t "$IMG_DEC" --build-arg BUILD_JOBS="${BUILD_JOBS:-0}" --build-arg STRATA_COMMIT="$commit" "$STRATA_SRC"
echo "built $IMG_PRE and $IMG_DEC from $commit"
