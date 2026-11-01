#!/usr/bin/env bash
# Render a Gource movie of a git repository's history using Docker.
#
# Usage: ./make-movie.sh [-o output.mp4] <path/to/repo>
#
# Tunables (environment variables):
#   RESOLUTION=1920x1080  SECONDS_PER_DAY=0.06  CRF=21
#   CAPTION_LINES=12      COUNT_ROWS=15         TITLE="..."
#   DOCKER_BUILD_ARGS="..." / DOCKER_RUN_ARGS="..."  extra arguments for
#   `docker build` / `docker run` (e.g. "--network host" behind a local proxy)
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
image=shamrock-gource
output=shamrock-history.mp4

usage() {
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

while getopts "o:h" opt; do
    case $opt in
        o) output=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[ $# -eq 1 ] || usage

repo=$(cd "$1" && pwd)
out_dir=$(mkdir -p "$(dirname "$output")" && cd "$(dirname "$output")" && pwd)
out_name=$(basename "$output")

if [ ! -d "$repo/.git" ]; then
    echo "error: $repo is not a git checkout (worktrees are not supported)" >&2
    exit 1
fi

if [ "$(git -C "$repo" rev-parse --is-shallow-repository)" = "true" ]; then
    echo "error: $repo is a shallow clone, run 'git -C $repo fetch --unshallow' first" >&2
    exit 1
fi

echo "==> building docker image '$image'"
# shellcheck disable=SC2086
docker build ${DOCKER_BUILD_ARGS:-} \
    --build-arg http_proxy --build-arg https_proxy \
    --build-arg HTTP_PROXY --build-arg HTTPS_PROXY \
    -t "$image" "$script_dir"

echo "==> rendering $repo -> $out_dir/$out_name"
# shellcheck disable=SC2086
docker run --rm \
    --user "$(id -u):$(id -g)" \
    -v "$repo":/repo:ro \
    -v "$script_dir/config":/conf:ro \
    -v "$out_dir":/out \
    -e OUTPUT="$out_name" \
    -e RESOLUTION -e SECONDS_PER_DAY -e CRF -e CAPTION_LINES -e COUNT_ROWS -e TITLE \
    -e http_proxy -e https_proxy -e HTTP_PROXY -e HTTPS_PROXY \
    ${DOCKER_RUN_ARGS:-} \
    "$image"
