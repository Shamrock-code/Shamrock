#!/usr/bin/env bash
# Runs inside the container. Expects:
#   /repo  - the git repository (full history, read-only is fine)
#   /conf  - aliases.txt and avatars.txt
#   /out   - output directory
set -euo pipefail

: "${OUTPUT:=shamrock-history.mp4}"
: "${RESOLUTION:=1920x1080}"
: "${SECONDS_PER_DAY:=0.06}"
: "${TITLE:=Shamrock — github.com/Shamrock-code/Shamrock}"
: "${CRF:=21}"
: "${CAPTION_LINES:=12}"
: "${COUNT_ROWS:=15}"

export HOME=/tmp
git config --global --add safe.directory '*'

if [ "$(git -C /repo rev-parse --is-shallow-repository)" = "true" ]; then
    echo "error: /repo is a shallow clone, run 'git fetch --unshallow' first" >&2
    exit 1
fi

work=$(mktemp -d)
cd "$work"

echo "==> generating gource log"
gource --output-custom-log raw.log /repo

# Merge author aliases: field 2 of the custom log is the author name.
awk -F'|' -v OFS='|' '
    NR == FNR { if ($0 !~ /^#/ && NF >= 2) alias[$1] = $2; next }
    { if ($2 in alias) $2 = alias[$2]; print }
' /conf/aliases.txt raw.log > repo.log

echo "==> generating captions"
git -C /repo log --pretty=format:"%at|%s" --reverse --no-merges > captions.txt

echo "==> downloading GitHub avatars"
mkdir -p avatars
while IFS='|' read -r name _login id; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    if curl -sSfL "https://avatars.githubusercontent.com/u/${id}?v=4&s=256" -o avatar.tmp; then
        ffmpeg -nostdin -y -loglevel error -i avatar.tmp -vf scale=256:256 "avatars/${name}.png" \
            || echo "warning: could not convert avatar for ${name}" >&2
    else
        echo "warning: could not download avatar for ${name}" >&2
    fi
done < /conf/avatars.txt
rm -f avatar.tmp

w=${RESOLUTION%x*}
h=${RESOLUTION#*x}
# fonts are scaled by the patched gource, keep the caption margin proportional too
caption_offset=$(( -20 * h / 1080 ))

echo "==> rendering ${RESOLUTION} to /out/${OUTPUT}"
GOURCE_USER_COUNTS="$COUNT_ROWS" GOURCE_CAPTION_MAX_LINES="$CAPTION_LINES" \
xvfb-run -a -s "-screen 0 ${w}x${h}x24" \
    gource repo.log "-${RESOLUTION}" \
        --title "$TITLE" \
        --caption-file captions.txt --caption-size 18 --caption-colour DDDDDD \
        --caption-duration 3 --caption-offset "$caption_offset" \
        --user-image-dir avatars --user-scale 1.5 \
        --highlight-users --highlight-colour FFD54F \
        --seconds-per-day "$SECONDS_PER_DAY" --auto-skip-seconds 0.3 \
        --hide filenames,mouse,progress --key \
        --file-idle-time 0 --max-user-speed 500 --bloom-multiplier 0.6 \
        --dir-name-depth 3 --date-format "%Y-%m-%d" --font-size 20 \
        -r 30 -o - \
| ffmpeg -nostdin -y -loglevel warning -r 30 -f image2pipe -vcodec ppm -i - \
    -vcodec libx264 -preset medium -pix_fmt yuv420p -crf "$CRF" -movflags +faststart \
    "/out/${OUTPUT}"

rm -rf "$work"
echo "==> done: ${OUTPUT}"
