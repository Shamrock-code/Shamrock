#!/usr/bin/env bash
# Compiler launcher: run a command and record its peak RSS.
# Linux: GNU /usr/bin/time -f '%M' (kilobytes)
# macOS: BSD /usr/bin/time -l (bytes)
#
# When MEMLOG_DIR is set, append one {rss_mb, src, obj} record to
# MEMLOG_DIR/compile_memory.json.

# Pull the source (.cpp/.cc/.cxx/.c) and -o object path out of the compiler
# command line (CMake runs: memlog.sh <compiler> ... -c src.cpp -o src.cpp.o).
src=""
obj=""
prev=""
for a in "$@"; do
    if [ "$prev" = "-o" ]; then
        obj=$a
    fi
    case "$a" in
        -o) ;;
        -o*) obj=${a#-o} ;;
        *.cpp | *.cc | *.cxx | *.c) src=$a ;;
    esac
    prev=$a
done

if [ ! -x /usr/bin/time ]; then
    echo "memlog.sh: /usr/bin/time not found; running without RSS measurement" >&2
    exec "$@"
fi

# Run the compiler under /usr/bin/time and capture peak RSS in $tmp.
tmp=$(mktemp) || exit 1
trap 'rm -f "$tmp"' EXIT

if [ "$(uname -s)" = Darwin ]; then
    /usr/bin/time -l -o "$tmp" "$@"
else
    /usr/bin/time -f '%M' -o "$tmp" "$@"
fi
status=$?

# Parse time output and append {rss_mb, src, obj} to MEMLOG_DIR/compile_memory.json.
python3 - "$tmp" "$src" "$obj" "$(uname -s)" "${MEMLOG_DIR:-}" <<'PY'
import fcntl
import json
import os
import sys
import tempfile


def parse_output(time_file, src, obj, uname):
    text = open(time_file).read()
    rss_kb = 0.0
    if uname == "Darwin":
        for line in text.splitlines():
            if "maximum resident set size" in line:
                rss_kb = int(line.split()[0]) / 1024.0
                break
    else:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.isdigit():
                rss_kb = float(stripped)
    return src or None, obj or None, round(rss_kb / 1024.0, 1)


def write_output(path, src, obj, rss_mb):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record = {"rss_mb": rss_mb, "src": src, "obj": obj}
    lock_path = path + ".lock"
    with open(lock_path, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = []
        if os.path.isfile(path):
            with open(path) as handle:
                records = json.load(handle)
        records.append(record)
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".json")
        with os.fdopen(fd, "w") as handle:
            json.dump(records, handle, indent=4)
            handle.write("\n")
        os.replace(tmp_path, path)


time_file, src, obj, uname, memlog_dir = sys.argv[1:6]
src, obj, rss_mb = parse_output(time_file, src, obj, uname)
print(f"MEM_PEAK_MB={rss_mb:.1f} FILE={src or obj}", file=sys.stderr)
if memlog_dir:
    write_output(os.path.join(memlog_dir, "compile_memory.json"), src, obj, rss_mb)
PY

exit "$status"
