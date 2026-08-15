# Count lines of tracked source files, excluding git submodules.

import json
import os
import subprocess
import sys
from pathlib import Path

# Kind -> repository path prefixes it covers (relative to the repo root).
# First matching partition kind wins, so more specific prefixes must come first.
# "code" has no prefixes: it is the fallback for unmatched files
# (src/, cmake/, tools/, env/, buildbot/, ...).
KINDS = {
    "code": [],
    "shammodels/sph": [
        "src/shammodels/sph",
    ],
    "shammodels/ramses": [
        "src/shammodels/ramses",
    ],
    "shammodels/zeus": [
        "src/shammodels/zeus",
    ],
    "shammodels/gsph": [
        "src/shammodels/gsph",
    ],
    "examples": [
        "examples",
        "doc/sphinx/examples",
    ],
    "doc": [
        "doc",
    ],
}

# Detail kinds are also counted in their parent. They are extra breakdowns,
# not exclusive buckets. total.all sums partition kinds only.
KIND_PARENT = {
    "shammodels/sph": "code",
    "shammodels/ramses": "code",
    "shammodels/zeus": "code",
    "shammodels/gsph": "code",
}

CATEGORIES = {
    "*.cpp": ["*.cpp"],
    "*.hpp": ["*.hpp"],
    "*.py": ["*.py"],
    "*.md": ["*.md"],
    "*.rst": ["*.rst"],
    "CMakeLists.txt + *.cmake": [
        "CMakeLists.txt",
        "**/CMakeLists.txt",
        "*.cmake",
    ],
}


def submodule_prefixes():
    try:
        out = subprocess.check_output(
            [
                "git",
                "config",
                "--file",
                ".gitmodules",
                "--get-regexp",
                r"^submodule\..*\.path$",
            ],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [line.split(maxsplit=1)[1].strip() for line in out.splitlines() if line.strip()]


def in_submodule(path, prefixes):
    s = str(path)
    return any(s == prefix or s.startswith(prefix + "/") for prefix in prefixes)


def matches_prefix(path_str, prefix):
    return path_str == prefix or path_str.startswith(prefix + "/")


def classify(path):
    s = path.as_posix()
    primary = None
    details = []
    for kind, prefixes in KINDS.items():
        if not any(matches_prefix(s, prefix) for prefix in prefixes):
            continue
        if kind in KIND_PARENT:
            details.append(kind)
        elif primary is None:
            primary = kind
    return primary or "code", details


def list_files(patterns, prefixes):
    out = subprocess.check_output(["git", "ls-files", "-z", "--"] + patterns)
    files = []
    seen = set()
    for raw in out.split(b"\0"):
        if not raw:
            continue
        path = Path(raw.decode())
        if path in seen or in_submodule(path, prefixes) or not path.is_file():
            continue
        seen.add(path)
        files.append(path)
    return files


def empty_kind_counts():
    return {kind: 0 for kind in KINDS}


def count_loc():
    prefixes = submodule_prefixes()
    result = {}
    grand = empty_kind_counts()

    for name, patterns in CATEGORIES.items():
        counts = empty_kind_counts()
        for path in list_files(patterns, prefixes):
            kind, details = classify(path)
            with path.open("rb") as handle:
                n = sum(1 for _ in handle)
            counts[kind] += n
            grand[kind] += n
            for detail in details:
                counts[detail] += n
                grand[detail] += n
        result[name] = counts

    partition_total = sum(grand[kind] for kind in KINDS if kind not in KIND_PARENT)
    result["total"] = {**grand, "all": partition_total}
    return result


def main():
    os.chdir(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())

    result = count_loc()
    text = json.dumps(result, indent=3) + "\n"

    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
