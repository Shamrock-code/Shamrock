#!/usr/bin/env python3
"""Run clang-tidy on a single file using the repo's own clang-tidy database
generator (buildbot/make_clang_tidy_db.py).

clang-tidy can't invoke the AdaptiveCpp `acpp` compiler wrapper directly, so
that script expands each compile command via `acpp --acpp-dryrun` into the
plain-clang invocation acpp actually runs, and writes the result to
build/clang-tidy.mod/compile_commands.json. This regenerates that database
when missing or stale, then runs the newest clang-tidy found on PATH
against it (not hardcoded to one version, so this also works on a host
with a different LLVM install than this container's).

Usage: .claude/tools/clang-tidy-check.py <path/to/file.cpp>
"""

import os
import re
import shutil
import subprocess
import sys


def pick_clang_tidy():
    """Newest clang-tidy-N found on PATH, falling back to unversioned clang-tidy."""
    best = None
    pattern = re.compile(r"^clang-tidy-(\d+)$")
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        try:
            entries = os.listdir(directory or ".")
        except OSError:
            continue
        for entry in entries:
            m = pattern.match(entry)
            if m and os.access(os.path.join(directory, entry), os.X_OK):
                best = max(best or 0, int(m.group(1)))
    if best is not None:
        return f"clang-tidy-{best}"
    if shutil.which("clang-tidy"):
        return "clang-tidy"
    print("error: no clang-tidy found on PATH", file=sys.stderr)
    sys.exit(1)


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path/to/file.cpp>", file=sys.stderr)
        return 1

    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True
    ).stdout.strip()
    build_dir = os.path.join(repo_root, "build")
    cdb_path = os.path.join(build_dir, "compile_commands.json")
    if not os.path.exists(cdb_path):
        print(f"error: {cdb_path} not found - run shamconfigure first", file=sys.stderr)
        return 1

    mod_dir = os.path.join(build_dir, "clang-tidy.mod")
    mod_db = os.path.join(mod_dir, "compile_commands.json")
    if not os.path.exists(mod_db) or os.path.getmtime(cdb_path) > os.path.getmtime(mod_db):
        print("Regenerating build/clang-tidy.mod (~30s) ...", file=sys.stderr)
        subprocess.run(
            [sys.executable, os.path.join(repo_root, "buildbot", "make_clang_tidy_db.py")],
            cwd=build_dir,
            check=True,
        )

    target = os.path.realpath(sys.argv[1])
    return subprocess.run([pick_clang_tidy(), target, "-p", mod_dir]).returncode


if __name__ == "__main__":
    sys.exit(main())
