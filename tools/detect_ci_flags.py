"""Detect CI flags from a unified diff against the PR base."""

import argparse
import re
import sys
from pathlib import Path

MODELS = ("sph", "ramses", "zeus", "gsph")

DIFF_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:a/|b/)?(.+?)\s*$")


def parse_changed_paths(diff_text: str) -> set[str]:
    paths: set[str] = set()
    for line in diff_text.splitlines():
        match = DIFF_PATH_RE.match(line)
        if not match:
            continue
        path = match.group(1)
        if path == "/dev/null":
            continue
        paths.add(path)
    return paths


def is_cpp_path(path: str) -> bool:
    return path.endswith(".cpp") or path.endswith(".hpp")


def is_under_prefix(path: str, prefix: str) -> bool:
    normalized = path.replace("\\", "/")
    return normalized == prefix or normalized.startswith(prefix + "/")


def is_model_specific_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    for model in MODELS:
        if is_under_prefix(normalized, f"src/shammodels/{model}"):
            return True
        if is_under_prefix(normalized, f"src/tests/shammodels/{model}"):
            return True
    return False


def is_core_component_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if not is_under_prefix(normalized, "src"):
        return False
    return not is_model_specific_path(normalized)


def is_model_path(path: str, model: str) -> bool:
    normalized = path.replace("\\", "/")
    return is_under_prefix(normalized, f"src/shammodels/{model}") or is_under_prefix(
        normalized, f"src/tests/shammodels/{model}"
    )


def evaluate_flags(paths: set[str]) -> dict[str, bool]:
    check_cpp = any(is_cpp_path(p) for p in paths)

    flags = {
        "check_cpp": check_cpp,
        "run_sph_tests": any(is_core_component_path(p) or is_model_path(p, "sph") for p in paths),
        "run_ramses_tests": any(
            is_core_component_path(p) or is_model_path(p, "ramses") for p in paths
        ),
        "run_zeus_tests": any(is_core_component_path(p) or is_model_path(p, "zeus") for p in paths),
        "run_gsph_tests": any(is_core_component_path(p) or is_model_path(p, "gsph") for p in paths),
    }
    return flags


def write_github_output(flags: dict[str, bool], output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as out:
        out.writelines(f"{name}={str(value).lower()}\n" for name, value in flags.items())


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect CI flags from a PR diff.")
    parser.add_argument("--diff", required=True, help="Path to unified diff file.")
    parser.add_argument(
        "--github-output",
        help="Path to GITHUB_OUTPUT file for workflow job outputs.",
    )
    args = parser.parse_args()

    diff_path = Path(args.diff)
    if not diff_path.exists():
        print(f"error: diff file not found: {args.diff}", file=sys.stderr)
        return 1

    diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
    paths = parse_changed_paths(diff_text)
    flags = evaluate_flags(paths)

    print(f"Changed paths ({len(paths)}):")
    for path in sorted(paths):
        print(f"  {path}")

    print("CI flags:")
    for name, value in flags.items():
        print(f"  {name}={value}")

    if args.github_output:
        write_github_output(flags, args.github_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
