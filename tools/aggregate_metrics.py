# Merge metric__*.json files into a single JSON document with a UTC timestamp.

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def metric_key(path):
    name = path.name
    name = name.removeprefix("metric__")
    name = name.removesuffix(".json")
    return name


def find_metric_files(root):
    files = [path for path in Path(root).rglob("*.json") if path.is_file()]
    return sorted(files, key=lambda path: path.as_posix())


def env_metadata():
    meta = {}
    mapping = (
        ("sha", "GITHUB_SHA", str),
        ("ref", "GITHUB_REF", str),
        ("event_name", "GITHUB_EVENT_NAME", str),
        ("repository", "GITHUB_REPOSITORY", str),
        ("workflow", "GITHUB_WORKFLOW", str),
        ("run_id", "GITHUB_RUN_ID", int),
        ("run_attempt", "GITHUB_RUN_ATTEMPT", int),
    )
    for key, env_name, conv in mapping:
        value = os.environ.get(env_name)
        if value is not None:
            meta[key] = conv(value)

    return meta


def aggregate(root, now=None):
    if now is None:
        now = datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc).replace(microsecond=0)

    payload = {"datetime": now.strftime("%Y-%m-%d %H:%M:%SZ")}
    payload.update(env_metadata())

    metrics = {}
    for path in find_metric_files(root):
        key = metric_key(path)
        if key in metrics:
            raise ValueError(f"duplicate metric key {key!r} from {path}")
        with path.open() as handle:
            metrics[key] = json.load(handle)

    payload["metrics"] = metrics
    return payload


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: aggregate_metrics.py INPUT_DIR OUTPUT.json\n")
        sys.exit(2)

    payload = aggregate(sys.argv[1])
    text = json.dumps(payload, indent=3) + "\n"
    Path(sys.argv[2]).write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
