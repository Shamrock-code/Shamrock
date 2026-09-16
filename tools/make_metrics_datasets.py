# Build JS-friendly plot datasets from metrics-history aggregated JSON files.

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

BUILD_PROFILE_SUMMARY_RE = re.compile(
    r"Compilation \((\d+) times\):\s+"
    r"Parsing \(frontend\):\s+([0-9.]+) s\s+"
    r"Codegen & opts \(backend\):\s+([0-9.]+) s"
)


def load_snapshots(root):
    aggregated = Path(root) / "aggregated"
    if not aggregated.is_dir():
        raise FileNotFoundError(f"missing aggregated directory: {aggregated}")

    snapshots = []
    for path in sorted(aggregated.glob("*.json")):
        with path.open() as handle:
            snapshots.append(json.load(handle))
    snapshots.sort(key=lambda item: item.get("datetime", ""))
    return snapshots


def to_iso8601(datetime_str):
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%SZ").replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_doxygen_warnings(snapshots):
    data = []
    for snapshot in snapshots:
        data.append(
            {
                "datetime": to_iso8601(snapshot["datetime"]),
                "doxygen_warning_count": snapshot["metrics"]["doxygen_warn"][
                    "doxygen_warning_count"
                ],
            }
        )
    return data


def parse_build_profile_summary(text):
    match = BUILD_PROFILE_SUMMARY_RE.search(text)
    if match is None:
        raise ValueError("could not parse build profile summary")
    parsing_time_s = float(match.group(2))
    codegen_time_s = float(match.group(3))
    return {
        "translation_unit_count": int(match.group(1)),
        "parsing_time_s": parsing_time_s,
        "codegen_time_s": codegen_time_s,
        "total_time_s": parsing_time_s + codegen_time_s,
    }


def build_build_time_total(snapshots):
    data = []
    for snapshot in snapshots:
        profile = snapshot.get("metrics", {}).get("build_profile")
        if profile is None:
            continue
        parsed = parse_build_profile_summary(profile["data"])
        data.append({"datetime": to_iso8601(snapshot["datetime"]), **parsed})
    return data


COMPILE_MEMORY_TOP_N = 10
PARSE_TIME_TOP_N = 10
CODEGEN_TIME_TOP_N = 10
FUNCTION_SETS_TOP_N = 10
REPO_PATH_MARKER = "/Shamrock/Shamrock/"
CMAKE_OBJ_RE = re.compile(r"^(?:\./)?(?P<root>.+)/CMakeFiles/[^/]+\.dir/(?P<rel>.+)\.o$")
PARSE_FILE_LINE_RE = re.compile(r"^\s*([0-9.]+)\s+ms:\s+(\S+)\s*$", re.MULTILINE)
FUNCTION_SET_LINE_RE = re.compile(
    r"^\s*(\d+)\s+ms:\s+(.+)\s+\((\d+)\s+times,\s+avg\s+(\d+)\s+ms\)\s*$"
)
PARSE_FILES_MARKER = "**** Files that took longest to parse (compiler frontend):"
CODEGEN_FILES_MARKER = "**** Files that took longest to codegen (compiler backend):"
FUNCTION_SETS_MARKER = "**** Function sets that took longest to compile / optimize:"


def normalize_compile_path(path):
    path = (path or "").replace("\\", "/").strip()
    path = path.removeprefix("./")
    idx = path.rfind(REPO_PATH_MARKER)
    if idx != -1:
        path = path[idx + len(REPO_PATH_MARKER) :]
    return path


def top_compile_memory_files(usage, limit=COMPILE_MEMORY_TOP_N):
    by_path = {}
    for item in usage:
        path = normalize_compile_path(item.get("path"))
        rss = item.get("rss_mb")
        if not path or rss is None:
            continue
        rss = float(rss)
        prev = by_path.get(path)
        if prev is None or rss > prev:
            by_path[path] = rss
    ranked = sorted(by_path.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:limit]


def compile_memory_top10_layout():
    return {
        "title": {"text": "Top 10 compile peak RSS"},
        "margin": {"l": 72, "r": 24, "t": 56, "b": 180},
        "xaxis": {
            "title": {"text": "Date (UTC)"},
            "type": "date",
        },
        "yaxis": {
            "title": {"text": "Peak RSS (MB)"},
            "rangemode": "tozero",
        },
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 11},
        },
        "hovermode": "x unified",
    }


def build_compile_memory_top10(snapshots):
    # One trace per file that is in any snapshot's top 10. y is null when that
    # file is outside the top 10 of a given commit, so Plotly drops it there.
    dated_top = []
    for snapshot in snapshots:
        profile = snapshot.get("metrics", {}).get("build_profile") or {}
        usage = profile.get("compile_memory_usage")
        if not usage:
            continue
        dated_top.append((to_iso8601(snapshot["datetime"]), top_compile_memory_files(usage)))

    xs = [dt for dt, _ranking in dated_top]
    rss_by_file = {}
    for dt_idx, (_dt, ranking) in enumerate(dated_top):
        for path, rss in ranking:
            series = rss_by_file.setdefault(path, [None] * len(xs))
            series[dt_idx] = rss

    files = sorted(
        rss_by_file,
        key=lambda path: (-max(v for v in rss_by_file[path] if v is not None), path),
    )

    traces = []
    for path in files:
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": path,
                "x": xs,
                "y": rss_by_file[path],
                "connectgaps": False,
                "hovertemplate": (
                    "%{fullData.name}<br>%{x|%Y-%m-%d %H:%M UTC}<br>RSS: %{y:.1f} MB<extra></extra>"
                ),
            }
        )
    return traces


def normalize_compile_obj_path(path):
    path = (path or "").replace("\\", "/").strip()
    idx = path.rfind(REPO_PATH_MARKER)
    if idx != -1:
        path = path[idx + len(REPO_PATH_MARKER) :]
    path = path.removeprefix("./")
    match = CMAKE_OBJ_RE.match(path)
    if match:
        return f"{match.group('root')}/{match.group('rel')}"
    return path.removesuffix(".o")


def parse_longest_files(text, marker, limit):
    start = text.find(marker)
    if start == -1:
        return []
    rest = text[start + len(marker) :]
    end = rest.find("\n**** ")
    section = rest if end == -1 else rest[:end]
    by_path = {}
    for match in PARSE_FILE_LINE_RE.finditer(section):
        time_ms = float(match.group(1))
        path = normalize_compile_obj_path(match.group(2))
        if not path:
            continue
        prev = by_path.get(path)
        if prev is None or time_ms > prev:
            by_path[path] = time_ms
    ranked = sorted(by_path.items(), key=lambda item: (-item[1], item[0]))
    return [(path, time_ms / 1000.0) for path, time_ms in ranked[:limit]]


def parse_longest_parse_files(text, limit=PARSE_TIME_TOP_N):
    return parse_longest_files(text, PARSE_FILES_MARKER, limit)


def parse_longest_codegen_files(text, limit=CODEGEN_TIME_TOP_N):
    return parse_longest_files(text, CODEGEN_FILES_MARKER, limit)


def parse_time_top10_layout():
    return {
        "title": {"text": "Top 10 files that took longest to parse"},
        "margin": {"l": 72, "r": 24, "t": 56, "b": 180},
        "xaxis": {
            "title": {"text": "Date (UTC)"},
            "type": "date",
        },
        "yaxis": {
            "title": {"text": "Parse time (s)"},
            "rangemode": "tozero",
        },
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 11},
        },
        "hovermode": "x unified",
    }


def build_parse_time_top10(snapshots):
    # One trace per file that is in any snapshot's top 10. y is null when that
    # file is outside the top 10 of a given commit, so Plotly drops it there.
    dated_top = []
    for snapshot in snapshots:
        profile = snapshot.get("metrics", {}).get("build_profile") or {}
        report = profile.get("data")
        if not report:
            continue
        ranking = parse_longest_parse_files(report)
        if not ranking:
            continue
        dated_top.append((to_iso8601(snapshot["datetime"]), ranking))

    xs = [dt for dt, _ranking in dated_top]
    time_by_file = {}
    for dt_idx, (_dt, ranking) in enumerate(dated_top):
        for path, parse_s in ranking:
            series = time_by_file.setdefault(path, [None] * len(xs))
            series[dt_idx] = parse_s

    files = sorted(
        time_by_file,
        key=lambda path: (-max(v for v in time_by_file[path] if v is not None), path),
    )

    traces = []
    for path in files:
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": path,
                "x": xs,
                "y": time_by_file[path],
                "connectgaps": False,
                "hovertemplate": (
                    "%{fullData.name}<br>%{x|%Y-%m-%d %H:%M UTC}<br>"
                    "Parse: %{y:.1f} s<extra></extra>"
                ),
            }
        )
    return traces


def codegen_time_top10_layout():
    return {
        "title": {"text": "Top 10 codegen time"},
        "margin": {"l": 72, "r": 24, "t": 56, "b": 180},
        "xaxis": {
            "title": {"text": "Date (UTC)"},
            "type": "date",
        },
        "yaxis": {
            "title": {"text": "Codegen time (s)"},
            "rangemode": "tozero",
        },
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 11},
        },
        "hovermode": "x unified",
    }


def build_codegen_time_top10(snapshots):
    # One trace per file that is in any snapshot's top 10. y is null when that
    # file is outside the top 10 of a given commit, so Plotly drops it there.
    dated_top = []
    for snapshot in snapshots:
        profile = snapshot.get("metrics", {}).get("build_profile") or {}
        report = profile.get("data")
        if not report:
            continue
        ranking = parse_longest_codegen_files(report)
        if not ranking:
            continue
        dated_top.append((to_iso8601(snapshot["datetime"]), ranking))

    xs = [dt for dt, _ranking in dated_top]
    time_by_file = {}
    for dt_idx, (_dt, ranking) in enumerate(dated_top):
        for path, time_s in ranking:
            series = time_by_file.setdefault(path, [None] * len(xs))
            series[dt_idx] = time_s

    files = sorted(
        time_by_file,
        key=lambda path: (-max(v for v in time_by_file[path] if v is not None), path),
    )

    traces = []
    for path in files:
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": path,
                "x": xs,
                "y": time_by_file[path],
                "connectgaps": False,
                "hovertemplate": (
                    "%{fullData.name}<br>%{x|%Y-%m-%d %H:%M UTC}<br>"
                    "Codegen: %{y:.1f} s<extra></extra>"
                ),
            }
        )
    return traces


def parse_function_sets_section(text, limit=FUNCTION_SETS_TOP_N):
    start = text.find(FUNCTION_SETS_MARKER)
    if start == -1:
        return []
    rest = text[start + len(FUNCTION_SETS_MARKER) :]
    end = rest.find("\n**** ")
    section = rest if end == -1 else rest[:end]
    by_name = {}
    for line in section.splitlines():
        match = FUNCTION_SET_LINE_RE.match(line)
        if match is None:
            continue
        name = match.group(2).strip()
        time_ms = int(match.group(1))
        if not name:
            continue
        prev = by_name.get(name)
        if prev is None or time_ms > prev["time_ms"]:
            by_name[name] = {
                "name": name,
                "time_ms": time_ms,
                "count": int(match.group(3)),
                "avg_ms": int(match.group(4)),
            }
    ranked = sorted(by_name.values(), key=lambda item: (-item["time_ms"], item["name"]))
    return ranked[:limit]


def skip_decltype_prefix(name):
    if not name.startswith("decltype"):
        return name
    depth = 0
    for i, ch in enumerate(name):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return name[i + 1 :].lstrip()
    return name


def short_function_set_name(name, max_len=64):
    # Legend label: drop return type / decltype, keep the qualified-id.
    s = skip_decltype_prefix(name.strip()).removesuffix(" const")

    first_paren = s.find("(")
    prefix = s if first_paren == -1 else s[:first_paren]

    depth = 0
    last_space = None
    for i, ch in enumerate(prefix):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == " " and depth == 0:
            last_space = i
    core = prefix[last_space + 1 :] if last_space is not None else prefix
    core = core.strip() or name.strip()
    if len(core) > max_len:
        return core[: max_len - 3] + "..."
    return core


def hover_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def unique_display_names(full_names):
    shorts = [short_function_set_name(name) for name in full_names]
    counts = {}
    for short in shorts:
        counts[short] = counts.get(short, 0) + 1
    seen = {}
    result = []
    for short in shorts:
        if counts[short] == 1:
            result.append(short)
            continue
        seen[short] = seen.get(short, 0) + 1
        result.append(f"{short} #{seen[short]}")
    return result


def function_sets_top10_layout():
    return {
        "title": {"text": "Top 10 function sets (compile / optimize)"},
        "margin": {"l": 72, "r": 24, "t": 56, "b": 180},
        "xaxis": {
            "title": {"text": "Date (UTC)"},
            "type": "date",
        },
        "yaxis": {
            "title": {"text": "Compile / optimize time (s)"},
            "rangemode": "tozero",
        },
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.28,
            "x": 0,
            "xanchor": "left",
            "font": {"size": 11},
        },
        "hovermode": "x unified",
    }


def build_function_sets_top10(snapshots):
    # One trace per function set that is in any snapshot's top 10. y is null
    # when that set is outside the top 10 of a given commit, so Plotly drops
    # it there.
    dated_top = []
    for snapshot in snapshots:
        profile = snapshot.get("metrics", {}).get("build_profile") or {}
        report = profile.get("data")
        if not report:
            continue
        ranking = parse_function_sets_section(report)
        if not ranking:
            continue
        dated_top.append((to_iso8601(snapshot["datetime"]), ranking))

    xs = [dt for dt, _ranking in dated_top]
    series_by_name = {}
    for dt_idx, (_dt, ranking) in enumerate(dated_top):
        for item in ranking:
            series = series_by_name.setdefault(
                item["name"],
                {
                    "y": [None] * len(xs),
                    "count": [None] * len(xs),
                    "avg_ms": [None] * len(xs),
                },
            )
            series["y"][dt_idx] = item["time_ms"] / 1000.0
            series["count"][dt_idx] = item["count"]
            series["avg_ms"][dt_idx] = item["avg_ms"]

    names = sorted(
        series_by_name,
        key=lambda name: (-max(v for v in series_by_name[name]["y"] if v is not None), name),
    )
    display_names = unique_display_names(names)

    traces = []
    for name, display in zip(names, display_names):
        series = series_by_name[name]
        escaped = hover_escape(name)
        customdata = []
        for y_val, count, avg_ms in zip(series["y"], series["count"], series["avg_ms"]):
            if y_val is None:
                customdata.append([None, None, None])
            else:
                customdata.append([escaped, count, avg_ms])
        traces.append(
            {
                "type": "scatter",
                "mode": "lines+markers",
                "name": display,
                "x": xs,
                "y": series["y"],
                "customdata": customdata,
                "connectgaps": False,
                "hovertemplate": (
                    "%{customdata[0]}<br>%{x|%Y-%m-%d %H:%M UTC}<br>"
                    "time: %{y:.1f} s (%{customdata[1]} times, avg %{customdata[2]} ms)"
                    "<extra></extra>"
                ),
            }
        )
    return traces


# Exclusive loc partitions from tools/count_loc.py. Nested shammodels/* counts
# are subsets of "code" and must not be added into an extension total.
LOC_PARTITION_KINDS = ("code", "examples", "doc")
LOC_TOTAL_KEYS = ("totals", "total")


def loc_extension_total(counts):
    return sum(counts.get(kind, 0) for kind in LOC_PARTITION_KINDS)


def flatten_loc_totals(totals):
    flattened = {}
    for key, value in totals.items():
        flattened[key if key == "all" else f"all_{key}"] = value
    return flattened


def build_loc(snapshots):
    data = []
    for snapshot in snapshots:
        loc = snapshot.get("metrics", {}).get("loc")
        if loc is None:
            continue
        totals = next((loc[key] for key in LOC_TOTAL_KEYS if key in loc), None)
        if totals is None:
            continue
        row = {"datetime": to_iso8601(snapshot["datetime"]), **flatten_loc_totals(totals)}
        for key, counts in loc.items():
            if key in LOC_TOTAL_KEYS:
                continue
            row[key] = loc_extension_total(counts)
        data.append(row)
    return data


def write_dataset(output_dir, name, data, layout=None):
    path = output_dir / f"{name}.json"
    payload = {"name": name, "data": data}
    if layout is not None:
        payload["layout"] = layout
    path.write_text(json.dumps(payload, indent=3) + "\n")
    print(path)


def build_datasets(root, output_dir):
    snapshots = load_snapshots(root)
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    write_dataset(output_dir, "doxygen_warnings", build_doxygen_warnings(snapshots))
    write_dataset(output_dir, "build_time_total", build_build_time_total(snapshots))
    write_dataset(output_dir, "loc", build_loc(snapshots))
    write_dataset(
        output_dir,
        "compile_memory_top10",
        build_compile_memory_top10(snapshots),
        layout=compile_memory_top10_layout(),
    )
    write_dataset(
        output_dir,
        "parse_time_top10",
        build_parse_time_top10(snapshots),
        layout=parse_time_top10_layout(),
    )
    write_dataset(
        output_dir,
        "codegen_time_top10",
        build_codegen_time_top10(snapshots),
        layout=codegen_time_top10_layout(),
    )
    write_dataset(
        output_dir,
        "function_sets_top10",
        build_function_sets_top10(snapshots),
        layout=function_sets_top10_layout(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build JS-friendly plot datasets from metrics-history aggregated JSON."
    )
    parser.add_argument(
        "metrics_history_root",
        type=Path,
        help="Root of the metrics-history checkout (contains aggregated/)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write dataset JSON files into",
    )
    args = parser.parse_args()
    build_datasets(args.metrics_history_root, args.output_dir)


if __name__ == "__main__":
    main()
