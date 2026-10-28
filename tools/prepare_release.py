#!/usr/bin/env python3
"""Prepare a Shamrock release, up to (but not including) the GitHub release itself.

Steps (see doc/sphinx/source/dev_doc/release_process.md):
  1. check that the release branch release/<major>.<minor>.x exists upstream
  2. check that every version site on that branch matches the requested version
  3. find the CI run ("On Push" workflow) on the head commit of that branch,
     wait for it if still running, and download the release artifacts it
     produced into <outdir>/artifacts/
  4. assemble the release assets into <outdir>/assets/ and print the draft
     GitHub release (tag, target, title, assets, notes) to the terminal

Nothing is created or triggered on GitHub: the release is only printed.

Requires the GitHub CLI (gh), authenticated with access to the repository.

Usage:
  python3 tools/prepare_release.py 2025.10.0
  python3 tools/prepare_release.py 2025.10.0 --run-id 123456789   # use this CI run
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile

try:
    import shamrock_tool_banner

    shamrock_tool_banner.print_tool_info("Release preparation tool")
except ImportError:
    print("--- Release preparation tool ---")

# CI run on every push to release/* (see .github/workflows/main_workflow.yml)
CI_WORKFLOW_FILE = "on_push_main.yml"

ARTIFACT_DOC = "shamrock_documentation"
ARTIFACT_COVERAGE = "code-coverage-report-acpp_omp_debug"


def artifact_code_archive(version):
    # name set in .github/workflows/make_code_archive.yml
    return f"shamrock-{version}.tar.gz"


class ReleaseError(Exception):
    pass


def step(msg):
    print(f"\n==> {msg}", flush=True)


def info(msg):
    print(f"    {msg}", flush=True)


def run(cmd, capture=True, check=True):
    """Run a command, return its stdout (str) when captured."""
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if check and res.returncode != 0:
        err = (res.stderr or "").strip() if capture else ""
        raise ReleaseError(f"command failed ({res.returncode}): {' '.join(cmd)}\n{err}")
    return res.stdout if capture else None


def gh_api_bytes(endpoint, accept=None):
    cmd = ["gh", "api", endpoint]
    if accept is not None:
        cmd += ["-H", f"Accept: {accept}"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise ReleaseError(
            f"gh api {endpoint} failed:\n{res.stderr.decode(errors='replace').strip()}"
        )
    return res.stdout


def parse_version(s):
    m = re.fullmatch(r"v?(\d{4})\.(\d{1,2})\.(\d+)", s.strip())
    if m is None:
        raise ReleaseError(
            f"invalid version '{s}', expected <year>.<month>.<patch>, e.g. 2025.10.0"
        )
    major, minor, patch = m.groups()
    return f"{major}.{minor}.{patch}", major, minor, patch


def version_key(version):
    return tuple(int(x) for x in version.split("."))


def ls_remote(repo, *refs):
    """Return {ref: sha} for the given refs (or all refs if none) on the remote."""
    out = run(["git", "ls-remote", f"https://github.com/{repo}.git", *refs])
    res = {}
    for line in out.splitlines():
        sha, ref = line.split("\t", 1)
        res[ref] = sha
    return res


############################################################
# 1. Release branch
############################################################


def check_branch(repo, branch):
    step(f"Checking that branch '{branch}' exists on {repo}")
    refs = ls_remote(repo, f"refs/heads/{branch}")
    sha = refs.get(f"refs/heads/{branch}")
    if sha is None:
        raise ReleaseError(
            f"branch '{branch}' not found on {repo}.\n"
            f"Create it from main first (after the version bump), e.g.:\n"
            f"  git push <upstream> <commit>:refs/heads/{branch}"
        )
    info(f"found {branch} @ {sha}")
    return sha


############################################################
# 2. Version sites
############################################################


def read_remote_file(repo, path, ref):
    return gh_api_bytes(
        f"repos/{repo}/contents/{path}?ref={ref}", accept="application/vnd.github.raw+json"
    )


def check_version_sites(read_file, version, major, minor, patch):
    """Return the list of mismatches; read_file(path) -> str."""
    errors = []

    def find(path, pattern):
        m = re.search(pattern, read_file(path), re.MULTILINE)
        if m is None:
            errors.append(f"{path}: could not find pattern {pattern!r}")
            return None
        return m.group(1)

    for name, expected in (("MAJOR", major), ("MINOR", minor), ("PATCH", patch)):
        val = find("CMakeLists.txt", rf"^\s*set\(SHAMROCK_VERSION_{name}\s+(\S+?)\s*\)")
        if val is not None:
            ok = val.isdigit() and int(val) == int(expected)
            info(f"[{'ok' if ok else 'MISMATCH'}] CMakeLists.txt SHAMROCK_VERSION_{name} = {val}")
            if not ok:
                errors.append(
                    f"CMakeLists.txt: SHAMROCK_VERSION_{name} is {val}, expected {expected}"
                )

    for path, pattern, label in (
        ("doc/doxygen/dox.conf", r"^PROJECT_NUMBER\s*=\s*(\S+)", "PROJECT_NUMBER"),
        ("env/helpers/_pysetup.py", r"^\s*version\s*=\s*\"([^\"]*)\"", "version"),
    ):
        val = find(path, pattern)
        if val is not None:
            ok = val == version
            info(f"[{'ok' if ok else 'MISMATCH'}] {path} {label} = {val}")
            if not ok:
                errors.append(f"{path}: {label} is {val}, expected {version}")

    return errors


def check_versions(repo, sha, version, major, minor, patch):
    step(f"Checking version sites on the release branch ({sha[:12]})")

    def read_file(path):
        return read_remote_file(repo, path, sha).decode()

    errors = check_version_sites(read_file, version, major, minor, patch)
    if errors:
        raise ReleaseError(
            "version sites are not up to date on the release branch:\n  "
            + "\n  ".join(errors)
            + "\nFix them on main (and the release branch) before releasing."
        )


############################################################
# 3. CI run artifacts
############################################################


def find_ci_run(repo, branch, sha):
    step(f"Looking for the '{CI_WORKFLOW_FILE}' run on {branch} @ {sha[:12]}")
    out = run(
        ["gh", "run", "list", "--repo", repo, "--workflow", CI_WORKFLOW_FILE,
         "--branch", branch, "--commit", sha, "--limit", "1",
         "--json", "databaseId,url,status,conclusion"]
    )  # fmt: skip
    runs = json.loads(out)
    if not runs:
        raise ReleaseError(
            f"no '{CI_WORKFLOW_FILE}' run found for {branch} @ {sha}.\n"
            f"CI runs on every push to release/*: wait for it to start, or run it manually with\n"
            f"  gh workflow run {CI_WORKFLOW_FILE} --repo {repo} --ref {branch}"
        )
    r = runs[0]
    info(f"run {r['databaseId']}: {r['url']}")
    return r["databaseId"]


def check_run(repo, run_id, sha):
    r = json.loads(
        run(["gh", "run", "view", str(run_id), "--repo", repo,
             "--json", "headSha,headBranch,url,workflowName"])
    )  # fmt: skip
    info(f"run {run_id} ({r['workflowName']} on {r['headBranch']}): {r['url']}")
    if r["headSha"] != sha:
        info(f"warning: run head {r['headSha']} != branch head {sha}")


def wait_run(repo, run_id):
    r = json.loads(
        run(["gh", "run", "view", str(run_id), "--repo", repo, "--json", "status,conclusion"])
    )
    if r["status"] != "completed":
        step(f"Waiting for run {run_id} to complete")
        subprocess.run(["gh", "run", "watch", str(run_id), "--repo", repo, "--interval", "60"])
        r = json.loads(
            run(["gh", "run", "view", str(run_id), "--repo", repo, "--json", "status,conclusion"])
        )
    if r["conclusion"] != "success":
        raise ReleaseError(
            f"run {run_id} ended with '{r['conclusion'] or r['status']}': the release must come "
            f"from a green CI.\nSee: gh run view {run_id} --repo {repo} --log-failed"
        )
    info(f"run {run_id} succeeded")


def download_artifacts(repo, run_id, artifacts_dir, names):
    step(f"Downloading artifacts of run {run_id} into {artifacts_dir}")
    out = run(
        ["gh", "api", "--paginate", f"repos/{repo}/actions/runs/{run_id}/artifacts",
         "--jq", ".artifacts[] | {id, name, expired}"]
    )  # fmt: skip
    artifacts = {}
    for line in out.splitlines():
        if line.strip():
            a = json.loads(line)
            artifacts[a["name"]] = a

    missing = [n for n in names if n not in artifacts]
    expired = [n for n in names if n in artifacts and artifacts[n]["expired"]]
    if missing or expired:
        raise ReleaseError(
            f"run {run_id} lacks required artifacts (missing: {missing or 'none'}, "
            f"expired: {expired or 'none'}).\nRe-run the CI on the release branch to regenerate them."
        )

    os.makedirs(artifacts_dir, exist_ok=True)
    for name in names:
        a = artifacts[name]
        dest = os.path.join(artifacts_dir, name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.makedirs(dest)

        tmp = dest + ".download"
        with open(tmp, "wb") as f:
            res = subprocess.run(
                ["gh", "api", f"repos/{repo}/actions/artifacts/{a['id']}/zip"],
                stdout=f,
                stderr=subprocess.PIPE,
            )
        if res.returncode != 0:
            os.remove(tmp)
            raise ReleaseError(
                f"failed to download artifact {name}:\n{res.stderr.decode().strip()}"
            )

        # Zipped artifacts are extracted; artifacts uploaded with `archive: false`
        # (e.g. the code archive) come back as the raw file
        if zipfile.is_zipfile(tmp):
            with zipfile.ZipFile(tmp) as z:
                z.extractall(dest)
            os.remove(tmp)
        else:
            os.replace(tmp, os.path.join(dest, name))
        info(f"downloaded {name}")


############################################################
# 4. Assets & draft
############################################################


def single_file(directory):
    files = [os.path.join(r, f) for r, _, fs in os.walk(directory) for f in fs]
    if len(files) != 1:
        raise ReleaseError(f"expected exactly one file in {directory}, found {len(files)}")
    return files[0]


def tar_ext(path):
    # name the archive after its content: older doc builds made a plain tar named .tar.gz
    with open(path, "rb") as f:
        return ".tar.gz" if f.read(2) == b"\x1f\x8b" else ".tar"


def assemble_assets(repo, sha, version, artifacts_dir, assets_dir, with_coverage):
    step(f"Assembling release assets into {assets_dir}")
    if os.path.exists(assets_dir):
        shutil.rmtree(assets_dir)
    os.makedirs(assets_dir)

    def require(name):
        d = os.path.join(artifacts_dir, name)
        if not os.path.isdir(d):
            raise ReleaseError(f"artifact '{name}' missing from {artifacts_dir}")
        return d

    src = single_file(require(artifact_code_archive(version)))
    shutil.copyfile(src, os.path.join(assets_dir, f"shamrock-{version}{tar_ext(src)}"))

    src = single_file(require(ARTIFACT_DOC))
    shutil.copyfile(
        src, os.path.join(assets_dir, f"shamrock-{version}-documentation{tar_ext(src)}")
    )

    if with_coverage:
        src = require(ARTIFACT_COVERAGE)
        with tarfile.open(
            os.path.join(assets_dir, f"shamrock-{version}-coverage.tar.gz"), "w:gz"
        ) as t:
            t.add(src, arcname=f"shamrock-{version}-coverage")

    for lic in ("LICENSE", "LICENSE.en"):
        with open(os.path.join(assets_dir, lic), "wb") as f:
            f.write(read_remote_file(repo, lic, sha))

    assets = sorted(os.path.join(assets_dir, f) for f in os.listdir(assets_dir))
    for a in assets:
        info(f"{os.path.basename(a)} ({os.path.getsize(a) / 1e6:.1f} MB)")
    return assets


def previous_tag(repo, version):
    tags = []
    for ref in ls_remote(repo, "refs/tags/v*"):
        m = re.fullmatch(r"refs/tags/(v\d+\.\d+\.\d+)", ref)
        if m is not None and version_key(m.group(1)[1:]) < version_key(version):
            tags.append(m.group(1))
    return max(tags, key=lambda t: version_key(t[1:]), default=None)


def generate_notes(repo, tag, branch, prev_tag):
    step(f"Generating release notes ({prev_tag or 'first release'}...{tag})")
    cmd = ["gh", "api", "-X", "POST", f"repos/{repo}/releases/generate-notes",
           "-f", f"tag_name={tag}", "-f", f"target_commitish={branch}"]  # fmt: skip
    if prev_tag is not None:
        cmd += ["-f", f"previous_tag_name={prev_tag}"]
    try:
        generated = json.loads(run(cmd))["body"]
    except ReleaseError as e:
        info(f"warning: could not generate the notes ({e}); use 'Generate release notes' on GitHub")
        generated = "TODO: generate the release notes from the last tag on GitHub."

    return (
        "## Outstanding changes\n\nTODO: highlights of this release.\n\n" + generated.strip() + "\n"
    )


def print_draft(repo, tag, branch, sha, title, assets, notes, notes_file, tag_exists):
    bar = "=" * 72
    print(f"\n{bar}\n DRAFT GITHUB RELEASE (not created)\n{bar}")
    print(f"Repository : {repo}")
    print(f"Tag        : {tag}{'  (WARNING: tag already exists)' if tag_exists else '  (new tag)'}")
    print(f"Target     : {branch} ({sha[:12]})")
    print(f"Title      : {title}")
    print("Assets     :")
    for a in assets:
        print(f"  - {a}")
    print(f"Notes      : {notes_file}")
    print(f"{'-' * 72}")
    print(notes)
    print(bar)
    print("To create it as a draft once reviewed:")
    print(
        f"  gh release create {tag} --repo {repo} --draft --target {branch} \\\n"
        f'    --title "{title}" --notes-file {notes_file} \\\n'
        f"    {' '.join(assets)}"
    )


############################################################
# main
############################################################


def main():
    parser = argparse.ArgumentParser(
        description="Check a Shamrock release branch, fetch its CI artifacts, then print the draft GitHub release."
    )
    parser.add_argument(
        "version", help="release version, e.g. 2025.10.0 (a leading 'v' is accepted)"
    )
    parser.add_argument("--repo", default="Shamrock-code/Shamrock", help="upstream repository")
    parser.add_argument("--outdir", help="output folder (default: ./shamrock-release-<version>)")
    parser.add_argument(
        "--run-id", type=int, help="use this CI run instead of the latest one on the branch head"
    )
    parser.add_argument(
        "--previous-tag", help="tag to generate the notes from (default: latest older tag)"
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="do not add the coverage report to the assets"
    )
    args = parser.parse_args()

    try:
        if shutil.which("gh") is None:
            raise ReleaseError("the GitHub CLI 'gh' is required (https://cli.github.com)")
        run(["gh", "auth", "status"])

        version, major, minor, patch = parse_version(args.version)
        tag = f"v{version}"
        branch = f"release/{major}.{minor}.x"
        title = f"Shamrock {version}"
        outdir = os.path.abspath(args.outdir or f"shamrock-release-{version}")
        artifacts_dir = os.path.join(outdir, "artifacts")
        assets_dir = os.path.join(outdir, "assets")

        sha = check_branch(args.repo, branch)
        check_versions(args.repo, sha, version, major, minor, patch)

        tag_exists = bool(ls_remote(args.repo, f"refs/tags/{tag}"))
        if tag_exists:
            info(f"warning: tag {tag} already exists on {args.repo}")

        if args.run_id is not None:
            step(f"Using run {args.run_id}")
            check_run(args.repo, args.run_id, sha)
            run_id = args.run_id
        else:
            run_id = find_ci_run(args.repo, branch, sha)
        wait_run(args.repo, run_id)

        with_coverage = not args.no_coverage
        needed = [artifact_code_archive(version), ARTIFACT_DOC]
        if with_coverage:
            needed.append(ARTIFACT_COVERAGE)
        download_artifacts(args.repo, run_id, artifacts_dir, needed)

        assets = assemble_assets(args.repo, sha, version, artifacts_dir, assets_dir, with_coverage)

        prev = args.previous_tag or previous_tag(args.repo, version)
        notes = generate_notes(args.repo, tag, branch, prev)
        notes_file = os.path.join(outdir, "release-notes.md")
        with open(notes_file, "w") as f:
            f.write(notes)

        print_draft(args.repo, tag, branch, sha, title, assets, notes, notes_file, tag_exists)
    except ReleaseError as e:
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
