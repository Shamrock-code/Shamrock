# Release process

Shamrock is released on a 6-month basis. The release process goes as follows.

## Changing the version

We first change the version of the code :
```cmake
set(SHAMROCK_VERSION_MAJOR 2025)
set(SHAMROCK_VERSION_MINOR 10)
set(SHAMROCK_VERSION_PATCH 0)
```
Here the major version is the current year, and the minor version is the current month. Finally, the patch version indicates the minor release (bug fix or other).

Also one should change the version tag also in `dox.conf` as well as `env/helpers/_pysetup.py`.

## The release branch

First, if this is a true release (opposed to a patch), we branch from `main` and name it `release/<major>.<minor>.x`.

## Creating the release

We then draft a new release on GitHub. The tag must be set to `v<major>.<minor>.<patch>`, and we set the target branch to `release/<major>.<minor>.x`. Then set the release name to `Shamrock <major>.<minor>.<patch>`.

Then generate the release note from the last tag.

For the files, add the documentation, license, and coverage, named `Shamrock-<major>.<minor>.<patch>-<filename>`, except for the license file that will be named `LICENSE`.

Finally, write the text and ... done!

## Helper script

Once the version is bumped and the release branch pushed upstream, `tools/prepare_release.py` automates the rest up to the GitHub release itself (requires an authenticated `gh` CLI):
```bash
python3 tools/prepare_release.py <major>.<minor>.<patch>
```
It checks that `release/<major>.<minor>.x` exists on `Shamrock-code/Shamrock`, checks every version site on that branch (`CMakeLists.txt`, `dox.conf`, `_pysetup.py`), finds the CI run (`On Push` workflow, run on every push to `release/*`) on the head commit of that branch, waits for it if needed, requires it to be green and downloads the release artifacts it produced (code archive, documentation, coverage report) into `./shamrock-release-<version>/artifacts/`. It then assembles the release files in `./shamrock-release-<version>/assets/` and prints the draft release (tag, target, title, assets and generated notes) in the terminal without creating it. Use `--run-id <id>` to pick a specific CI run, and `--help` for the other options.
