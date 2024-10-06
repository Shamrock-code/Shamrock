# Release process

Shamrock is released on a 6 months basis. The release process goes as follows.

## Changing the version

We first change the version of the code :
```cmake
set(SHAMROCK_VERSION_MAJOR 2024)
set(SHAMROCK_VERSION_MINOR 10)
set(SHAMROCK_VERSION_PATCH 0)
```
Here the major version is the current year, and the minor version is the current month. Finally, the patch version indicate the minor release (bug fix or other).

## The release branch

First, if this is a true release (opposed to a patch), we branch from `main` and name it `release/<major>.<minor>.x`.

## The release workflow

We launch the `Prepare release` workflow on the release branch. This will generate the doc, the coverage info and the source code archive.

## Creating the release

We then draft a new release on GitHub. The tag must be set to `v<major>.<minor>.<patch>`, and we set the target branch to `release/<major>.<minor>.x`. Then set the release name to `Shamrock <major>.<minor>.<patch>`.

Then generate the release note from the last tag.

For the files, add the documentation, license, and coverage, named `Shamrock-<major>.<minor>.<patch>-<filename>`, except for the license file that will be named `LICENSE`.

Finally, write the text and ... done !
