# Gource movie of the repository history

Renders a [Gource](https://gource.io) video of the Shamrock git history in a
Docker container (no GPU or display needed), with:

- GitHub avatars as user icons and yellow user names,
- commit messages scrolling in order in the bottom right,
- a live per-user commit counter in the top right.

The last two features come from `gource-shamrock.patch`, applied on top of
Gource 0.54 when the image is built.

## Usage

```bash
# the clone must have the full history
git fetch --unshallow   # only if the clone is shallow

./tools/gource/make-movie.sh -o shamrock-history-1080p.mp4 .
```

The first run builds the `shamrock-gource` image (a few minutes), then the
render itself takes a few minutes on CPU.

Tunables are passed as environment variables, e.g.
`RESOLUTION=1280x720 SECONDS_PER_DAY=0.15 ./tools/gource/make-movie.sh .`:

| Variable          | Default       | Meaning                                    |
| ----------------- | ------------- | ------------------------------------------ |
| `RESOLUTION`      | `1920x1080`   | output resolution                          |
| `SECONDS_PER_DAY` | `0.06`        | animation speed (higher is slower/longer)  |
| `CRF`             | `21`          | x264 quality (lower is better/bigger)      |
| `CAPTION_LINES`   | `12`          | max commit messages shown at once          |
| `COUNT_ROWS`      | `15`          | max rows in the commit counter             |
| `TITLE`           | Shamrock link | title in the bottom left                   |

`DOCKER_BUILD_ARGS` / `DOCKER_RUN_ARGS` are appended to `docker build` /
`docker run` (e.g. `--network host` behind a proxy listening on localhost).

## Contributors

- `config/aliases.txt` merges the different spellings of an author name.
- `config/avatars.txt` maps each displayed name to a GitHub user id; the id of
  a login can be found at `https://api.github.com/users/<login>`. Authors not
  listed get the default Gource icon.
