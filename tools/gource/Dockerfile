# Gource 0.54 patched with (the patch is applied with -l since the repo's
# pre-commit hooks normalise whitespace in its context lines):
#   - ordered, scrolling commit captions (GOURCE_CAPTION_MAX_LINES)
#   - live per-user commit counter in the top right (GOURCE_USER_COUNTS)
# plus Xvfb + ffmpeg to render headless to an mp4.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

COPY gource-shamrock.patch /tmp/gource-shamrock.patch

RUN sed 's/^Types: deb$/Types: deb-src/' /etc/apt/sources.list.d/ubuntu.sources \
        > /etc/apt/sources.list.d/ubuntu-src.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential dpkg-dev ca-certificates curl git ffmpeg xvfb xauth \
 && apt-get build-dep -y gource \
 && mkdir -p /tmp/gsrc && cd /tmp/gsrc \
 && apt-get source gource \
 && cd gource-0.54 \
 && patch -p1 -l < /tmp/gource-shamrock.patch \
 && ./configure --prefix=/usr/local \
 && make -j"$(nproc)" \
 && make install \
 && cd / && rm -rf /tmp/gsrc /tmp/gource-shamrock.patch \
 && rm -rf /var/lib/apt/lists/*

COPY render.sh /usr/local/bin/render.sh
RUN chmod +x /usr/local/bin/render.sh

ENTRYPOINT ["/usr/local/bin/render.sh"]
