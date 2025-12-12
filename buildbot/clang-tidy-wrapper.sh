#!/bin/bash
set -eo pipefail

# Run clang-tidy, forwarding all script arguments
"$CLANGTIDYBINARY" --quiet "$@" 2>&1 | \
    grep -vE '^[0-9]+ warnings? generated\.$'
