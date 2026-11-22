#!/bin/bash
set -euo pipefail
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0
cd "$CLAUDE_PROJECT_DIR"
grep -q '^-' <<< "$(git submodule status)" || exit 0
git submodule update --init --depth 1
