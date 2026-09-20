#!/usr/bin/env bash
# Sync sciMAS files between the local checkout and a remote host.
#
# The remote is deliberately not baked into this file — point it at your own
# host through the environment:
#
#   export SCIMAS_REMOTE_HOST=user@your-server.example.com
#   export SCIMAS_REMOTE_REPO=/path/to/sciMAS     # default: ~/sciMAS
#
# The local side needs no configuration: it is derived from this script's
# location, so a clone works wherever it happens to live.
#
# Usage:
#   ./scripts/sync_dataset.sh                # default: full repo, push
#   ./scripts/sync_dataset.sh push [full|dataset]
#   ./scripts/sync_dataset.sh pull [full|dataset]
#   ./scripts/sync_dataset.sh verify
#
# Modes:
#   dataset  - only the test dataset/ subdir
#   full     - the whole sciMAS tree (excludes .git, __pycache__,
#              .DS_Store, *.pyc, runs/, tests/results/, *.log,
#              .gitignore.tmp)
#
# push = local → remote (default)
# pull = remote → local
# verify = compare md5 of every file on both sides without transferring

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_DATASET="$LOCAL_REPO/dataset"

REMOTE_HOST="${SCIMAS_REMOTE_HOST:-user@your-server.example.com}"
REMOTE_REPO="${SCIMAS_REMOTE_REPO:-~/sciMAS}"
REMOTE_DATASET="$REMOTE_REPO/dataset"

if [[ "$REMOTE_HOST" == *your-server.example.com* ]]; then
  echo "error: set SCIMAS_REMOTE_HOST (and SCIMAS_REMOTE_REPO) first:" >&2
  echo "  export SCIMAS_REMOTE_HOST=user@your-server.example.com" >&2
  echo "  export SCIMAS_REMOTE_REPO=/path/to/sciMAS" >&2
  exit 2
fi

DIRECTION="${1:-push}"
MODE="${2:-full}"

# ---- exclusions (used for both rsync and verify) ---------------------------
EXCLUDES=(
  --exclude='.DS_Store'
  --exclude='__pycache__'
  --exclude='*.pyc'
  --exclude='.git'
  --exclude='.gitignore.tmp'
  --exclude='runs'
  --exclude='*.log'
)

# ---- argument parsing ------------------------------------------------------
case "$DIRECTION" in
  push|--push) DIRECTION=push ;;
  pull|--pull) DIRECTION=pull ;;
  verify|--verify) DIRECTION=verify ;;
  *)
    echo "usage: $0 {push|pull|verify} [full|dataset]" >&2
    exit 2
    ;;
esac

case "$MODE" in
  full|dataset) ;;
  *) echo "unknown mode: $MODE (use full or dataset)" >&2; exit 2 ;;
esac

# ---- verify (no transfer) --------------------------------------------------
# Find predicate (portable across macOS BSD find and Linux GNU find).
FIND_EXCLUDES=(
  -not -path './.git*'
  -not -path '*/__pycache__*'
  -not -name '.DS_Store'
  -not -name '*.pyc'
  -not -name '.gitignore.tmp'
  -not -path './runs*'
  -not -name '*.log'
)

if [[ "$DIRECTION" == "verify" ]]; then
  echo "→ verifying $MODE parity (no transfer)"
  if [[ "$MODE" == "dataset" ]]; then
    ( cd "$LOCAL_DATASET" && find . "${FIND_EXCLUDES[@]}" -type f | sort ) > /tmp/_local.txt
    ssh "$REMOTE_HOST" "bash -s" <<EOF > /tmp/_remote.txt
cd "$REMOTE_DATASET"
find . \\
  -not -path './.git*' \\
  -not -path '*/__pycache__*' \\
  -not -name '.DS_Store' \\
  -not -name '*.pyc' \\
  -type f | sort
EOF
  else
    ( cd "$LOCAL_REPO" && find . "${FIND_EXCLUDES[@]}" -type f | LC_ALL=C sort ) > /tmp/_local.txt
    ssh "$REMOTE_HOST" "bash -s" <<EOF > /tmp/_remote.txt
cd "$REMOTE_REPO"
find . \\
  -not -path './.git*' \\
  -not -path '*/__pycache__*' \\
  -not -name '.DS_Store' \\
  -not -name '*.pyc' \\
  -not -name '.gitignore.tmp' \\
  -not -path './runs*' \\
  -not -name '*.log' \\
  -type f | LC_ALL=C sort
EOF
  fi
  if diff -q /tmp/_local.txt /tmp/_remote.txt >/dev/null; then
    echo "✅ file lists match (C-sort)"
  else
    echo "❌ file lists differ:"; diff /tmp/_local.txt /tmp/_remote.txt; exit 1
  fi
  exit 0
fi

# ---- rsync -----------------------------------------------------------------
if [[ "$DIRECTION" == "push" ]]; then
  SRC="$LOCAL_REPO"
  DEST="$REMOTE_HOST:$REMOTE_REPO"
  echo "→ pushing $MODE: $SRC → $DEST"
else
  SRC="$REMOTE_HOST:$REMOTE_REPO"
  DEST="$LOCAL_REPO"
  echo "← pulling $MODE: $SRC → $DEST"
fi

if [[ "$MODE" == "dataset" ]]; then
  rsync -av --delete --progress \
    "${EXCLUDES[@]}" \
    "$SRC/dataset/" "$DEST/dataset/"
else
  rsync -av --delete --progress \
    "${EXCLUDES[@]}" \
    "$SRC/" "$DEST/"
fi

echo
echo "✅ done — running verify"
"$0" verify "$MODE"
