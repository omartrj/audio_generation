#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
PLOT_SCRIPT="$SCRIPT_DIR/plot_trajectory.py"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

# Fall back to system python if venv is not present
[[ -f "$PYTHON" ]] || PYTHON="$(command -v python3)"

seqs=("$DATA_DIR"/seq*)
if [[ ${#seqs[@]} -eq 0 || ! -d "${seqs[0]}" ]]; then
    echo "No sequences found in $DATA_DIR"
    exit 1
fi

for seq_dir in "${seqs[@]}"; do
    gt="$seq_dir/gt.csv"
    if [[ ! -f "$gt" ]]; then
        echo "  SKIP  $(basename "$seq_dir")  (gt.csv not found)"
        continue
    fi
    echo "  PLOT  $(basename "$seq_dir")"
    "$PYTHON" "$PLOT_SCRIPT" --csv_path "$gt"
done

echo "Done."
