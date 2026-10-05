#!/usr/bin/env bash
# Print the progress of the running batches.
#   bash batch_status.sh
cd "$(dirname "$0")/.."
for f in logs/batch_status/*.txt; do [ -f "$f" ] && { echo "######## $(basename $f .txt)"; cat "$f"; echo; }; done
[ -f logs/batch_summary/loc_table.txt ] && { echo "######## measurement "; cat logs/batch_summary/loc_table.txt; echo; }
[ -f logs/batch_summary/acc_table.txt ] && { echo "######## accuracy "; cat logs/batch_summary/acc_table.txt; }
