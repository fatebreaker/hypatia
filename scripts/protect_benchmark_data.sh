#!/bin/bash
# Make benchmark data files read-only.
#
# Sandbox.link_data hard-links instead of copying, which is what makes BixBench
# affordable (1.1 GB per instance, copied per rollout, is terabytes over a
# training run and blocks the event loop for minutes). Hard links share the
# inode, so a cell that writes to data_files/ would corrupt the benchmark for
# every future rollout. Read-only source turns that silent corruption into a
# loud PermissionError.
set -u
# Derived from this script's location, like env.sh and paths.py -- a clone
# anywhere works, and benchmarks/*/problems may be a symlink onto scratch.
P="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Resolved, because benchmarks/*/problems is often a symlink onto scratch and
# `find` does not follow one given as its starting point -- it walked the link
# itself and chmod-ed nothing while still exiting 0.
D="$(readlink -f "${1:-$P/benchmarks/bixbench/problems}")"
find "$D" -type f -exec chmod a-w {} +
n=$(find "$D" -type f | wc -l)
echo "made read-only: $n files under $D"
# Rule 1 of this repo: assert non-zero coverage rather than succeed silently.
[ "$n" -gt 0 ] || { echo "FATAL: no files found under $D" >&2; exit 1; }
