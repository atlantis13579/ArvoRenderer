#!/bin/sh
set -eu
build_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cmake -S "$build_dir" -B "$build_dir/Xcode" -G Xcode
