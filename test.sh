#!/bin/bash
set -e

OUTPUT_PATH=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}"

PYTHON="${PYTHON:-python3}"

case "$1" in
base)
  $PYTHON -m pytest mypy/test/testcheck.py \
    -k "not check-literal-cast-inference" \
    --deselect "mypy/test/testcheck.py::TypeCheckSuite::check-plugin-attrs.test::testAttrsGeneric" \
    --deselect "mypy/test/testcheck.py::TypeCheckSuite::check-dataclasses.test::testDataclassGenerics" \
    -v ${OUTPUT_PATH:+--junitxml="$OUTPUT_PATH"}
  ;;
new)
  $PYTHON -m pytest \
    "mypy/test/testcheck.py::TypeCheckSuite::check-plugin-attrs.test::testAttrsGeneric" \
    "mypy/test/testcheck.py::TypeCheckSuite::check-dataclasses.test::testDataclassGenerics" \
    "mypy/test/testcheck.py::TypeCheckSuite::check-literal-cast-inference.test" \
    -v ${OUTPUT_PATH:+--junitxml="$OUTPUT_PATH"}
    ;;
  *)
    echo "Usage: $0 [--output_path <path>] {base|new}"
    exit 1
    ;;
esac
