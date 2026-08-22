#!/usr/bin/env bash
# Configure, build, and test the C++ control-loop core.
#
# Run from anywhere:   ./cpp/build.sh
# Requires the project venv (cmake and pybind11 are installed into it).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/venv/bin/activate"
else
  echo "warning: ${REPO_ROOT}/venv not found; using whatever python3 is on PATH" >&2
fi

PYBIND11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"

echo "==> Configuring"
cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="${PYBIND11_DIR}"

echo "==> Building"
cmake --build "${SCRIPT_DIR}/build" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

echo "==> Running GoogleTest suite"
"${SCRIPT_DIR}/build/adas_core_tests"

echo "==> Running Python parity + scenario-equivalence tests"
cd "${REPO_ROOT}"
python3 -m pytest tests/test_cpp_python_parity.py tests/test_cpp_scenario_equivalence.py -q

echo
echo "Done. The extension module is at ${REPO_ROOT}/adas_core*.so"
