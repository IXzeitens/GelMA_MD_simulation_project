#!/usr/bin/env bash
# Source-build LAMMPS with Kokkos+CUDA for RTX 50xx (Blackwell sm_120).
# Run from WSL after `conda activate lammps_gpu` (provides nvcc + CUDA libs).
# Expects clone at ~/lammps-src (shallow OK).
set -eo pipefail
# Conda activate scripts use unset env vars; avoid `set -u`.
export NVCC_PREPEND_FLAGS=""

SRC=$HOME/lammps-src
BUILD=$SRC/build
INSTALL=$SRC/install

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lammps_gpu

rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"

# Use the conda env's nvcc and toolkit
export CUDACXX=$CONDA_PREFIX/bin/nvcc
export CUDA_HOME=$CONDA_PREFIX

cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX="$INSTALL" \
  -D BUILD_MPI=ON \
  -D BUILD_OMP=ON \
  -D PKG_KOKKOS=ON \
  -D Kokkos_ENABLE_CUDA=ON \
  -D Kokkos_ARCH_BLACKWELL120=ON \
  -D Kokkos_ENABLE_OPENMP=ON \
  -D PKG_KSPACE=ON \
  -D PKG_MOLECULE=ON \
  -D PKG_RIGID=ON \
  -D PKG_EXTRA-DUMP=ON \
  -D PKG_EXTRA-FIX=ON \
  -D PKG_EXTRA-COMPUTE=ON \
  -D CMAKE_CUDA_COMPILER="$CUDACXX" \
  ../cmake

echo "=== cmake config done; starting build ==="
make -j 6
make install
echo "=== install done -> $INSTALL/bin/lmp ==="
ls -la "$INSTALL/bin/lmp"
