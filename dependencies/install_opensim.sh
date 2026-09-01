#!/bin/bash

# Exit when an error happens instead of continue.
set -e

: "${OPENSIM_CORE_SOURCE_DIR:?not set; run this script via install_opensim.py}"
: "${SIMBODY_SOURCE_DIR:?not set; run this script via install_opensim.py}"

# Default values for flags.
DEBUG_TYPE="Release"
NUM_JOBS=${OPENSIM_BUILD_JOBS:-24}
MOCO="off"
GENERATOR="Ninja"
PYTHON_ROOT_DIR=$1
WORKING_DIR="$(pwd)/opensim"
SIMBODY_INSTALL_DIR="$WORKING_DIR/simbody_install"
DEPENDENCIES_INSTALL_DIR="$WORKING_DIR/opensim_dependencies_install"
mkdir -p "$WORKING_DIR"

# Build and install Simbody from the submodule.
mkdir -p "$WORKING_DIR/simbody_build"
cd "$WORKING_DIR/simbody_build"
cmake "$SIMBODY_SOURCE_DIR" -G"$GENERATOR" -DCMAKE_BUILD_TYPE=$DEBUG_TYPE -DCMAKE_INSTALL_PREFIX="$SIMBODY_INSTALL_DIR" -DBUILD_EXAMPLES=off -DBUILD_TESTING=off
cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS
cmake --install .

# Build the remaining opensim-core dependencies.
mkdir -p "$WORKING_DIR/opensim_dependencies_build"
cd "$WORKING_DIR/opensim_dependencies_build"
cmake "$OPENSIM_CORE_SOURCE_DIR/dependencies" -G"$GENERATOR" -DCMAKE_BUILD_TYPE=$DEBUG_TYPE -DCMAKE_INSTALL_PREFIX="$DEPENDENCIES_INSTALL_DIR/" -DSUPERBUILD_ezc3d=off -DSUPERBUILD_simbody=off -DOPENSIM_WITH_CASADI=$MOCO -DBUILD_PYTHON_WRAPPING=on -DPython3_ROOT_DIR="$PYTHON_ROOT_DIR"
cmake . -LAH
cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS


# Build and install opensim-core.
mkdir -p "$WORKING_DIR/opensim_core_build"
cd "$WORKING_DIR/opensim_core_build"
cmake "$OPENSIM_CORE_SOURCE_DIR" -G"$GENERATOR" -DCMAKE_BUILD_TYPE=$DEBUG_TYPE -DOPENSIM_DEPENDENCIES_DIR="$DEPENDENCIES_INSTALL_DIR/" -DSIMBODY_HOME="$SIMBODY_INSTALL_DIR" -DOPENSIM_C3D_PARSER=None -DBUILD_TESTING=off -DCMAKE_INSTALL_PREFIX="$WORKING_DIR/opensim_core_install" -DOPENSIM_INSTALL_UNIX_FHS=off -DOPENSIM_WITH_CASADI=$MOCO -DBUILD_PYTHON_WRAPPING=on -DPython3_ROOT_DIR="$PYTHON_ROOT_DIR"
cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS
cmake --install .
