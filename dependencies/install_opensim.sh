#!/bin/bash

# Exit when an error happens instead of continue.
set -e

# Default values for flags.
DEBUG_TYPE="Release"
NUM_JOBS=${OPENSIM_BUILD_JOBS:-24}
MOCO="off"
ORG="nickbianco"
BRANCH="987a0cf44ee634ffeaa143ac0510446f71eb8c4d"
GENERATOR="Ninja"
PYTHON_ROOT_DIR=$1
WORKING_DIR="$(pwd)/opensim"
if [ -d "$WORKING_DIR" ]; then
    rm -rf "$WORKING_DIR"
fi
mkdir -p "$WORKING_DIR"

# Get opensim-core.
git clone https://github.com/$ORG/opensim-core.git "$WORKING_DIR/opensim-core"
cd "$WORKING_DIR/opensim-core"
git checkout $BRANCH

# Build opensim-core dependencies.
mkdir -p "$WORKING_DIR/opensim-core/dependencies/build"
cd "$WORKING_DIR/opensim-core/dependencies/build"
cmake "$WORKING_DIR/opensim-core/dependencies" -G"$GENERATOR" -DCMAKE_BUILD_TYPE=$DEBUG_TYPE -DCMAKE_INSTALL_PREFIX="$WORKING_DIR/opensim_dependencies_install/" -DSUPERBUILD_ezc3d=off -DOPENSIM_WITH_CASADI=$MOCO -DBUILD_PYTHON_WRAPPING=on -DPython3_ROOT_DIR="$PYTHON_ROOT_DIR"
cmake . -LAH
cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS


# Build and install opensim-core.
mkdir -p "$WORKING_DIR/opensim-core/build"
cd "$WORKING_DIR/opensim-core/build"
cmake "$WORKING_DIR/opensim-core" -G"$GENERATOR" -DCMAKE_BUILD_TYPE=$DEBUG_TYPE -DOPENSIM_DEPENDENCIES_DIR="$WORKING_DIR/opensim_dependencies_install/" -DOPENSIM_C3D_PARSER=None -DBUILD_TESTING=off -DCMAKE_INSTALL_PREFIX="$WORKING_DIR/opensim_core_install" -DOPENSIM_INSTALL_UNIX_FHS=off -DOPENSIM_WITH_CASADI=$MOCO -DBUILD_PYTHON_WRAPPING=on -DPython3_ROOT_DIR="$PYTHON_ROOT_DIR"
cmake --build . --config $DEBUG_TYPE -j$NUM_JOBS
cmake --install .
