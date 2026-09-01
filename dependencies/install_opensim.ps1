$ErrorActionPreference = 'Stop'
# PowerShell 7.3+ propagates non-zero exit codes from native commands
# (git, cmake) through $ErrorActionPreference when this is enabled.
# Harmless no-op on Windows PowerShell 5.1.
$PSNativeCommandUseErrorActionPreference = $true

foreach ($Name in 'OPENSIM_CORE_SOURCE_DIR', 'SIMBODY_SOURCE_DIR') {
    if (-not (Test-Path "env:$Name")) {
        throw "$Name not set; run this script via install_opensim.py"
    }
}

$DebugType = 'Release'
$NumJobs = if ($env:OPENSIM_BUILD_JOBS) { $env:OPENSIM_BUILD_JOBS } else { 24 }
$Moco = 'off'
$Generator = 'Ninja'
$PythonRootDir = $args[0]
$WorkingDir = Join-Path $PWD 'opensim'
$SimbodyInstall = Join-Path $WorkingDir 'simbody_install'
$DepsInstall = Join-Path $WorkingDir 'opensim_dependencies_install/'

New-Item -ItemType Directory -Path $WorkingDir -Force | Out-Null

# Build and install Simbody from the submodule.
$SimbodyBuild = Join-Path $WorkingDir 'simbody_build'
New-Item -ItemType Directory -Path $SimbodyBuild -Force | Out-Null
Set-Location $SimbodyBuild
cmake $env:SIMBODY_SOURCE_DIR `
    "-G$Generator" `
    "-DCMAKE_BUILD_TYPE=$DebugType" `
    "-DCMAKE_INSTALL_PREFIX=$SimbodyInstall" `
    '-DBUILD_EXAMPLES=off' `
    '-DBUILD_TESTING=off'
cmake --build . --config $DebugType -j $NumJobs
cmake --install .

# Build the remaining opensim-core dependencies.
$DepsSrc = "$env:OPENSIM_CORE_SOURCE_DIR/dependencies"
$DepsBuild = Join-Path $WorkingDir 'opensim_dependencies_build'
New-Item -ItemType Directory -Path $DepsBuild -Force | Out-Null
Set-Location $DepsBuild
cmake $DepsSrc `
    "-G$Generator" `
    "-DCMAKE_BUILD_TYPE=$DebugType" `
    "-DCMAKE_INSTALL_PREFIX=$DepsInstall" `
    '-DSUPERBUILD_ezc3d=off' `
    '-DSUPERBUILD_simbody=off' `
    "-DOPENSIM_WITH_CASADI=$Moco" `
    '-DBUILD_PYTHON_WRAPPING=on' `
    "-DPython3_ROOT_DIR=$PythonRootDir"
cmake . -LAH
cmake --build . --config $DebugType -j $NumJobs

# Build and install opensim-core.
$CoreBuild = Join-Path $WorkingDir 'opensim_core_build'
$CoreInstall = Join-Path $WorkingDir 'opensim_core_install'
New-Item -ItemType Directory -Path $CoreBuild -Force | Out-Null
Set-Location $CoreBuild
cmake $env:OPENSIM_CORE_SOURCE_DIR `
    "-G$Generator" `
    "-DCMAKE_BUILD_TYPE=$DebugType" `
    "-DOPENSIM_DEPENDENCIES_DIR=$DepsInstall" `
    "-DSIMBODY_HOME=$SimbodyInstall" `
    '-DOPENSIM_C3D_PARSER=None' `
    '-DBUILD_TESTING=off' `
    "-DCMAKE_INSTALL_PREFIX=$CoreInstall" `
    '-DOPENSIM_INSTALL_UNIX_FHS=off' `
    "-DOPENSIM_WITH_CASADI=$Moco" `
    '-DBUILD_PYTHON_WRAPPING=on' `
    "-DPython3_ROOT_DIR=$PythonRootDir"
cmake --build . --config $DebugType -j $NumJobs
cmake --install .
