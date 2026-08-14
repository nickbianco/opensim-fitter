# PowerShell port of install_opensim.sh for Windows. Mirrors the bash
# script step-for-step: clones opensim-core, builds its dependencies
# into opensim_dependencies_install/, then builds and installs
# opensim-core into opensim_core_install/.

$ErrorActionPreference = 'Stop'
# PowerShell 7.3+ propagates non-zero exit codes from native commands
# (git, cmake) through $ErrorActionPreference when this is enabled.
# Harmless no-op on Windows PowerShell 5.1.
$PSNativeCommandUseErrorActionPreference = $true

$DebugType = 'Release'
$NumJobs = if ($env:OPENSIM_BUILD_JOBS) { $env:OPENSIM_BUILD_JOBS } else { 24 }
$Moco = 'off'
$Org = 'nickbianco'
$Branch = '987a0cf44ee634ffeaa143ac0510446f71eb8c4d'
$Generator = 'Ninja'
$PythonRootDir = $args[0]
$WorkingDir = Join-Path $PWD 'opensim'

if (Test-Path $WorkingDir) {
    Remove-Item -Recurse -Force $WorkingDir
}
New-Item -ItemType Directory -Path $WorkingDir | Out-Null

# Get opensim-core.
git clone "https://github.com/$Org/opensim-core.git" `
    (Join-Path $WorkingDir 'opensim-core')
Set-Location (Join-Path $WorkingDir 'opensim-core')
git checkout $Branch

# Build opensim-core dependencies.
$DepsSrc = Join-Path $WorkingDir 'opensim-core/dependencies'
$DepsBuild = Join-Path $DepsSrc 'build'
$DepsInstall = Join-Path $WorkingDir 'opensim_dependencies_install/'
New-Item -ItemType Directory -Path $DepsBuild -Force | Out-Null
Set-Location $DepsBuild
cmake $DepsSrc `
    "-G$Generator" `
    "-DCMAKE_BUILD_TYPE=$DebugType" `
    "-DCMAKE_INSTALL_PREFIX=$DepsInstall" `
    '-DSUPERBUILD_ezc3d=off' `
    "-DOPENSIM_WITH_CASADI=$Moco" `
    '-DBUILD_PYTHON_WRAPPING=on' `
    "-DPython3_ROOT_DIR=$PythonRootDir"
cmake . -LAH
cmake --build . --config $DebugType -j $NumJobs

# Build and install opensim-core.
$CoreSrc = Join-Path $WorkingDir 'opensim-core'
$CoreBuild = Join-Path $CoreSrc 'build'
$CoreInstall = Join-Path $WorkingDir 'opensim_core_install'
New-Item -ItemType Directory -Path $CoreBuild -Force | Out-Null
Set-Location $CoreBuild
cmake $CoreSrc `
    "-G$Generator" `
    "-DCMAKE_BUILD_TYPE=$DebugType" `
    "-DOPENSIM_DEPENDENCIES_DIR=$DepsInstall" `
    '-DOPENSIM_C3D_PARSER=None' `
    '-DBUILD_TESTING=off' `
    "-DCMAKE_INSTALL_PREFIX=$CoreInstall" `
    '-DOPENSIM_INSTALL_UNIX_FHS=off' `
    "-DOPENSIM_WITH_CASADI=$Moco" `
    '-DBUILD_PYTHON_WRAPPING=on' `
    "-DPython3_ROOT_DIR=$PythonRootDir"
cmake --build . --config $DebugType -j $NumJobs
cmake --install .
