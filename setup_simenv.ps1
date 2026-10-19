param(
    [string]$CondaExe = "$env:USERPROFILE\miniconda3\Scripts\conda.exe"
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Executable $Arguments" }
}

$condaEnvs = (& $CondaExe env list --json | ConvertFrom-Json).envs
$simPrefix = $condaEnvs | Where-Object { (Split-Path -Leaf $_) -eq 'simenv' } | Select-Object -First 1
if (-not $simPrefix) {
    Invoke-Checked $CondaExe @('create','-n','simenv','python=3.11','pip','-y')
    $condaEnvs = (& $CondaExe env list --json | ConvertFrom-Json).envs
    $simPrefix = $condaEnvs | Where-Object { (Split-Path -Leaf $_) -eq 'simenv' } | Select-Object -First 1
}
if (-not (Test-Path -LiteralPath (Join-Path $simPrefix 'python.exe'))) {
    Invoke-Checked $CondaExe @('install','-n','simenv','python=3.11','pip','-y')
}
$simPython = Join-Path $simPrefix 'python.exe'
$env:GIT_LFS_SKIP_SMUDGE = '1'
if (-not (Test-Path 'vendor/openpi/.git')) {
    Invoke-Checked 'git' @('clone','--depth','1','https://github.com/Physical-Intelligence/openpi.git','vendor/openpi')
    Invoke-Checked 'git' @('-C','vendor/openpi','fetch','--depth','1','origin','215abfb217dbac7d5f1273282331b9b1866c0479')
    Invoke-Checked 'git' @('-C','vendor/openpi','checkout','--detach','215abfb217dbac7d5f1273282331b9b1866c0479')
}
if (-not (Test-Path 'vendor/ur_description/.git')) {
    Invoke-Checked 'git' @('clone','--depth','1','--filter=blob:none','--sparse','--branch','rolling','https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git','vendor/ur_description')
    Invoke-Checked 'git' @('-C','vendor/ur_description','sparse-checkout','set','urdf','config','meshes/ur5e/collision')
    Invoke-Checked 'git' @('-C','vendor/ur_description','fetch','--depth','1','origin','d4042c1f5123c86645cc5dbd650b4f461f697ee9')
    Invoke-Checked 'git' @('-C','vendor/ur_description','checkout','--detach','d4042c1f5123c86645cc5dbd650b4f461f697ee9')
}
# Resolve all pinned dependencies together; editable source packages are installed separately.
Get-Content requirements-simenv.lock.txt | Where-Object { $_ -notmatch '^(-e |openpi[= @])' } | Set-Content requirements-inference.lock.txt
Invoke-Checked $simPython @('-m','pip','install','--extra-index-url','https://download.pytorch.org/whl/cu126','-r','requirements-inference.lock.txt')
Invoke-Checked $simPython @('-m','pip','install','--no-deps','-e','vendor/openpi','-e','vendor/openpi/packages/openpi-client')
Invoke-Checked $simPython @('tools/patch_transformers.py')
Invoke-Checked $simPython @('build_scene.py')
Invoke-Checked $simPython @('-u','tools/download_checkpoint.py')
if (-not (Test-Path 'models/pi05_libero_torch/model.safetensors.index.json')) {
    Invoke-Checked $simPython @('-u','tools/convert_pi05_torch.py')
}
Write-Output 'Environment, scene, and CUDA checkpoint ready. Run python run_policy.py --backend cuda --chunks 3 in simenv.'
