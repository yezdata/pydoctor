$ErrorActionPreference = 'Stop'

$Repo = "yezdata/pydoctor"
$ModelRepo = "yezdata/SmolLM2-1.7B-Instruct-DocstringGenerator"
$ModelFile = "smollm2_1_7b_instruct_merged-q8_0.gguf"

$InstallDir = "$env:LOCALAPPDATA\pydoctor"
$CacheDir = "$env:LOCALAPPDATA\pydoctor"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

Write-Host "Downloading PyDoctor CLI for Windows..."
$Url = "https://github.com/$Repo/releases/latest/download/pydoctor-windows-amd64.exe"
Invoke-WebRequest -Uri $Url -OutFile "$InstallDir\pydoctor.exe"

$UserPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($UserPath -notlike "*$InstallDir*") {
    Write-Host "Adding $InstallDir to User PATH..."
    [Environment]::SetEnvironmentVariable("PATH", "$UserPath;$InstallDir", "User")
}

$ModelPath = Join-Path $CacheDir $ModelFile
if (!(Test-Path $ModelPath)) {
    Write-Host "Downloading PyDoctor model..."
    $ModelUrl = "https://huggingface.co/$ModelRepo/resolve/main/$ModelFile"
    
    Invoke-WebRequest -Uri $ModelUrl -OutFile "$ModelPath.download"
    Move-Item -Path "$ModelPath.download" -Destination $ModelPath -Force
    Write-Host "Model successfully cached."
}

Write-Host "--------------------------------------------------------"
Write-Host "PyDoctor successfully installed to: $InstallDir\pydoctor.exe"
Write-Host "Please restart your terminal/IDE for the PATH changes to take effect."
