$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'


function Download-WithProgress {
    param (
        [string]$Url,
        [string]$DestinationPath,
        [string]$DisplayName
    )

    [void][System.Reflection.Assembly]::LoadWithPartialName("System.Net.Http")

    $HttpClient = [System.Net.Http.HttpClient]::new()
    
    try {
        $Response = $HttpClient.GetAsync($Url, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        $Response.EnsureSuccessStatusCode() | Out-Null

        $TotalBytes = $Response.Content.Headers.ContentLength
        $ResponseStream = $Response.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
        $FileStream = [System.IO.FileStream]::new($DestinationPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)

        $Buffer = [byte[]]::new(1048576)
        $TotalRead = 0

        while ($true) {
            $Read = $ResponseStream.Read($Buffer, 0, $Buffer.Length)
            if ($Read -eq 0) { break }

            $FileStream.Write($Buffer, 0, $Read)
            $TotalRead += $Read

            if ($TotalBytes) {
                $Percent = ($TotalRead / $TotalBytes) * 100
                $TotalReadMB = $TotalRead / 1MB
                $TotalBytesMB = $TotalBytes / 1MB
                
                [System.Console]::Write([string]::Format("`rDownloading {0}: {1:F1}% ({2:F1}/{3:F1} MB)", $DisplayName, $Percent, $TotalReadMB, $TotalBytesMB))
            }
        }
    }
    finally {
        if ($FileStream) {
            $FileStream.Close()
            $FileStream.Dispose()
        }
        if ($ResponseStream) {
            $ResponseStream.Close()
            $ResponseStream.Dispose()
        }
        if ($HttpClient) {
            $HttpClient.Dispose()
        }
        [System.Console]::WriteLine()
    }
}


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

$ModelUrl = "https://huggingface.co/$ModelRepo/resolve/main/$ModelFile"
$ModelPath = Join-Path $CacheDir $ModelFile
$TempModelPath = "$ModelPath.download"

if (!(Test-Path $ModelPath)) {
    if (Test-Path $TempModelPath) {
        Remove-Item -Path $TempModelPath -Force -ErrorAction Stop
    }

    try {
        Download-WithProgress -Url $ModelUrl -DestinationPath $TempModelPath -DisplayName "Model"
        Move-Item -Path $TempModelPath -Destination $ModelPath -Force
        Write-Host "Model successfully cached."
    } catch {
        if (Test-Path $TempModelPath) {
            try { Remove-Item -Path $TempModelPath -Force } catch {}
        }
        throw $_
    }
}

$env:PATH = "$env:PATH;$InstallDir"

Write-Host "--------------------------------------------------------"
Write-Host "PyDoctor successfully installed to: $InstallDir\pydoctor.exe"
