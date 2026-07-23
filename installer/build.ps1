param(
    [switch]$AllowUnsigned,
    [string]$TimestampUrl = "http://timestamp.digicert.com"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

python -m PyInstaller --clean --noconfirm MeetingRecorder.spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed."
}

$IsccCommand = Get-Command iscc.exe -ErrorAction SilentlyContinue
if ($IsccCommand) {
    $IsccPath = $IsccCommand.Source
} else {
    $DefaultIscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $DefaultIscc) {
        $IsccPath = $DefaultIscc
    } else {
        throw "Inno Setup 6 was not found. Install it or add iscc.exe to PATH."
    }
}

$Thumbprint = $env:SIGN_CERT_SHA1
if (-not $Thumbprint -and -not $AllowUnsigned) {
    throw "Set SIGN_CERT_SHA1 to a code-signing certificate thumbprint, or pass -AllowUnsigned for a local test build."
}

function Sign-Artifact([string]$Path) {
    if (-not $Thumbprint) {
        return
    }
    & signtool.exe sign /sha1 $Thumbprint /fd SHA256 /tr $TimestampUrl /td SHA256 $Path
    if ($LASTEXITCODE -ne 0) {
        throw "Signing failed for $Path."
    }
}

Sign-Artifact "$Root\dist\MeetingRecorder\MeetingRecorder.exe"
& $IsccPath "$Root\installer\MeetingRecorder.iss"
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed."
}
Sign-Artifact "$Root\dist\MeetingRecorder-0.1.0-Setup.exe"

Write-Host "Created dist\MeetingRecorder-0.1.0-Setup.exe"
