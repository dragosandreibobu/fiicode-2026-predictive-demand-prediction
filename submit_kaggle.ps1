param(
    [Parameter(Mandatory = $true)]
    [string]$Competition,

    [Alias("SubmissionPath")]
    [string]$File = "submission.csv",

    [string]$Message = "Auto submission"
)

$ErrorActionPreference = "Stop"

$resolvedFile = Resolve-Path -LiteralPath $File -ErrorAction SilentlyContinue
if (-not $resolvedFile) {
    throw "Submission file not found: $File"
}

$submissionPath = $resolvedFile.Path

$headers = Get-Content -LiteralPath $submissionPath -TotalCount 1
if ($headers -ne "row_id,demand") {
    throw "Invalid submission header. Expected exactly: row_id,demand"
}

if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    throw "kaggle CLI is not installed or not on PATH. Run: python -m pip install --user kaggle"
}

Write-Host "Submitting $submissionPath to competition '$Competition'..."
Write-Host "Message: $Message"
kaggle competitions submit -c $Competition -f $submissionPath -m $Message
