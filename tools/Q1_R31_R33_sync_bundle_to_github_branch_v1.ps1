param(
    [string]$BundleZip = "F:\MEDSEG_SAFETTA\R31_R33_public_script_bundle_v1.zip",
    [string]$WorkParent = "F:\MEDSEG_SAFETTA",
    [string]$RepoUrl = "https://github.com/kewanglong22-lab/SafeTTA.git",
    [string]$Branch = "r31-r33-public-sync-20260912"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-UniquePath([string]$BasePath) {
    if (-not (Test-Path $BasePath)) { return $BasePath }
    $i = 1
    while ($true) {
        $p = "${BasePath}_fix$i"
        if (-not (Test-Path $p)) { return $p }
        $i++
    }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

Write-Host ("=" * 110)
Write-Host "SafeTTA R31-R33 exact-script GitHub synchronization"
Write-Host ("=" * 110)
Write-Host "Bundle : $BundleZip"
Write-Host "Branch : $Branch"

if (-not (Test-Path -LiteralPath $BundleZip -PathType Leaf)) {
    throw "Bundle ZIP not found: $BundleZip"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git was not found in PATH."
}
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found in PATH."
}

$bundleSha = Get-Sha256 $BundleZip
Write-Host "Bundle SHA256: $bundleSha"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$extractDir = Get-UniquePath (Join-Path $WorkParent "R31_R33_bundle_extract_$stamp")
$repoDir = Get-UniquePath (Join-Path $WorkParent "SafeTTA_github_sync_$stamp")

New-Item -ItemType Directory -Force -Path $extractDir | Out-Null
Expand-Archive -LiteralPath $BundleZip -DestinationPath $extractDir -Force

$roots = Get-ChildItem -LiteralPath $extractDir -Directory
if ($roots.Count -ne 1) {
    throw "Expected exactly one top-level directory in ZIP; observed $($roots.Count)."
}
$bundleRoot = $roots[0].FullName

$audit = Join-Path $bundleRoot "provenance\R31_R33\AUDIT_REPORT.txt"
if (-not (Test-Path -LiteralPath $audit)) { throw "AUDIT_REPORT.txt missing from bundle." }
$auditText = Get-Content -LiteralPath $audit -Raw
if ($auditText -notmatch "GATE=PASS_R31_R33_22_FINAL_SCRIPTS_COLLECTED_AND_SYNTAX_CHECKED") {
    throw "Bundle audit gate is not PASS."
}
Write-Host "Bundle audit gate: PASS"

Write-Host "Cloning synchronization branch..."
git clone --branch $Branch --single-branch $RepoUrl $repoDir
if ($LASTEXITCODE -ne 0) { throw "git clone failed." }

Push-Location $repoDir
try {
    $headBefore = (git rev-parse HEAD).Trim()
    Write-Host "Branch HEAD before sync: $headBefore"

    $manifestPath = Join-Path $repoDir "provenance\R31_R33\R31_R33_SHA256.csv"
    if (-not (Test-Path -LiteralPath $manifestPath)) {
        throw "Public SHA manifest is missing from branch: $manifestPath"
    }
    $manifest = Import-Csv -LiteralPath $manifestPath
    if ($manifest.Count -ne 28) {
        throw "Expected 28 SHA manifest rows (22 final + 6 ancestors); observed $($manifest.Count)."
    }

    $copied = 0
    foreach ($row in $manifest) {
        $rel = ($row.relative_path -replace '/', '\')
        $src = Join-Path $bundleRoot $rel
        $dst = Join-Path $repoDir $rel
        if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
            throw "Bundle file missing: $rel"
        }
        $parent = Split-Path -Parent $dst
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
        Copy-Item -LiteralPath $src -Destination $dst -Force

        $got = Get-Sha256 $dst
        $exp = ([string]$row.sha256).ToLowerInvariant()
        if ($got -ne $exp) {
            throw "SHA mismatch after copy: $rel expected=$exp observed=$got"
        }
        $copied++
        Write-Progress -Activity "Copy + SHA verify R31-R33 scripts" -Status "$copied / $($manifest.Count)" -PercentComplete (($copied / $manifest.Count) * 100)
    }
    Write-Progress -Activity "Copy + SHA verify R31-R33 scripts" -Completed
    Write-Host "SHA verification: $copied/$($manifest.Count) PASS"

    $pyFiles = @()
    $pyFiles += Get-ChildItem -LiteralPath (Join-Path $repoDir "experiments\R31_action_transfer") -Filter "*.py" -File
    $pyFiles += Get-ChildItem -LiteralPath (Join-Path $repoDir "experiments\R31_controller_negative") -Filter "*.py" -File
    $pyFiles += Get-ChildItem -LiteralPath (Join-Path $repoDir "experiments\R32_validation") -Filter "*.py" -File
    $pyFiles += Get-ChildItem -LiteralPath (Join-Path $repoDir "experiments\R33_external_joint_shift") -Filter "*.py" -File
    $pyFiles += Get-ChildItem -LiteralPath (Join-Path $repoDir "provenance\R31_R33\sha_bound_ancestors") -Filter "*.py" -File

    if ($pyFiles.Count -ne 28) {
        throw "Expected 28 synchronized Python files; observed $($pyFiles.Count)."
    }

    $i = 0
    foreach ($f in $pyFiles) {
        $i++
        python -m py_compile $f.FullName
        if ($LASTEXITCODE -ne 0) { throw "py_compile failed: $($f.FullName)" }
        Write-Progress -Activity "py_compile audit" -Status "$i / $($pyFiles.Count)" -PercentComplete (($i / $pyFiles.Count) * 100)
    }
    Write-Progress -Activity "py_compile audit" -Completed
    Write-Host "py_compile: 28/28 PASS"

    python code/Q1_R31_R33_public_paper_stat_replay_v1.py --root .
    if ($LASTEXITCODE -ne 0) { throw "R31-R33 public replay failed." }

    git add -- \
        experiments/R31_action_transfer \
        experiments/R31_controller_negative \
        experiments/R32_validation \
        experiments/R33_external_joint_shift \
        provenance/R31_R33/sha_bound_ancestors
    if ($LASTEXITCODE -ne 0) { throw "git add failed." }

    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw "git diff --cached --check failed." }

    $staged = git diff --cached --name-only
    if (-not $staged) { throw "No staged changes detected." }
    Write-Host "Staged files:"
    $staged | ForEach-Object { Write-Host "  $_" }

    git commit -m "Add exact retained R31-R33 experiment scripts"
    if ($LASTEXITCODE -ne 0) { throw "git commit failed." }

    $commitSha = (git rev-parse HEAD).Trim()
    Write-Host "Created commit: $commitSha"

    git push origin "HEAD:$Branch"
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed. Check your GitHub authentication/credential manager."
    }

    Write-Host ("=" * 110)
    Write-Host "PASS_R31_R33_GITHUB_SCRIPT_SYNC"
    Write-Host "Branch  : $Branch"
    Write-Host "Commit  : $commitSha"
    Write-Host "Scripts : 22 final + 6 SHA-bound ancestors"
    Write-Host "Replay  : PASS"
    Write-Host ("=" * 110)
}
finally {
    Pop-Location
}
