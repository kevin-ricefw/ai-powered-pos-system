# Deploy veggielens-api to the QA VM (Azure "ImageDetection")
# VM: azureuser@4.153.173.194 | Key: ImageDetection_key.pem (repo root, alongside ai-powered-pos-system/)
# See DEPLOYMENT.md for the manual/equivalent steps and troubleshooting.

$ErrorActionPreference = "Stop"

$VmHost = "4.153.173.194"
$VmUser = "azureuser"
$KeyPath = Join-Path $PSScriptRoot "..\..\ImageDetection_key.pem"
$Image = "veggielens-api:latest"
$Tar = "veggielens-api.tar"

Set-Location $PSScriptRoot

Write-Host "=== Step 1: Build image ===" -ForegroundColor Cyan
docker build -t $Image .

Write-Host "`n=== Step 2: Save image to tar ===" -ForegroundColor Cyan
docker save $Image -o $Tar

Write-Host "`n=== Step 3: Copy image + redeploy script to VM ===" -ForegroundColor Cyan
scp -i $KeyPath $Tar redeploy.sh "${VmUser}@${VmHost}:~/"

Write-Host "`n=== Step 4: Run redeploy on VM (preserves existing mounts/ports) ===" -ForegroundColor Cyan
ssh -i $KeyPath "${VmUser}@${VmHost}" "bash ~/redeploy.sh"

Write-Host "`n=== Step 5: Health check ===" -ForegroundColor Cyan
Start-Sleep -Seconds 5
Invoke-RestMethod "http://${VmHost}:8080/health"

Write-Host "`nDone. Swagger: http://${VmHost}:8080/docs" -ForegroundColor Green
