@echo off
REM Deploy veggielens-api to the QA VM (Azure "ImageDetection")
REM VM: azureuser@4.153.173.194 | Key: ImageDetection_key.pem (repo root, alongside ai-powered-pos-system/)
REM See DEPLOYMENT.md for the manual/equivalent steps and troubleshooting.

set VMHOST=4.153.173.194
set VMUSER=azureuser
set KEYPATH=%~dp0..\..\ImageDetection_key.pem
set IMAGE=veggielens-api:latest
set TAR=veggielens-api.tar

cd /d "%~dp0"

echo === Step 1: Build image ===
docker build -t %IMAGE% .
if errorlevel 1 goto :error

echo.
echo === Step 2: Save image to tar ===
docker save %IMAGE% -o %TAR%
if errorlevel 1 goto :error

echo.
echo === Step 3: Copy image + redeploy script to VM ===
scp -i "%KEYPATH%" %TAR% redeploy.sh %VMUSER%@%VMHOST%:~/
if errorlevel 1 goto :error

echo.
echo === Step 4: Run redeploy on VM (preserves existing mounts/ports) ===
ssh -i "%KEYPATH%" %VMUSER%@%VMHOST% "bash ~/redeploy.sh"
if errorlevel 1 goto :error

echo.
echo === Step 5: Health check ===
timeout /t 5 /nobreak >nul
curl http://%VMHOST%:8080/health

echo.
echo Done. Swagger: http://%VMHOST%:8080/docs
exit /b 0

:error
echo.
echo Deploy failed at the step above.
exit /b 1
