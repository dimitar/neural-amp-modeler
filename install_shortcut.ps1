# Creates a desktop shortcut that launches the NAM Parametric Trainer.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktop = [Environment]::GetFolderPath("Desktop")
$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut((Join-Path $desktop "NAM Trainer.lnk"))
$sc.TargetPath = (Join-Path $here "launch_trainer.bat")
$sc.WorkingDirectory = $here
$sc.WindowStyle = 7  # minimized — hides the brief launcher console
$icon = Join-Path $here "trainer_app\assets\icon.ico"
if (Test-Path $icon) { $sc.IconLocation = $icon }
$sc.Save()
Write-Host "Created desktop shortcut: NAM Trainer"
