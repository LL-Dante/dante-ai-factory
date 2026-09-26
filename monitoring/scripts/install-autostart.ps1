# Run once in an elevated PowerShell only if you choose to enable autostart.
$task = 'DanteAI Hardware Monitoring'
$action = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument '"C:\DanteAI\monitoring\scripts\start-monitoring.vbs"'
$trigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger -Description 'Starts the local, read-only DanteAI hardware monitor.' -Force
# To disable later: Disable-ScheduledTask -TaskName $task
# To remove later: Unregister-ScheduledTask -TaskName $task -Confirm:$false
