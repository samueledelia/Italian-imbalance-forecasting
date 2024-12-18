dim shell
Set objShell = Wscript.CreateObject("WScript.Shell")
objShell.Run "C:\imbalance_forecast\notebook\BATCHFILES\sign_forecast.bat", 2, True
set shell=nothing
WScript.Quit