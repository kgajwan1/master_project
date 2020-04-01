c:\
cd C:\Users\gajwa\OneDrive\Desktop\c++practice\master_project
:start 
sample.exe
set /p choice="Do you want to restart benchmark? Press 'y' and enter for Yes: "
if '%choice%'=='y' goto start
if '%choice%'=='n' goto exit
:exit