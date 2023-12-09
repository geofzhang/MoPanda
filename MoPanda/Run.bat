@echo off
SET ANACONDA_PROMPT=C:\Users\%USERNAME%\AppData\Local\anaconda3

REM Check if the Anaconda Prompt exists
IF NOT EXIST "%ANACONDA_PROMPT%" (
    ECHO "Anaconda Prompt not found."
    ECHO "Please ensure Anaconda is installed at %ANACONDA_PROMPT%."
    GOTO End
)

CALL "%ANACONDA_PROMPT%\Scripts\activate.bat"

REM List of environment names to check
SET ENVS=mopanda MoPanda MOPANDA MoPAnDA denova

REM Loop through each environment name
FOR %%E IN (%ENVS%) DO (
    REM Check if the environment exists
    conda info --envs | findstr /I "%%E" >nul
    IF NOT ERRORLEVEL 1 (
        ECHO Found environment: %%E
        CALL conda activate %%E
        CALL python main.py
        CALL conda deactivate
        GOTO End
    )
)

ECHO No specified environment found.

:End
pause
