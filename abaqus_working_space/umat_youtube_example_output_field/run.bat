@echo off

call "C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2018.5.274\windows\bin\ifortvars.bat" intel64
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64
REM call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

REM copy C:\temp\TMS.inp C:\peter_abaqus\Summer-Research-Project\abaqus_working_space\umat_youtube_example_output_field\TMS.inp

C:\SIMULIA\CAE\2019\win_b64\code\bin\ABQLauncher.exe -job TMS1 -user umatTMS
REM C:\SIMULIA\CAE\2019\win_b64\code\bin\ABQLauncher.exe -job umat_random -user umat_elastic
REM run example_2 -job udfluxxx -user udfluxxx

REM shutil.copyfile('C:/temp/umat_test.inp', r'C:/peter_abaqus/Summer-Research-Project/abaqus_working_space/abaqus_out/umat_test.inp')