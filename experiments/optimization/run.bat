REM run.bat [system] [model type] [num legacy] [data per legacy] [first seed] [last seed]

set system=%1
set model=%2
set nleg=%3
set dpleg=%4
set s1=%5
set s2=%6

REM set output_log="output\%system%\logs\%model%"
REM mkdir %output_log%

for /l %%s in (%s1%, 1, %s2%) do (
    python main.py --seed %%s --system %system% --model %model% --save ^
    --num-legacy %nleg% --data-per-legacy %dpleg%
)
REM > %output_log%\%%s.log 2>&1
