REM run.bat 
REM [system] 
REM [model type] 
REM [num legacy] 
REM [data per legacy] 
REM [current task] (Pump and Additive)
REM [n train] 
REM [first seed]
REM [last seed]

set system=%1
set model=%2
set nleg=%3
set dpleg=%4
set ctask=%5
set ntrain=%6
set s1=%7
set s2=%8

for /l %%s in (%s1%, 1, %s2%) do (
    python main.py --seed %%s --system %system% --model %model% --save ^
    --num-legacy %nleg% --data-per-legacy %dpleg% --current-task %ctask% ^
    --train-current %ntrain%
)
