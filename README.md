# BEBO
Bayesian embeddings for Bayesian optimization.

Code accompanying [Atkinson et al., *Bayesian task embedding for few-shot Bayesian optimization* (2020)](https://arxiv.org/abs/2001.00637)

## Installation
Using Anaconda,
```
conda env create -f environment.yml
conda activate bebo
```

## Running the experiments

### Regression
For example,
```
cd examples\regression
run.bat synthetic BGP 3 4 0 5 1 10
```
runs the toy system ("synthetic") using the Bayesian GP ("BGP") with three 
legacy tasks, 4 data per legacy task, 5 data on the current task, running the
cases corresponding to RNG seeds 1 through 10.
The `0` input is a placeholder and refers to the ID of the task to regard as the
current task of interest (used on the pump and additive examples).

### Bayesian optimization
For example,
```
cd examples\optimization
run.bat synthetic BGP 5 5 1 10
```
runs the toy system ("synthetic") using the Bayesian GP (BGP) model with 5 
legacy tasks, 5 data per legacy task, running the cases corresponding to RNG
seeds 1 though 10.
