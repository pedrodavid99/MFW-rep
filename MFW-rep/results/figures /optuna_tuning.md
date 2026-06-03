# this file will present the logs of th eoptuna tunners for the lightgbm´s models

i used 20 trials per study in both lightgbm's, and a 5 crossfolds 
the main findings for the lightgbm's when it comes to HP optimization are:

------------------------------------------------------------------------------for  the lapse model------------------------------------------------------------------------------
the study included th esearch space :
```python
params = {
    'n_estimators':      trial.suggest_int('n_estimators', 100, 600),
    'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
    'num_leaves':        trial.suggest_int('num_leaves', 20, 150),
    'max_depth':         trial.suggest_int('max_depth', 3, 10),
    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
    'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
    'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
    'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 5.0, log=True),
    'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 5.0, log=True),
}
```
the study took 10 min to run  and yielded the folowing results 
<img width="616" height="829" alt="image" src="https://github.com/user-attachments/assets/a2454507-3a88-4137-a18f-95e14eeba573" />

 <img width="1176" height="623" alt="image" src="https://github.com/user-attachments/assets/bfb4996b-5ad7-49a6-b38c-9dfec14edeb0" />

<img width="1168" height="657" alt="image" src="https://github.com/user-attachments/assets/2f6198d3-f729-4950-b5fd-372abd376546" />

<img width="2700" height="450" alt="image" src="https://github.com/user-attachments/assets/d68ce83c-43f2-4b1a-8a80-a8c211e86771" />
<img width="1164" height="626" alt="image" src="https://github.com/user-attachments/assets/9b885477-46af-4e18-84ce-8bca52922138" />

-
-
the best trial was the 11th trial after that trials cluster about the same objective value. Max depth is without a doubt the moast important hyperparameter, folowed by the learning rate  and the number of estimators in the ensemble 
