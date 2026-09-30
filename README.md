# mma3001-project
Predictive maintenance and remaining useful life estimation using NASA C-MAPSS turbofan data
## Initial Resutls 
The FD001 training dataset contained 20,631 observations from 100
simulated turbofan engines. Constant sensor columns were removed before
model development.

The data was separated by engine ID to ensure that observations from the
same engine did not appear in both the training and validation datasets.
This reduced the risk of data leakage.

Three approaches were evaluated:

- Mean RUL baseline
- Linear regression
- Random Forest regression

Performance was assessed using mean absolute error (MAE), root mean
squared error (RMSE), and the coefficient of determination (R²).
The results are stored in `results/metrics/initial_model_results.csv`.
