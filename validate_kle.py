from pathlib import Path
from typing import Dict, Sequence
from functional import seq, pseq
import numpy as np
import matplotlib.pyplot as plt

# For file reading and writing
import csv
import ast

from pprint import pprint

# Local Deps
def set_cwd():
    import sys
    import os
    sys.path.insert(1, os.path.dirname(__file__))
set_cwd()
DIR = Path(__file__).resolve().parent

from kle import train_kle, predict_kle
import kle
import forward_model

def perform_kle(Nsamples: int, filename:str = f"{DIR}/out/training_data.csv"):
    print(f"Performing KLE on {Nsamples}")
    print("--> Retrieving data")
    x_vals, Q_set, F_set= seq(get_u(Nsamples*2, filename))
    Q_train = Q_set[:Nsamples]
    F_train = F_set[:Nsamples]
    Q_val = Q_set[Nsamples:]
    F_val = F_set[Nsamples:]
    print("--> Training KLE")
    trained_kle = train_kle(Q_train, F_train)
    print("--> Predicting outsample")
    F_tild = predict_kle(trained_kle, Q_val)

    return x_vals, F_val, F_tild

def get_u(Nsamples: int, filename:str = f"{DIR}/out/training_data.csv"):
    def get_x_vals(filename: str) -> Sequence[float]:
        with open(filename) as f:
            next(f)                # skip first line
            line = next(f)         # read second line
            return ast.literal_eval(line)
    def get_run_vals(filename: str) -> Dict:
        with open(filename, newline="") as f:
            next(f)
            next(f)
            next(f)
            # CSV content starts on line 5 (index 4)
            reader = csv.DictReader(f)
            data = list(reader)
            return data
    x_vals = get_x_vals(filename)
    run_vals = get_run_vals(filename)

    q_vals = (seq(run_vals)
              .map(lambda run: run['q'])
              .map(ast.literal_eval)
              .to_list())
    u_vals = (seq(run_vals)
              .map(lambda run: run['u'])
              .map(ast.literal_eval)
              .to_list())

    if Nsamples > len(u_vals):
        raise Exception(f"Requested more samples, {Nsamples}, than exists in the file {len(u_vals)}")

    return x_vals, np.array(q_vals[:Nsamples]), np.array(u_vals[:Nsamples])

def calc_errors(x_vals, F_true, F_approx):
    print("--> Computing RMSE metrics")
    rmses = (seq(zip(F_true.T, F_approx.T))
             .map(lambda Fx: kle.RMSE(Fx[0],Fx[1]))
             .to_list())
    nrmses = (seq(zip(F_true.T, F_approx.T))
              .map(lambda Fx: kle.NRMSE(Fx[0],Fx[1]))
              .to_list())
    rmse_plots(x_vals, rmses, nrmses)

def rmse_plots(x_vals, rmses, nrmses):
    plt.figure(figsize=(8,5))
    plt.plot(x_vals, rmses, label='RMSE')
    plt.plot(x_vals, nrmses, label='NRMSE')
    plt.xlabel("x")
    plt.title("Errors v x")
    plt.legend()

def show_f(x_vals, F):
    N = F.shape[0]
    mean = np.mean(F, axis=0)
    std = np.std(F, axis=0)
    ci_low = mean-(2*std)
    ci_high= mean+(2*std)

    plt.figure(figsize=(8,5))
    ## mean
    plt.plot(x_vals, mean, color="black", linewidth=2, label="Mean")
    ## 95% CI
    plt.fill_between(x_vals, ci_low, ci_high, color="gray", alpha=0.3, label="95% CI")

    ## Meta data
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.xlim(0, 1)
    plt.ylim(0, 1.8)
    plt.xticks(np.linspace(0, 1, 6))        # 0, 0.2, ..., 1
    plt.yticks(np.linspace(0, 1.8, 7))      # nice spacing
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.ylim(0,2.0)
    plt.title(f"{N} Realizations (Stable Stochastic Diffusion)")
    plt.legend()

def compare_val_true(x_vals, F_tild, F_val):
    N = F_tild.shape[0]
    mean_tild = np.mean(F_tild, axis=0)
    std_tild = np.std(F_tild, axis=0)
    ci_low_tild = mean_tild-(2*std_tild)
    ci_high_tild= mean_tild+(2*std_tild)

    mean_val = np.mean(F_val, axis=0)
    std_val = np.std(F_val, axis=0)
    ci_low_val = mean_val-(2*std_val)
    ci_high_val = mean_val +(2*std_val)

    plt.figure(figsize=(8,5))
    ## mean
    plt.plot(x_vals, mean_tild, color="black", linewidth=2, label="Mean_prediction")
    plt.plot(x_vals, mean_val, color="green", linewidth=2, label="Mean_true")
    ## 95% CI
    plt.fill_between(x_vals, ci_low_tild, ci_high_tild, color="gray", alpha=0.3, label="95% CI prediction")
    plt.fill_between(x_vals, ci_low_val, ci_high_val, color="lightgreen", alpha=0.3, label="95% CI prediction")

    ## Meta data
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.xlim(0, 1)
    plt.ylim(0, 1.8)
    plt.xticks(np.linspace(0, 1, 6))        # 0, 0.2, ..., 1
    plt.yticks(np.linspace(0, 1.8, 7))      # nice spacing
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.ylim(0,2.0)
    plt.title(f"{N} Realizations (Stable Stochastic Diffusion)")
    plt.legend()

if __name__ == "__main__":
    forward_model.generate_training_samples(2000)
    x_vals, F_val, F_tild = perform_kle(1000)
    calc_errors(x_vals, F_val, F_tild)
    compare_val_true(x_vals, F_tild, F_val)
    plt.show()

