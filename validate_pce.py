from pathlib import Path
from typing import Dict, Sequence
from forward_model import generate_training_samples
from functional import seq, pseq
import numpy as np
import matplotlib.pyplot as plt

# For file reading and writing
import csv
import ast

# Local Deps
def set_cwd():
    import sys
    import os
    sys.path.insert(1, os.path.dirname(__file__))
set_cwd()
DIR = Path(__file__).resolve().parent

from pce import train_pce, predict_pce
import pce
import kle #TODO: lol fix this such that error metrics are shared

def get_u(Nsamples: int,
          filename:str = f"{DIR}/out/training_data.csv"):
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

    q_vals = (pseq(run_vals)
              .map(lambda run: run['q'])
              .map(ast.literal_eval)
              .to_list())
    u_vals = (pseq(run_vals)
              .map(lambda run: run['u'])
              .map(ast.literal_eval)
              .to_list())

    if Nsamples > len(u_vals):
        raise Exception(f"Requested more samples, {Nsamples}, than exists in the file {len(u_vals)}")

    return x_vals, np.array(q_vals[:Nsamples]), np.array(u_vals[:Nsamples])


def perform_pce(Nsamples: int, P: int,
                filename:str = f"{DIR}/out/training_data.csv"):
    print(f"Performing PCE with Polynomial Order {P} on {Nsamples}")
    generate_training_samples(Nsamples*2, N=1501)
    print("--> Retrieving data")
    x_vals, Q_set, F_set= seq(get_u(Nsamples*2, filename))
    Q_train = Q_set[:Nsamples]
    F_train = F_set[:Nsamples]
    Q_val = Q_set[Nsamples:]
    F_val = F_set[Nsamples:]
    print("--> Training PCE")
    pce_meta = train_pce(Q_train, F_train, "normal", P)
    print("--> Predicting Outsample")
    F_tild = predict_pce(pce_meta, Q_val)

    return x_vals, F_val, F_tild

def rmse_plots(x_vals, rmses, nrmses):
    plt.figure(figsize=(8,5))
    plt.plot(x_vals, rmses, label='RMSE')
    plt.plot(x_vals, nrmses, label='NRMSE')
    plt.xlabel("x")
    plt.legend()

def closest_idx(target: float, arr: np.ndarray) -> int:
    return np.abs(arr - target).argmin()

def calc_error(x_vals, F_true, F_approx):
    # 0.25
    qidx = closest_idx(np.array(x_vals), 0.25)
    qrmse = kle.RMSE(F_true[:,qidx], F_approx[:,qidx])

    # 0.5
    hidx = closest_idx(np.array(x_vals), 0.5)
    hrmse = kle.RMSE(F_true[:,hidx], F_approx[:,hidx])
    return qrmse, hrmse

def plot_rmses(polys, rmses):
    q_rmses = list(map(lambda pair: pair[0],rmses))
    h_rmses = list(map(lambda pair: pair[1],rmses))

    plt.figure(figsize=(8,5))
    plt.plot(polys, q_rmses, label='NRMSE(u(0.25))')
    plt.plot(polys, h_rmses, label='NRMSE(u(0.5))')
    plt.xlabel("Polynomial Order")
    plt.ylabel("RMSE")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.title("PCE - How RMSE changes with Polynomial Order")
    plt.legend()

def plot_convergence(samples, rmses):
    q_rmses = list(map(lambda pair: pair[0],rmses))
    h_rmses = list(map(lambda pair: pair[1],rmses))

    plt.figure(figsize=(8,5))
    plt.plot(samples, q_rmses, label='RMSE(u(0.25))')
    plt.plot(samples, h_rmses, label='RMSE(u(0.5))')
    plt.xticks([100, 1000, 10000, 100000])
    plt.xlabel("Nsamples")
    plt.ylabel("RMSE")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.title("PCE Convergence - P=3")
    plt.legend()


if __name__ == "__main__":
    print("Hello world")
    #generate_training_samples(200000, N=1501)

    def calc_rmses(poly: int):
        x_vals, F_val, F_tild = perform_pce(1000, poly)
        return calc_error(x_vals, F_val, F_tild)

    rmses = list(map(calc_rmses, range(0,7)))
    plot_rmses(range(0,7), rmses)

    #def calc_sample_convergence(Nsamples, poly=3):
    #    x_vals, F_val, F_tild = perform_pce(Nsamples, poly)
    #    return calc_error(x_vals, F_val, F_tild)

    #Nsamples = [100,1000,10000, 100000]
    #rmses = list(map(calc_sample_convergence, Nsamples))
    #plot_convergence(Nsamples, rmses)

    plt.show()
