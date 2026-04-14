from pathlib import Path
from typing import Dict, Sequence
from functional import seq, pseq
from kle import train_kle
import numpy as np

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

from kle import train_kle
import forward_model

def perform_kle(Nsamples: int, filename:str = f"{DIR}/out/training_data.csv"):
    Q_train, F_train = get_u(Nsamples, filename)
    print(f"Q_train: {np.array(Q_train).shape}, F_train: {np.array(F_train).shape}")
    trained_kle = train_kle(np.array(Q_train), np.array(F_train))

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

    return q_vals[:Nsamples], u_vals[:Nsamples]


if __name__ == "__main__":
    forward_model.generate_training_samples(Nsamples=10, filename=f"{DIR}/out/training_data-N100.csv")
    #s1 = get_u(Nsamples=1)
    perform_kle(10, filename=f"{DIR}/out/training_data-N100.csv")
    #1. Create training data -> Write to a file
    #2. Train KLE for all of U on a value of xi (Rout = 1000, Rsd=5)
    #   Create a function that takes in a number of samples, reads that many from the file, then calculates:
    #   -> See how many modes are retained
    #   -> Show in sample error
