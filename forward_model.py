from functional import seq, pseq
from typing import Tuple, List

# For file reading and writing
from pathlib import Path
import csv
import ast

from math import sin, pi, tanh
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import numpy as np

import matplotlib.pyplot as plt


DIR = Path(__file__).resolve().parent
def generate_training_samples(Nsamples, N=1000, filename=f"{DIR}/out/training_data.csv"):
    print(f"brrr generating {Nsamples} training samples to file:")

    # Constants
    RSD = 5     # <- Stochastic Dimension
    FILE_OFFSET=4 # CSV body starts at line 5, index 4

    file_exists = Path(filename).exists()

    # Calculate how many samples to take based on samples in file
    samples_in_file = 0
    if file_exists:
        print(f"--> file exists: {filename}")
        with open(filename) as f:
            samples_in_file = sum(1 for _ in f) - FILE_OFFSET

    samples_to_draw = Nsamples - samples_in_file
    print(f"--> {Nsamples} samples requested, file contains: {samples_in_file} samples.")
    print(f"--> Drawing {samples_to_draw if (samples_to_draw >= 0) else 0} samples")
    if samples_to_draw <= 0:
        return

    rng = np.random.default_rng()
    Q_train = rng.normal(0.0, 1.0, size=(Nsamples, RSD))
    F_train = (pseq(Q_train).map(lambda qi: stochastic_diffusion_model(qi, N))
               .to_list())

    x_vals = F_train[0][0]
    u_vals = seq(F_train).map(lambda f: f[1]).to_list()

    # Write a header of x_vals if the file doesn't exist
    if not file_exists:
        print(f"--> creating file: {filename}")
        with open(filename, "w") as f:
            f.write("x_vals:\n")
            f.write(str(x_vals))
            f.write("\n\n")
            f.write("q,u\n")

    # Append Q_train and u_vals in csv format
    with open(filename) as f:
        next(f)                # skip first line
        line = next(f)         # read second line
        x_vals_file = ast.literal_eval(line)
        if len(x_vals_file) != len(x_vals):
            raise Exception("Different resolution of forward model. ABORTING")

    # Append Q_train and u_vals in csv format
    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)
        for q_row, u_row in zip(Q_train, u_vals):
            writer.writerow([q_row.tolist(), u_row.tolist()])


R5 = Tuple[float, float, float, float, float]
def stochastic_diffusion_model(xi: R5, N: int) -> Tuple[List[float], List[float]]:
    # 0. Construct Discretization
    L = 1 # Length of domain
    dx = L/N
    def i2x(idx: int) -> float: # Index to x coordinate
        return float(idx)*dx + 0.5*dx

    #print(list(enumerate(xi)))
    # 1. Construct A Matrix (Sparse)
    def nu(x):
        v0 = 0.1
        alpha = 0.5
        tan_term = sum([xi_i * sin((idx+1) *pi*x) for idx, xi_i in enumerate(xi)])
        return v0 * (1.0 + alpha * tanh(tan_term))

    main = (seq(range(N))
             .map(lambda i: nu(i2x(i)+0.5*dx) + nu(i2x(i)-0.5*dx))
             .to_list())
    sub   = (seq(range(N-1))
             .map(lambda i: i+1)
             .map(lambda i: -1* nu(i2x(i)-0.5*dx))
             .to_list())
    upper = (seq(range(N-1))
             .map(lambda i: -1 * nu(i2x(i)+0.5*dx))
             .to_list())
    A = diags(
        diagonals=[sub, main, upper],
        offsets=[-1, 0, 1],
        format="csr"
    )

    # 2. Construct dx^2 matrix
    s = (lambda x : sin(pi * x))
    rhs = [dx**2 * s(i2x(i)) for i in range(N)]

    # 3. Solve sytem
    xs = list(map(i2x ,range(N)))
    u = spsolve(A,rhs)

    return xs, u
#fn done



def inspect_behavior(N: int = 1000, M: int = 100):
    # Qualitative test to make sure stochastic diffusion model is working correctly
    # * N = Number of points to solve model at
    # * M = Number of samples to draw
    rng = np.random.default_rng()

    # 1. Generate M xi vectors
    xis = rng.normal(0.0, 1.0, size=(M, 5))

    # 2. Run the model for each
    def run_model(m):
        _, u = stochastic_diffusion_model(xis[m], N)
        return u
    U = np.array([run_model(m) for m in range(M)])

    # 3. Compute mean @ each N
    mean = np.mean(U, axis=0)
    #ci_low = np.quantile(U, 0.025, axis=0)
    #ci_high= np.quantile(U, 0.975, axis=0)
    std = np.std(U, axis=0)
    ci_low = mean-(2*std)
    ci_high= mean+(2*std)
    x, _ = stochastic_diffusion_model(xis[0], N)

    # 4. Plot
    plt.figure(figsize=(8,5))
    ## individual realizations
    for m in range(M):
        plt.plot(x, U[m], color="blue", alpha=0.12, linewidth=1)
    ## 95% CI
    plt.fill_between(x, ci_low, ci_high, color="gray", alpha=0.3, label="95% CI")
    ## mean
    plt.plot(x, mean, color="black", linewidth=2, label="Mean")
    ## Meta data
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.xlim(0, 1)
    plt.ylim(0, 1.8)
    plt.xticks(np.linspace(0, 1, 6))        # 0, 0.2, ..., 1
    plt.yticks(np.linspace(0, 1.8, 7))      # nice spacing
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.ylim(0,2.0)
    plt.title(f"{M} Realizations (Stable Stochastic Diffusion)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    inspect_behavior(N=2000, M=100)
    #rng = np.random.default_rng()
    #xi = rng.normal(loc=0.0, scale=1.0, size=5)
    #print("xi: ", xi)
    #[x,u] = stochastic_diffusion_model(xi, 1000)

    #plt.figure()
    #plt.plot(x, u)
    #plt.xlabel("x")
    #plt.xlim(0,1)
    #plt.ylim(0,1.8)
    #plt.ylabel("u(x)")
    #plt.title("Stochastic Diffusion Model")
    #plt.show()
