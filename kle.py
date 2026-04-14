from typing import Sequence, TypedDict
from functional import seq,pseq

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

from pprint import pprint

class KLE(TypedDict):
    Fbar: np.ndarray      # Vector: Rout tall
    lam_trunc: np.ndarray # Vector: Ntrunc tall
    V_trunc:   np.ndarray # Matrix: Ntrunc x Ntrunc
    gps:       Sequence[GaussianProcessRegressor] # Vector: Ntrunc tall
def Rout(kle: KLE) -> int:
    return kle['Fbar'].shape
def Ntrunc(kle: KLE) -> int:
    return kle['lam_trunc'].shape

class EigenDict(TypedDict):
    λ_i: Sequence[float]
    v_i: Sequence[Sequence[float]]
def truncate_eigenmodes(eigenvalues: Sequence[float],
                        eigenvectors:Sequence[Sequence[float]]) -> EigenDict:
    Rout = len(eigenvalues)
    # Couple eigenvalues and eigenvectors
    eigen_pairs = [
        {"λ_i": lam, "v_i": v}
        for lam, v in zip(eigenvalues, eigenvectors)
    ]

    # Sort in decending order of eigenvalue
    eig_dom = sorted(
        eigen_pairs,
        key=lambda d: d["λ_i"],
        reverse=True  # descending
    )
    val_dom = list(map(lambda mode: mode["λ_i"], eig_dom))

    # Calculate variablilty of kth mode, and truncate
    total_variability = sum(val_dom)
    def k_var(k: int) -> float:
        return sum(val_dom[:k+1])/total_variability

    k_variability = np.array(list(map(k_var, range(len(val_dom)))))
    trunc_at = np.argmax(k_variability > 0.99) # We know that the idx[last] == 1
    print(f"Truncating at {trunc_at+1}th index out of {Rout} indexes. k_variability at idx {trunc_at+1}: {k_var(trunc_at)}")

    return eig_dom[:trunc_at+1]


def gp_regression(train_in: np.ndarray, train_out: np.ndarray):
    kernel = RBF(length_scale=np.ones(train_in.shape[1]))

    # ↓ Could try to scale inputs.
    #Qs = StandardScaler().fit_transform(Q_train)

    gp = GaussianProcessRegressor(
        kernel=kernel,
        # No Alpha, we are certain of ourmeasurement accuracy
        normalize_y=True,
        n_restarts_optimizer=10,
    )

    gp.fit(train_in, train_out)
    return gp

def fit_C(Q_train: np.ndarray, C_train: np.ndarray) -> Sequence[GaussianProcessRegressor]:
    modes = C_train.T # Get columns
    return list(map((lambda ck: gp_regression(Q_train, ck)), modes))

def train_kle(Q_train: np.ndarray, F_train: np.ndarray) -> KLE:
    # Q_train: Uncertain Parameters of Training Set (Independent Var)- Nsamples x Rsd
    # F_train: Traingin Set output (Dependent Var) - Nsamples x Rout

    # Nsamples: # of Samples
    # Rsd: Stochastic Dimension of Uncertain Variable q
    # Rout: Dimension of model output

    # 0. Arg Checks
    if Q_train.shape[0] != F_train.shape[0]:
        raise Exception("Training set inputs and outputs need the same number of samples")
    (Nsamples, _) = F_train.shape

    # 1. Calculate Coefficients Exactly
    mean_field = np.mean(F_train, axis=0)                # Vector: Rout tall
    f_fluct = F_train- mean_field                        # Matrix: Nsamples x Rout
    cov_fluct = (1/Nsamples) * (f_fluct.T @ f_fluct)     # Matrix: Rout x Rout
    eigenvalues, eigenvectors = np.linalg.eig(cov_fluct) # λ_full <- Vector: Rout tall
                                                         # V_full <- Matrix: Rout x Rout
    # Ntrunc: Number of Egienmoodes after truncation
    # ↓ {lam: <- Vector: Ntrunc tall,   V: <- Matrix: Ntrunc x Ntrunc}
    eigen_pairs = truncate_eigenmodes(eigenvalues, eigenvectors)
    V = np.array(list(map(lambda pair: pair["v_i"], eigen_pairs))).T
    lam = list(map(lambda pair: pair["λ_i"], eigen_pairs))
    C_train = f_fluct @ V / np.sqrt(lam)          # C_train: <- Matrix: Nsamples x Ntrunc

    # 2. Train GP On Coefficients
    gps = fit_C(Q_train, C_train)       # gps: <- Vector: Ntrunc tall

    KLE = {'Fbar'      : mean_field,
           'lam_trunc' : lam,
           'V_trunc'   : V,
           'gps'       : gps}
    return KLE


def predict_kle(kle: KLE, Q_novel: np.ndarray) -> np.ndarray:
    # kle: Dictionary containing KLE attributes (See def of KLE)
    # Q_novel: Novel uncertain parameters to estimate f(Q_novel) for - Nnew x Rsd
    C_mu = (seq(kle['gps'])
                     .map(lambda gp : gp.predict(Q_novel, return_std=False))
                     .to_list())
    C_hat = np.column_stack(C_mu)  # Matrix: Nnovel, Ntrunc

    F_tild = kle['Fbar']+ (C_hat* np.sqrt(kle['lam_trunc'])) @ kle['V_trunc'].T
    return F_tild


def RMSE(actual: np.ndarray, predicted: np.ndarray) -> float:
    if actual.shape[0] != predicted.shape[0]:
        raise Exception(" Data set need the same number of samples")
    N = float(actual.shape[0])
    return np.sqrt(1.0/N * np.sum((predicted - actual)**2))

def mean(u: np.ndarray) -> float:
    N = float(u.shape[0])
    return 1.0/N * np.sum(u)

def NRMSE(actual: np.ndarray, predicted: np.ndarray) -> float:
    return RMSE(actual, predicted)/mean(actual)

# -----------------------------------------------------------------------

def test_kle(Rsd: int, Rout: int, Nsamples: int) -> ():
    print(f"  -> testing KLE: stochastic dimension: {Rsd}, output dimension {Rout}, Nsamples: {Nsamples}")
    rng = np.random.default_rng(0)
    q_train = rng.uniform(-2.0, 2.0, size=(Nsamples, Rsd))
    f_train = rng.uniform(-5.0, 5.0, size=(Nsamples, Rout))
    print("f_train:")
    pprint(f_train)

    print("\nKLE Training")
    kle = train_kle(q_train, f_train)

    print("\nKLE In-Sample Prediction")
    aprx_insample = predict_kle(kle, q_train)
    pprint(aprx_insample)
    print(RMSE(f_train, aprx_insample))
    print(NRMSE(f_train, aprx_insample))

    print("\nKLE Out-Sample Prediction")
    q_novel = rng.uniform(-2.0, 2.0, size=(Nsamples, Rsd))
    aprx_outsample= predict_kle(kle, q_novel)
    pprint(aprx_outsample)


if __name__ == "__main__":
    print("\n\n======= KLE =======")
    print("Test Common Case")
    test_kle(Rsd=5, Rout=3, Nsamples=10)

    #print("Test d = 1")
    #test_kle(d=1, samples=10)
