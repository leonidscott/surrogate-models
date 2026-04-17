from pprint import pprint
from typing import Callable, Sequence, TypeAlias, TypedDict
from itertools import product
from math import prod, factorial
from numpy.polynomial import Legendre, HermiteE
import numpy as np

PolyClass: TypeAlias = type[Legendre] | type[HermiteE]

def Nbasis(n: int, p: int) -> int:
    ''' n <- stochastic dimension
        p <- polynomial order '''
    return factorial(n + p)//(factorial(n) * factorial(p))

class PCE_Param(TypedDict):
    coeffs: Sequence[float]
    dist: str
    p: int

Psi_Type = Sequence[Sequence[Callable[[float], float]]]
def psi_factory(n: int, p: int, poly_fn: PolyClass) -> Psi_Type:
    def total_order_index(n: int, p: int):
        return [
            alpha
            for alpha in product(range(p + 1), repeat=n)
            if sum(alpha) <= p
        ]

    def build_poly(poly: PolyClass, p: int) -> Callable[[float], float]:
        coeffs = np.concatenate([np.zeros(p, dtype=int), [1]])
        return poly(coeffs)

    multi_index = total_order_index(n,p)

    return list(map(
        (lambda mi: [build_poly(poly_fn, p) for p in mi]),
        multi_index
    ))

def eval_psi_point(psi: Psi_Type, bidx: int, xi: Sequence[float]) -> float:
    polys = psi[bidx]
    n = len(polys)

    if len(xi) != n:
        raise Exception(f"Stochastic Dimension = {n}, but len(xi_vec) = {len(xi)}")

    return prod(poly(xi[i]) for i, poly in enumerate(polys))

def from_shape_fn(shape, fn):
    return np.fromfunction(np.vectorize(fn), shape, dtype=int)

def build_psi(psi: Psi_Type, xis: Sequence[Sequence[float]]) -> np.ndarray:
    Nsamples = len(xis)
    Nbasis   = len(psi)
    print(f"Nsamples: {Nsamples}, Nbasis: {Nbasis}")
    # from_shape_fn not that effecient. Make improvements eventually
    return from_shape_fn((Nsamples, Nbasis),
                         (lambda row,col : eval_psi_point(psi, col, xis[row])))

def train_pce(xis: Sequence[Sequence[float]], ys: Sequence[float], dist: str, p: int) -> PCE_Param:
    # xis <- Training set inputs: Vector of xis
    # ys  <- Training set outputs: Model evaluations of xis
    # dist <- Distribution of xi vec. 'uniform' | 'normal'
    # p    <- Polynomial order
    if len(xis) == 0: raise Exception("empty xis not allowwed")
    if len(ys) != len(xis):
        raise Exception(f"""Must have the same number of training outputs as inputs.
        len(xis) = {len(xis)}, len(ys) = {len(ys)}""")
    if dist not in {"uniform", "normal"}:
        raise Exception(f"Distribution must be 'normal' or 'uniform'. Recieved '{dist}'")
    if p < 0: raise Exception("p must be greater than zero")

    n = len(xis[0]) # Stochastic Dimension
    poly_fam = Legendre if dist=="uniform" else HermiteE
    psifns = psi_factory(n, p, poly_fam)
    psi = build_psi(psifns, xis)
    pprint(psi.shape)
    coeffs, _, _, _ = np.linalg.lstsq(psi, ys, rcond=None)
    return {'coeffs': coeffs, 'dist': dist, 'p': p}


def predict_pce(pce_params: PCE_Param, novel_xis: Sequence[Sequence[float]]) -> Sequence[float]:
    dist = pce_params['dist']
    if dist not in {"uniform", "normal"}:
        raise Exception(f"Distribution must be 'normal' or 'uniform'. Recieved '{dist}'")

    n = len(novel_xis[0]) # Stochastic Dimension
    poly_fam = Legendre if dist =="uniform" else HermiteE
    psifns = psi_factory(n, pce_params['p'], poly_fam)
    psi_novel = build_psi(psifns, novel_xis)

    return psi_novel @ pce_params['coeffs']


def test_case(samples: int, n: int, p: int):
    print("Nbasis(n,p): ", Nbasis(n,p))
    rng = np.random.default_rng(0)
    xis = rng.uniform(-1.0, 1.0, size=(samples, n))
    ys = rng.uniform(-5.0, 5.0, size=(samples, 1))
    print("xis:")
    pprint(xis)
    print("ys:")
    pprint(ys)

    # --------- Training PCE ----------
    print("trained pce")
    pce_meta = train_pce(xis, ys, "uniform", 3)
    print(pce_meta['coeffs'])

    # --------- Predicting with PCE ----------
    print("testing pce")
    pred_ys = predict_pce(pce_meta, xis)
    print(f"predicted ys: {pred_ys}")

if __name__ == "__main__":
    #print(Nbasis(5,3))
    #psifns = psi_factory(5,3,HermiteE)
    #res = eval_psi_point(psifns, 1, [0.1, 0.5, 0.3, 0.4, 0.5])
    #train_pce([[0.1, 0.5, 0.3, 0.4, 0.5]], [0.1], "normal", 3)
    test_case(samples=5,n=3,p=3)
