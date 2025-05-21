import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

index_ref = torch.FloatTensor([1 / 9, 1 / 7, 1 / 5, 1 / 3, 1, 3, 5, 7, 9])


def create_parser():
    parser = argparse.ArgumentParser(description='Generate RI matrix')
    parser.add_argument('min_size', type=int, help='Minimal size of the matrix')
    parser.add_argument('max_size', type=int, help='Maximal size of the matrix')
    parser.add_argument("nb_experiments", type=int, help="Number of experiments")
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file name')
    return parser


def create_saaty_matrix(N: int, nb_mat: int) -> torch.Tensor:
    """Create a list of nb_mat random Saaty matrix of size N x N"""
    center_idx = len(index_ref) // 2  # Index de la valeur 1
    triu_idx = torch.randint(0, len(index_ref) - 1, (nb_mat, N, N))
    triu_idx = triu_idx.triu(diagonal=1)

    # Génération symétrique par inversion des indices
    tril_idx = (len(index_ref) - 1 - triu_idx.transpose(1, 2)).tril(diagonal=-1)

    # Diagonale avec index correspondant à 1
    diag_idx = torch.eye(N, dtype=torch.int).repeat(nb_mat, 1, 1) * center_idx

    # Composition finale
    encoded = triu_idx + tril_idx + diag_idx
    return index_ref[encoded]


def get_list_CI(saaty: torch.Tensor) -> torch.Tensor:
    """Get the list of CI from a list of Saaty matrix"""
    N = saaty.size(1)
    eigvals = torch.linalg.eigvals(saaty)
    eigvals_abs = eigvals.abs()
    eigvals_argmax = torch.argmax(eigvals_abs, dim=1)
    eig_max = eigvals.real[torch.arange(eigvals_argmax.size(0)), eigvals_argmax]
    return (eig_max - N) / (N - 1)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    assert args.nb_experiments > 0
    assert args.min_size > 2
    assert args.max_size >= args.min_size

    df = pd.DataFrame({
        "size": pd.Series(dtype="int"),
        "avg_CI": pd.Series(dtype="float"),
        "std_CI": pd.Series(dtype="float"),
        "nb_exp": pd.Series(dtype="int"),
        "confidence_interval": pd.Series(dtype="float")
    })
    df.set_index("size", inplace=True)
    if args.output is not None and os.path.exists(args.output):
        df = pd.read_csv(args.output, index_col="size")

    for size in range(args.min_size, args.max_size + 1):
        t0 = time.time()
        saaty = create_saaty_matrix(size, args.nb_experiments)
        ci = get_list_CI(saaty)
        avg_ci = torch.mean(ci).item()
        std_ci = torch.std(ci).item()
        confidence_interval = 3 * std_ci / np.sqrt(args.nb_experiments)
        t1 = time.time()
        print(
            f"size : {size},\tavg_ci: {avg_ci:.4f},\tconfidence_interval: {confidence_interval:.4f},\ttime required: {t1 - t0:.2f}")

        if size not in df.index or df.loc[size, "nb_exp"] < args.nb_experiments:
            df.loc[size] = [avg_ci, std_ci, args.nb_experiments, confidence_interval]

    if args.output is not None:
        df.to_csv(args.output)
    else:
        print(df)
