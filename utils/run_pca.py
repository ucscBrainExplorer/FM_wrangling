"""
Run PCA on an embedding matrix (e.g. a UCE embedding) and save the result
to disk as an HDF5 file.

Accepts three kinds of input, auto-detected by file extension:
    .npy    - a plain numpy array saved with np.save (rows = cells,
              cols = features), e.g. mouse_mammary_epi_noPal_uce.npy
    .h5ad   - an AnnData file; the array is read from .X by default,
              or from .obsm[<key>] if --key is given (e.g. --key X_uce)
    .h5     - a generic HDF5 file; the array is read from the dataset
              named by --key (required in this case)

Output is an HDF5 file (default: pca.h5) containing:
    pca                      (n_cells, n_components) PCA-transformed data
                             (key matches FM_wrangling/runIPCA.py's convention)
    explained_variance_ratio (n_components,)
    explained_variance       (n_components,)
    singular_values          (n_components,)
    components               (n_components, n_features) PCA loadings
    mean                     (n_features,) feature means subtracted before PCA
    obs_names                (n_cells,) cell barcodes, if available (h5ad input)

It also writes a pickled sklearn Pipeline (default: pca_model.pkl, alongside
the HDF5 output) for projecting new data into the same PCA space:

    import joblib
    model = joblib.load("pca_model.pkl")
    X_new_pca = model.transform(X_new)   # applies the same scaler (if any) + PCA

The pipeline has a "scaler" step ("passthrough" unless --standardize was used)
and a "pca" step. Disable with --no-model.

Usage:
    python run_pca.py --input mouse_mammary_epi_noPal_uce.npy \
        --output pca.h5 --n-components 50

    python run_pca.py --input mouse_mammary_epi.h5ad --key X_uce \
        --output pca.h5 --n-components 50
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_array(input_path: str, key: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    """Load the embedding matrix (and obs_names, if available) from disk."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".npy":
        return np.load(input_path), None

    if ext == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(input_path)
        obs_names = adata.obs_names.to_numpy()
        if key is None:
            X = adata.X
        else:
            X = adata.obsm[key]
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        return X, obs_names

    if ext in (".h5", ".hdf5"):
        if key is None:
            raise ValueError("--key (dataset name) is required for a generic .h5 input")
        with h5py.File(input_path, "r") as f:
            return np.asarray(f[key]), None

    raise ValueError(f"Unsupported input extension: {ext} (expected .npy, .h5ad, or .h5)")


def run_pca(
    X: np.ndarray,
    n_components: int = 50,
    standardize: bool = False,
    random_state: int = 0,
) -> Pipeline:
    """Fit a (scaler -> PCA) pipeline. The scaler step is "passthrough" unless
    standardize=True, so the returned pipeline can transform new data as-is."""
    model = Pipeline([
        ("scaler", StandardScaler() if standardize else "passthrough"),
        ("pca", PCA(n_components=n_components, random_state=random_state)),
    ])
    model.fit(X)
    model._X_pca = model.transform(X)  # stash for convenience
    return model


def save_pca(model: Pipeline, output_path: str, obs_names: np.ndarray | None = None) -> None:
    pca = model.named_steps["pca"]
    with h5py.File(output_path, "w") as f:
        f.create_dataset("pca", data=model._X_pca)
        f.create_dataset("explained_variance_ratio", data=pca.explained_variance_ratio_)
        f.create_dataset("explained_variance", data=pca.explained_variance_)
        f.create_dataset("singular_values", data=pca.singular_values_)
        f.create_dataset("components", data=pca.components_)
        f.create_dataset("mean", data=pca.mean_)
        if obs_names is not None:
            f.create_dataset(
                "obs_names", data=np.asarray(obs_names, dtype="S")
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to .npy, .h5ad, or .h5 input")
    parser.add_argument(
        "--key",
        default=None,
        help="obsm key (.h5ad) or dataset name (.h5) holding the array; "
        "unused for .npy input",
    )
    parser.add_argument("--output", default="pca.h5", help="Output HDF5 path (default: pca.h5)")
    parser.add_argument(
        "--model-output",
        default=None,
        help="Path for the pickled (scaler -> PCA) sklearn Pipeline for transforming "
        "new data (default: pca_model.pkl next to --output)",
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Skip writing the pickled pipeline",
    )
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Z-score each feature before running PCA",
    )
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()

    X, obs_names = load_array(args.input, args.key)
    print(f"Loaded input array with shape {X.shape} from {args.input}")

    model = run_pca(
        X,
        n_components=args.n_components,
        standardize=args.standardize,
        random_state=args.random_state,
    )
    pca = model.named_steps["pca"]
    print(f"Ran PCA: {args.n_components} components")
    print("Explained variance ratio (first 10):", pca.explained_variance_ratio_[:10])
    print("Total explained variance:", pca.explained_variance_ratio_.sum())

    save_pca(model, args.output, obs_names=obs_names)
    print(f"Saved PCA results to {args.output}")

    if not args.no_model:
        import joblib

        model_path = args.model_output or os.path.join(
            os.path.dirname(args.output) or ".", "pca_model.pkl"
        )
        del model._X_pca  # don't pickle the stashed training projection
        joblib.dump(model, model_path)
        print(f"Saved PCA pipeline (for transforming new data) to {model_path}")


if __name__ == "__main__":
    main()
