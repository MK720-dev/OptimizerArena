from __future__ import annotations
from tkinter import HIDDEN 
import numpy as np
from typing import List, Optional, Tuple, Callable
from numpy.random import normal
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# === Import your existing code ===
from neural_network import NeuralNetwork
from optimizers import OptimizerFactory, LineSearch
from app.services.pca_analysis import compute_pca_from_snapshots, prepare_trajectory_for_plotting
from app.utils.json_utils import to_jsonable
from app.services.loss_surface import generate_loss_surface

def _compute_loss(model: NeuralNetwork, X: np.ndarray, y:np.ndarray) -> float:
    # Prefer model.compute_loss(y_pred, y) if present 
    if hasattr(model, "predict"):
        y_pred = model.predict(X)
    elif hasattr(model, "forward"):
        y_pred = model.forward(X)
    else:
        raise RuntimeError("Model has neither predict nor forwad attributes.")

    if hasattr(model, "compute_loss"):
        loss = float(model.compute_loss(y_pred, y))
    else:
        loss = float(np.mean((y_pred-y)**2))

    return loss

def regression_function_simple(X: np.ndarray) -> np.ndarray:
    """
    Simple linear regression: y = 3x1 + 2x2 + noise
    """
    noise = np.random.normal(0, 0.1, X.shape[1])
    y = 3 * X[0] + 2 * X[1] + noise
    return y.reshape(1, -1)

def regression_function_medium(X: np.ndarray) -> np.ndarray:
    """
    Nonlinear regression: y = x1^2 + sin(x2) + noise
    """
    noise = np.random.normal(0, 0.1, X.shape[1])
    y = X[0] ** 2 + np.sin(X[1]) + noise
    return y.reshape(1, -1)

def regression_function_complex(X: np.ndarray) -> np.ndarray:
    """
    Highly nonlinear regression: y = sin(x1 * x2) + 0.5*x1^3 - x2^2 + noise
    """
    noise = np.random.normal(0, 0.1, X.shape[1])
    y = np.sin(X[0] * X[1]) + 0.5 * (X[0] ** 3) - (X[1] ** 2) + noise
    return y.reshape(1, -1)

# ---------- Public API for the service ----------
def get_optimizer_factory(name: str, **kwargs):
    factory = OptimizerFactory()
    name = name.lower()

    if name in ["sgd", "fullbatch_gradientdescent", "minibatch_gradientdescent"]:
        optimizer = factory.create("gradientdescent", **kwargs)

    elif name == "adam":
        optimizer = factory.create("adam", **kwargs)

    elif name == "rmsprop":
        optimizer = factory.create("rmsprop", **kwargs)

    elif name == "bfgs":
        optimizer = factory.create("bfgs", **kwargs)

    else:
        raise ValueError(f"Unknown optimizer name: {name}")
    # Let factory decide which class to instantiate
    return optimizer


def get_dataset(dataset_name: str, task_type: str, seed: int = 42, func_variant: str="simple"):
    """
    Loads or generates a dataset based on its name and task type.

    Returns:
        X (np.ndarray): Feature matrix of shape (n_features, n_samples)
        y (np.ndarray): Target labels or values
        meta (dict): Metadata about dataset
            {
              "input_dim": int,
              "output_dim": int,
              "task_type": str,
              "name": str
            }

    Important: func_variant is only used for synthetic regression tasks
    """

    rng =  np.random.default_rng(seed)

    # ------------- Synthetic dataset (default) ---------------
    if task_type == "regression":
        if dataset_name.lower() == "synthetic":
            X = rng.normal(size=(2,400))
            if func_variant == "simple":
                y = regression_function_simple(X)
            elif func_variant == "medium":
                y = regression_function_medium(X)
            elif func_variant == "complex":
                y = regression_function_complex(X)
            else: 
                raise ValueError("This is not a valid variant for the regression task.")
        elif dataset_name.lower() == "california_housing":
            print("loading california_housing data")
            data = datasets.fetch_california_housing()
            X = data.data.T
            y = data.target.reshape(1, -1)
        else:
            raise ValueError("This regression dataset is not available at the moment.")
        meta = {
                "input_dim": X.shape[0],
                "output_dim": 1,
                "task_type": "regression",
                "name": dataset_name
            }
        return X, y, meta

    elif task_type == "binary_classification":
        if dataset_name == "breast_cancer":
            data = datasets.load_breast_cancer()
            X = data.data.T
            y = data.target.reshape(1, -1)
        else:
            raise ValueError("This binary classification dataset is not available at the moment.")
        meta = {
                "input_dim": X.shape[0],
                "output_dim": 1,
                "task_type": "binary_classification",
                "name": dataset_name
            }
        return X, y, meta
    elif task_type == "multiclass_classification":
        if dataset_name.lower() == "iris":
            data = datasets.load_iris()
            X = data.data.T
            y = data.target.reshape(1, -1)
        else: 
            raise ValueError("This multi-class classification is dataset not available at the moment.")
        meta = {
            "input_dim": X.shape[0],
            "output_dim": len(np.unique(y)),
            "task_type": "multiclass_classification",
            "name": dataset_name
        }
        return X, y, meta
    else: 
        raise ValueError("Task type not defined.")

def build_network_architecture(
    input_dim: int,
    output_dim: int,
    hidden_layers: list[int] | None=None,
    activation: str = "relu",
    seed: int | None=None
):
    """
    Dynamically build a neural network based on task type and dataset dimensions.
    
    input_dim: number of features
    output_dim: number of target classes (only used for multiclass)
    hidden_layers: list of hidden layer sizes (user-defined or default)
    """
    # ---------------- Defaults ------------------
    if hidden_layers is None:
        hidden_layers = [8, 8] # good generic start; can be tuned later 

    # --------------- Build the model ---------------
    model = NeuralNetwork(input_size=input_dim, hidden_layers=hidden_layers, output_size=output_dim, activation=activation, seed=seed)

    return model

def run_training(
    dataset_name: str = "synthetic",
    task_type: str = "regression",
    optimizer_name: str = "adam",
    K: int = 5,
    epochs: int = 10,
    batch_size: int | None = None,
    lr: float = 1e-2,
    line_search: LineSearch = LineSearch(),
    tol: float = 1e-5,
    max_iter: int = 100,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    seed: int | None = 42,
    hidden_layers: list[int] | None = None,
    func_variant: str = "simple",
    compute_pca: Optional[bool] = "True",
    k_pca: Optional[int] = 3,
) -> dict:
    """
    Executes K-fold cross validation training for a given dataset and optimizer.
    Returns a dictionary with all training and validation metrics.
    """

    # Load dataset
    X, y, meta = get_dataset(dataset_name, task_type, seed=seed, func_variant=func_variant)
    input_dim, output_dim = meta["input_dim"], meta["output_dim"]

    print(f"Hidden Layer: {hidden_layers}")

    # Optimizer parameters
    opt_name = optimizer_name.lower()
    if any(key in opt_name.split('_') for key in ["gradientdescent", "sgd", "fullbatch", "minibatch"]):
        optimizer_args = {"lr": lr}
    elif opt_name == "bfgs":
        optimizer_args = {"line_search": line_search, "tol": tol, "max_iter": max_iter}
    elif opt_name == "adam":
        optimizer_args = {"lr": lr, "beta1": beta1, "beta2": beta2, "epsilon": epsilon}
    elif opt_name == "rmsprop":
        optimizer_args = {"lr": lr, "beta": beta1, "epsilon": epsilon}

    # Model + optimizer setup
    model = build_network_architecture(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        seed=seed,
    )
    if y.shape[0] > 1:
        model.set_loss('cross_entropy')
    elif np.all(np.isin(y,[0,1])):
        model.set_loss("binary_cross_entropy")
    else:
        model.set_loss("mse")
    optimizer = get_optimizer_factory(name=optimizer_name, **optimizer_args)
    model.set_optimizer(optimizer)

    # Cross-validation
    fold_results = model.cross_validate(
        X, y,
        k=K,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
        verbose=False,
    )

    # Validate return type
    if not isinstance(fold_results, dict):
        raise TypeError("cross_validate() must return a dictionary of metrics and losses.")

    # Combine with metadata
    result_package = {
        "optimizer": optimizer_name,
        "dataset": dataset_name,
        "task_type": task_type,
        "fold_results": fold_results
    }

    if compute_pca:
        result_package["pca"] = {}
        # Carry on per fold PCA
        for fold_idx, (weight_hist, loss_hist) in  enumerate(zip(fold_results["weight_histories"], fold_results["loss_histories"])):

            # Convert each fold’s weight history to np.ndarray
            weights = np.array(weight_hist)  # shape (T, n_params)
            losses  = np.array(loss_hist)    # shape (T,)

            # Prepare dictionary for this fold
            snapshots = {f"{opt_name}": weights}

            # Compute PCA for this fold only
            pca_model, projections = compute_pca_from_snapshots(snapshots, k=k_pca)

            # Check variance explained
            variance_explained = pca_model.evr.sum()
            print(f"Fold {fold_idx+1}: {variance_explained:.2%} variance explained with k={k_pca}") 

            # Attach loss for color mapping
            projections[f"{opt_name}"]["loss"] = losses

            # Access fold's training data 
            X_train_fold = fold_results["train_data"][fold_idx]["X_train"]
            y_train_fold = fold_results["train_data"][fold_idx]["y_train"]

            # Reconstruct Z projection and re-compute loss
            # The reconstructed loss is not exactly the same as the real loss at these projected trajectory points 
            # Indeed, picking k less than the number of original features makes us lose some of the information about the loss landscape
            # So the following are the reconstructed version of our original weights learned by the optimizer during training
            reconstructed_original_weigths_flat = pca_model.inverse_transform(projections[f"{opt_name}"]["Z_full"])

            # Array to store losses for the reconstructed learned weights
            reconstructed_losses = []
            # Evaluate loss at each reconstructed weight 
            for w_flat in reconstructed_original_weigths_flat:
                model.unpack_weights(w_flat)
                y_pred_fold, _ = model.forward(X_train_fold) # Froward pass to get 
                loss = model.compute_loss(y_pred_fold, y_train_fold)
                reconstructed_losses.append(loss)

            projections[f"{opt_name}"]["reconstructed_loss"] = np.array(reconstructed_losses)

            # Generate loss surface data 
            loss_surface_data = generate_loss_surface(
                pca_model=pca_model,
                model=model,
                X_train=X_train_fold,
                y_train=y_train_fold, 
                alpha_range=(-5,5),
                beta_range=(-5,5),
                grid_size=100,
                use_components=2
            )

            # ===== CONSISTENT NORMALIZATION USING SURFACE MIN/MAX =====

            # Extract global min/max from the original surface
            surface_L = np.array(loss_surface_data["original_loss"]["L"])
            surface_loss_min = surface_L.min()
            surface_loss_max = surface_L.max()

            # Helper function for consistent normalization
            def normalize_with_surface_reference(values):
                """Normalize using the loss surface's min/max as reference."""
                if surface_loss_max > surface_loss_min:
                    return (values - surface_loss_min) / (surface_loss_max - surface_loss_min)
                else:
                    return np.zeros_like(values)

            # Normalize original trajectory losses
            projections[f"{opt_name}"]["normalized_loss"] = normalize_with_surface_reference(
                projections[f"{opt_name}"]["loss"]
            )
    
            # Normalize reconstructed trajectory losses
            projections[f"{opt_name}"]["normalized_reconstructed_loss"] = normalize_with_surface_reference(
                projections[f"{opt_name}"]["reconstructed_loss"]
            )
    
            # Normalize the surface
            loss_surface_data["normalized_loss"] = {
                "A": loss_surface_data["original_loss"]["A"],
                "B": loss_surface_data["original_loss"]["B"],
                "L": normalize_with_surface_reference(surface_L).tolist()
            }
    
            # ===== END NORMALIZATION =====

            # Store PCA results under fold key
            result_package["pca"][f"fold_{fold_idx+1}"] = {
                "explained_variance": pca_model.evr.tolist(),
                "projections": {
                    k: {
                        "Z_full": v["Z_full"].tolist(),
                        "Z_viz": v["Z_viz"].tolist(),
                        "error": v["error"],
                        "loss": v["loss"].tolist(),
                        "normalized_loss": v["normalized_loss"].tolist() ,
                        "reconstructed_loss": v["reconstructed_loss"].tolist(),
                        "normalized_reconstructed_loss": v["normalized_reconstructed_loss"].tolist(),

                    } for k, v in projections.items()
                },
                "loss_surface": loss_surface_data
            }

            print(f"Optimizer: {result_package['optimizer']}")
            print(f"Task type: {result_package['task_type']}")
            print(f"Dataset: {result_package['dataset']}")
            print(f"Error: {result_package['pca'][f'fold_{fold_idx+1}']['projections'][f'{optimizer_name}']['error']}")
    return to_jsonable(result_package)



    


        







