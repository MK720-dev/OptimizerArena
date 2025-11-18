import numpy as np
from typing import Dict, Tuple
from app.services.pca_analysis import PCAModel
from app. utils.json_utils import to_jsonable
from neural_network import NeuralNetwork

def generate_loss_surface(
    pca_model: PCAModel,
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha_range: Tuple[float, float] = (-3.0, 3.0),
    beta_range: Tuple[float, float] = (-3.0, 3.0),
    grid_size: int = 25,
    use_components: int =2,
    ) -> Dict[str, list]:

    """
    Generate a 3D loss surface L(alpha, beta) in PCA-defined subspace.

    Args:
        pca_model: Trained PCAModel (with mean and components)
        model: Neural network instance (must have unpack_weights() and compute_loss())
        X_val, y_val: Validation data for evaluating loss
        alpha_range, beta_range: Ranges for 2D grid exploration in PCA coordinates
        grid_size: Number of steps per axis (grid_size x grid_size points)
        normalize_loss: Whether to normalize loss values (optional)

    Returns:
        Dictionary with JSON-ready lists:
        {
            "A": 2D list of alpha grid values,
            "B": 2D list of beta grid values,
            "L": 2D list of loss values,
        }
    """

    # ---- Create 2D PCA grid ----
    alpha_vals = np.linspace(alpha_range[0], alpha_range[1], grid_size)
    beta_vals = np.linspace(beta_range[0], beta_range[1], grid_size)
    A, B = np.meshgrid(alpha_vals, beta_vals)

    grid_points = np.stack([A.ravel(), B.ravel()], axis=1) # shape (N,2)

     # Only use first 'use_components' dimensions
    if use_components == 2:
        grid_points = np.stack([A.ravel(), B.ravel()], axis=1)  # (N, 2)
    else:
        raise ValueError("Currently only 2D surfaces are supported for visualization")

    # Create a restricted PCA model using only first 2 components
    pca_2d = PCAModel(
        mean = pca_model.mean,
        components= pca_model.components[:,:use_components],
        evr = pca_model.evr[:use_components],
        singular_values=pca_model.singular_values[:use_components]
    )

    # Map back to original weight space using ONLY first 2 components
    reconstructed_weights = pca_2d.inverse_transform(grid_points)

    # ---- Evaluate loss at each reconstructed weight ----
    losses = []
    for w_flat in reconstructed_weights:
        model.unpack_weights(w_flat)
        y_pred, _ = model.forward(X_train)
        #print(f"y_pred.shape: {y_pred.shape}", flush=True)
        #print(f"y_train.shape: {y_train.shape}", flush=True)
        loss = model.compute_loss(y_pred, y_train)
        losses.append(loss)
    print(f"losses.shape (before reshaping) {np.array(losses).shape}", flush=True)
    L = np.array(losses).reshape(A.shape)
    print(f"L.shape (after reshaping) {L.shape}", flush=True)

    loss_surface_data = { 
        "original_loss":{
            "A": A.tolist(),
            "B": B.tolist(),
            "L": L.tolist()
        }
    }
    
    return to_jsonable(loss_surface_data)
