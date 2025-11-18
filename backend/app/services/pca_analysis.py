from dataclasses import dataclass
from typing import Dict, Tuple 
from app.utils.json_utils import to_jsonable
import numpy as np


@dataclass
class PCAModel:
    """
    PCA model for neural network weight trajectories.
    Attributes:
        mean: Mean weight vector across all snapshots. Shape (n,)
        components: Principal component directions (columns). Shape (n, k)
                   These form an orthonormal basis in weight space.
        evr: Explained variance ratio for each component. Shape (k,)
             Sum of evr[i] gives total variance explained.
        singular_values: Singular values from SVD. Shape (k,)
    
    """
    mean: np.ndarray
    components: np.ndarray
    evr: np.ndarray
    singular_values: np.ndarray

    def transform(self, weights: np.ndarray) -> np.ndarray:
        """Project weights into PC space."""
        # Ensure z is 2D for consistent matrix operations
        weights = np.atleast_2d(weights) # Single: (3,) → (1, 3) | Batch: (N, 3) → (N, 3) unchanged
        centered = weights - self.mean
        # Remove extra dimension for single sample
        return centered @ self.components

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        """Reconstruct weights from PC coordinates"""
        z = np.atleast_2d(z)
        reconstructed = z @ self.components.T + self.mean
        return reconstructed.squeeze() if z.shape[0] == 1 else reconstructed

    def reconstruction_error(self, weights: np.ndarray) -> float:
        """Compute relative reconstruction error."""
        z = self.transform(weights)
        reconstructed = self.inverse_transform(z)
        error = np.linalg.norm(weights - reconstructed)
        original_norm = np.linalg.norm(weights - self.mean)
        return error / original_norm if original_norm > 0 else 0.0

def compute_pca_from_snapshots(
    snapshots: Dict[str, np.ndarray],
    k: int = 2
) -> Tuple[PCAModel, Dict[str, Dict[str,np.ndarray]]]:
    """
    Compute PCA on neural network weight trajectories.
    
    Args:
        snapshots: Dictionary mapping optimizer names to lists of weight vectors.
                  Each weight vector should be a flat 1D array of shape (n,).
                  w0 represents vector of all flattened weights and biases at t=0 (initial params)
                  w1 represents vector of all flattened weights and biases at t=1 (1st epoch)
                  Example: {"adam": [w0, w1, ..., wT], "sgd": [w0, w1, ..., wT]}
        k: Number of principal components to retain (must be >= 1)
    
    Returns:
        pca_model: PCAModel containing the fitted PCA transformation
        projections: Dictionary mapping optimizer names to their projections.
                    Each projection is a dict with key "Z" containing the
                    PC coordinates of shape (T, k) where T is trajectory length.
    
    Raises:
        ValueError: If snapshots is empty, k is invalid, or data has issues
        TypeError: If weight arrays are not numpy arrays
        RuntimeError: If SVD fails
    
    Example:
        >>> snapshots = {
        ...     "adam": [np.random.randn(100) for _ in range(50)],
        ...     "sgd": [np.random.randn(100) for _ in range(50)]
        ... }
        >>> pca_model, projections = compute_pca_from_snapshots(snapshots, k=3)
        >>> print(f"Variance explained: {pca_model.evr.sum():.2%}")
    """
    # Validation 
    if not snapshots:
        raise ValueError("Snapshots dictionary is empty.")
    if k < 1:
        raise ValueError(f"k must be >=1, got {k}")
    
    index_map = [] # Track which optimizer and timestep each row belongs to 

    for opt_name, trajectory in snapshots.items():
        if not snapshots:
            raise ValueError(f"Optimizer '{opt_name}' has empty trajectory.")

        for timestep, weights in enumerate(trajectory):
            if not isinstance(weights, np.ndarray):
                raise TypeError(
                    f"Expected numpy array for {opt_name} at timestep {timestep}, "
                    f"got {type(weights)}"
                    )
 
                index_map.append((opt_name, timestep))
    
    # Create data matrix 
    M = np.stack([weights for weights in snapshots.values()], axis=0).squeeze() # Shape of M is (N, n) where N = numbers of snapshots (# epochs * # optimizers) and n = params
    n_samples, n_features = M.shape

    # Check k validity
    max_k = min(n_samples, n_features)
    if k > max_k:
        raise ValueError(
            f"k={k} is too large. Maximum possible is {max_k} "
            f"for data shape {M.shape}"
        )

    # Center the data for PCA
    mean = M.mean(axis=0) # (n,)
    X = M - mean # (N, n)

    # Check for zero variance
    if np.allclose(X,0):
        raise ValueError("All weight snapshots are identical (zero variance)")

    # Perform SVD on X
    # S is an array containing the singular values of X
    # The rows of Vh are the eigenvectors of (X.T)(X)
    # The corresponding eigenvalues are given by s**2 for s in S 
    try: 
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(f"SVD decomposition failed: {e}")

    # Numerical stability check 
    if np.any(np.isnan(S)) or np.any(np.isinf(S)):
        raise RuntimeError("SVD produced NaN or Inf singular values")

    # Extract k components 
    V = Vt.T[:, :k] # (n, k) - principal component directions
    Z = X @ V # (N, k) - projected coordinates

    # Compute explained variance ratio
    total_variance = (S**2).sum()
    evr = (S[:k]**2) / total_variance

    # Create PCA model
    pca_model = PCAModel(
        mean=mean, 
        components=V, 
        evr=evr, 
        singular_values=S[:k]
    )

    # Compute reconstruction error for the model
    error = pca_model.reconstruction_error(M)

    # Split projections back per optimizer 
    projections = {}
    ptr = 0

    for opt_name, trajectory in snapshots.items():
        T = len(trajectory)
        Z_opt_full = Z[ptr:ptr+T, :] # (T, k)
        Z_opt_2d = Z[ptr:ptr+T, :2]
        projections[opt_name] = {
            "Z_full": Z_opt_full,
            "Z_viz": Z_opt_2d,
            "error": error
        }
        ptr += T
        
    return pca_model, projections

def prepare_trajectory_for_plotting(
    projections: Dict[str, Dict[str, np.ndarray]],
    pc_indices: Tuple[int,...] = (0,1)
    ) -> Dict[str, np.ndarray]:
    """
    Extract specific PC coordinates for 3D plotting.
    
    Args:
        projections: Output from compute_pca_from_snapshots
        pc_indices: Which PCs to extract (default: first 2)
    
    Returns:
        Dictionary mapping optimizer names to (T, 2) arrays
    """
    plot_data = {}
    for opt_name, data in enumerate(projections):
        if "Z" not in data:
            raise KeyError(f"Missing 'Z' key in projections for optimizer {opt_name}")
        Z = data["Z"] # (T, k)
        k = Z.shape[1]
        # Safely handle if fewer PCs than requested
        valid_indices = [i for i in pc_indices if i < k]
        # Extract PCs
        selected_pcs = Z[:, list(valid_indices)] # (T,len(valid_indices)) -> (T,2) for PCA with k=2
        plot_data[opt_name] = selected_pcs 
    return to_jsonable(plot_data)


    

