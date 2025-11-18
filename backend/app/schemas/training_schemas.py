from pydantic import BaseModel
from typing import List, Literal, Optional, Any, Dict

class TrainRequest(BaseModel):
    # Core parameters
    dataset_name: str = "synthetic"
    task_type: Literal["regression", "binary_classification", "multiclass_classification"] = "regression"
    optimizer_name: Literal["sgd", "adam", "rmsprop", "bfgs", "fullbatch_gradientdescent", "minibatch_gradientdescent"] = "adam"
    
    # Training parameters
    K: int = 5
    epochs: int = 10
    batch_size: Optional[int] = None
    lr: float = 1e-2
    hidden_layers: Optional[List[int]] = None
    func_variant: Literal["simple", "medium", "complex"] = "simple"

    # Optimizer hyperparameters
    line_search: Optional[Dict] = None  # handled internally as LineSearch() if None
    tol: float = 1e-5
    max_iter: int = 100
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    # Random seed
    seed: Optional[int] = 42

    # PCA parameters
    compute_pca: Optional[bool] = True
    k_pca: Optional[int] = 2

class TrainResponse(BaseModel):
    optimizer: str
    dataset: str
    task_type: str
    fold_results: Dict[str, Any]
    pca: Optional[Dict[str,Dict[str, Any]]] = None

