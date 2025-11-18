from pydantic import BaseModel
from typing import Dict, List, Any

class PCARequest(BaseModel):
    snapshots: Dict[str, List[List[float]]]
    k_pca: int = 3

class PCAResponse(BaseModel):
    explained_variance: List[float]
    projections: Dict[str, Any]
