from doctest import debug
from http.client import HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.training_schemas import TrainRequest, TrainResponse
from app.schemas.pca_schemas import PCARequest, PCAResponse
from app.services.pca_analysis import PCAModel
from app.services.training import run_training
from app.services.pca_analysis import compute_pca_from_snapshots
from optimizers import LineSearch
from dotenv import load_dotenv
import numpy as np
import os

# Load .env file 
load_dotenv()

DEBUG = os.getenv("DEBUG", "False") == "True"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

app = FastAPI(title="Optimizer Arena Backend", debug=DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is running!"}

@app.post("/train", response_model=TrainResponse)
def train(req: TrainRequest):
    """
    API endpoint to run cross-validated training.
    Expects JSON body from frontend matching TrainRequest schema.
    """
    try:
        # IMPORTANT NOTES ON LINE_SEARCH INITIALIZATION
        # For now line_search will be handled automatically with the default parameters using backtracking + Wolfe conditions
        # However in the fututre it could be defined through a request of its own and then passed to the relevant optimizers 
        line_search = LineSearch() if req.line_search is None else req.line_search

        # Execute the training routine
        results = run_training(
                dataset_name=req.dataset_name,
                task_type=req.task_type,
                optimizer_name=req.optimizer_name,
                K=req.K,
                epochs=req.epochs,
                batch_size=req.batch_size,
                lr=req.lr,
                line_search=line_search,
                tol=req.tol,
                max_iter=req.max_iter,
                beta1=req.beta1,
                beta2=req.beta2,
                epsilon=req.epsilon,
                seed=req.seed,
                hidden_layers=req.hidden_layers,
                func_variant=req.func_variant,
                compute_pca=req.compute_pca,
                k_pca=req.k_pca
        )
        return results 
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/pca", response_model=PCAResponse)
async def run_pca(req: PCARequest):
    pca_model, projections = compute_pca_from_snapshots(req.snapshots, k=req.k_pca)
    return {
        "explained_variance": pca_model.evr.tolist(),
        "projections": {
            k: {
                "Z": v["Z"].tolist(),
                "loss": v.get("loss", []).tolist() if isinstance(v.get("loss"), np.ndarray) else v.get("loss", [])
            }
            for k, v in projections.items()
        } 
    }
