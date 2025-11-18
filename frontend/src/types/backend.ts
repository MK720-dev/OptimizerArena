export interface TrainData {
    X_train: number[][];
    y_train: number[][];
    shape: [number, number];
}

export interface ValData {
    X_val: number[][];
    y_val: number[][];
    shape: [number, number];
}

export interface FoldResults {
    K: number;
    train_data: Record<string, TrainData>[];
    val_data: Record<string, ValData>[];
    train_losses: number[];
    val_losses: number[];
    loss_histories: number[][];
    normalized_loss_histories: number[][];
    weight_histories: number[][][];
    avg_train_loss: number | null;
    avg_val_loss: number | null;
    std_val_loss: number | null;
    metric_name: string | null;
    train_metrics: number[];
    val_metrics: number[];
    avg_train_metric: number | null;
    avg_val_metric: number | null;
}

export interface PCAProjection {
    Z_full: number[][];
    Z_viz: number[][];
    error: number;
    loss?: number[];
    normalized_loss?: number[];
    reconstructed_loss?: number[];
    normalized_reconstructed_loss?: number[];
}

// Result returned by a pure PCA service API call 
export interface PCAResult {
    explained_variance: number[];
    projections: Record<string, PCAProjection>; // key: optimizer,  value: trajectory info
}

export interface LossSurface {
    A: number[][];
    B: number[][];
    L: number[][];
}

// Result returned by a call to the training service contains loss 
// surface info from loss_surface service on top of the PCA results
export interface PCALossResult {
    explained_variance: number[];
    projections: Record<string, PCAProjection>; // key: Fold Index, value: trajectory info
    loss_surface: Record<string,LossSurface>;                  // Loss surface info containes both normalized and original surface 
}

export interface TrainRequest {
    dataset_name: string;
    task_type: "regression" | "binary_classification" | "multiclass_classification";
    optimizer_name: string;
    K?: number;
    epochs?: number;
    batch_size?: number | null;
    lr?: number;
    hidden_layers?: number[] | null;
    func_variant?: string;
    tol?: number;
    max_iter?: number;
    beta1?: number;
    beta2?: number;
    epsilon?: number;
    seed?: number | null;
    compute_pca?: boolean | null;
    k_pca?: number | null;
}

export interface TrainResponse {
    optimizer: string;
    dataset: string;
    task_type: string;
    fold_results: FoldResults;
    pca?: Record<string, PCALossResult>
}

export interface PCARequest {
    snapshots: Record<string, number[][]>;
    k_pca?: number | null;
}

export interface PCAResponse {
    explained_variance: number[];
    projections: Record<string, PCAProjection>;
}

