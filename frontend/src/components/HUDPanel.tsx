import React from "react";
import type { TrainResponse } from "../types/backend"

interface HUDPanelProps{
    trainingResult: TrainResponse;
    selectedFold: number;
    isNormalized: boolean;
    showOriginal: boolean;
    showReconstructed: boolean;
    trainingParams?: {
        epochs?: number;
        batch_size?: number | null;
        lr?: number;
    };
}
const HUDPanel: React.FC<HUDPanelProps> = ({
    trainingResult,
    selectedFold,
    isNormalized,
    showOriginal,
    showReconstructed,
    trainingParams
}) => {
    if (!trainingResult) {
        return null;
    }

    const { optimizer, dataset, task_type, fold_results, pca } = trainingResult;

    // Get PCA metrics for selected fold
    const pcaData = pca?.[`fold_${selectedFold}`];
    const projections = pcaData?.projections;
    const explainedVariance: number[] = pcaData?.explained_variance || [];
    const pc1Variance: number = explainedVariance[0];
    const pc2Variance: number = explainedVariance[1];
    const reconstructionError: number | undefined = projections ? projections[optimizer].error : undefined;
    const totalVariance2D = pc1Variance + pc2Variance;
    const numComponents = explainedVariance.length;

    // Format batch size display
    const batchSizeDisplay = trainingParams?.batch_size ?
        trainingParams?.batch_size : "Full Batch";

    return (
        <div className="bg-gradient-to-br from-slate-50 to-slate-100 border border-slate-300 rounded-lg shadow-md p-4 mb-4">
            <h3 className="text-lg font-bold text-slate-800 mb-3 border-b border-slate-300 pb-2">
                Training & Analysis Summary
            </h3>

            <div className="grid grid-cols-2 gap-4">
                {/* Training Configuration */}
                <div className="bg-white rounded p-3 shadow-sm">
                    <h4 className="font-semibold text-sm text-blue-700 mb-2">Configuration</h4>
                    <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                            <span className="text-slate-600">Optimizer: </span>
                            <span className="font-medium text-slate-900">{optimizer}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-600">Dataset: </span>
                            <span className="font-medium text-slate-900">{dataset}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-600">Task: </span>
                            <span className="font-medium text-slate-900">
                                {task_type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                            </span>
                        </div>
                        {trainingParams?.epochs && (
                            <div className="flex justify-between">
                                <span className="text-slate-600">Epochs: </span>
                                <span className="font-medium text-slate-900">{trainingParams.epochs}</span>
                            </div>
                        )}
                        <div className="flex justify-between">
                            <span className="text-slate-600">Batch Size: </span>
                            <span className="font-medium text-slate-900">{batchSizeDisplay}</span>
                        </div>
                        {trainingParams?.lr && (
                            <div className="flex justify-between">
                                <span className="text-slate-600">Learning Rate: </span>
                                <span className="font-medium text-slate-900">{trainingParams.lr.toExponential(2)}</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* Cross-Validation Results */}
                <div className="bg-white rounded p-3 shadow-sm">
                    <h4 className="font-semibold text-sm text-green-700 mb-2">
                        {fold_results.K}-Fold CV Results
                    </h4>
                    <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                            <span className="text-slate-600">Avg Train Loss: </span>
                            <span className="font-medium text-slate-900">
                                {fold_results.avg_train_loss?.toFixed(4)}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-600">Avg Val Loss: </span>
                            <span className="font-medium text-slate-900">
                                {fold_results.avg_val_loss?.toFixed(4)} ± {fold_results.std_val_loss?.toFixed(4)}
                            </span>
                        </div>
                        {fold_results.metric_name && fold_results.avg_train_metric !== undefined && (
                            <>
                                <div className="flex justify-between">
                                    <span className="text-slate-600">Avg Train {fold_results.metric_name}: </span>
                                    <span className="font-medium text-slate-900">
                                        {fold_results.metric_name === 'accuracy'
                                            ? `${(fold_results.avg_train_metric ? (fold_results.avg_train_metric * 100).toFixed(2) : 'N/A')}%`
                                            : fold_results.avg_train_metric?.toFixed(4)
                                        }
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-600">Avg Val {fold_results.metric_name}: </span>
                                    <span className="font-medium text-slate-900">
                                        {fold_results.metric_name === 'accuracy'
                                            ? `${(fold_results.avg_val_metric! * 100).toFixed(2)}%`
                                            : fold_results.avg_val_metric!.toFixed(4)
                                        }
                                    </span>
                                </div>
                            </>
                        )}
                    </div>
                </div>

                {/* PCA Analysis */}
                <div className="bg-white rounded p-3 shadow-sm">
                    <h4 className="font-semibold text-sm text-purple-700 mb-2">
                        PCA Analysis (Fold {selectedFold})
                    </h4>
                    <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                            <span className="text-slate-600">Components (k): </span>
                            <span className="font-medium text-slate-900">{numComponents}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-600">PC1 Variance: </span>
                            <span className="font-medium text-slate-900">
                                {(pc1Variance * 100).toFixed(2)}%
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-600">PC2 Variance: </span>
                            <span className="font-medium text-slate-900">
                                {(pc2Variance * 100).toFixed(2)}%
                            </span>
                        </div>
                        <div className="flex justify-between border-t border-slate-200 pt-1 mt-1">
                            <span className="text-slate-600 font-semibold">2D Variance: </span>
                            <span className={`font-bold ${totalVariance2D > 0.8 ? 'text-green-600' :
                                    totalVariance2D > 0.6 ? 'text-yellow-600' :
                                        'text-red-600'
                                }`}>
                                {(totalVariance2D * 100).toFixed(2)}%
                            </span>
                        </div>
                        <div className="flex justify-between border-t border-slate-200 pt-1 mt-1">
                            <span className="text-slate-600 font-semibold">Reconstruction Error: </span>
                            <span className="font-medium text-slate-900">
                                {reconstructionError?.toFixed(15)}
                            </span>
                        </div>
                        <div className="text-xs text-slate-500 italic mt-2">
                            {totalVariance2D > 0.8
                                ? "Excellent 2D representation"
                                : totalVariance2D > 0.6
                                    ? "Good 2D representation"
                                    : "Limited 2D representation"
                            }
                        </div>
                    </div>
                </div>

                {/* Current View Settings */}
                <div className="bg-white rounded p-3 shadow-sm">
                    <h4 className="font-semibold text-sm text-orange-700 mb-2">Current View </h4>
                    <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                            <span className="text-slate-600">Display Mode: </span>
                            <span className={`font-medium px-2 py-0.5 rounded ${isNormalized
                                    ? 'bg-blue-100 text-blue-800'
                                    : 'bg-gray-100 text-gray-800'
                                }`}>
                                {isNormalized ? 'Normalized' : 'Original'}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-600">Original Trajectory: </span>
                            <span className={`font-medium ${showOriginal ? 'text-green-600' : 'text-gray-400'}`}>
                                {showOriginal ? '✓ Visible' : '✗ Hidden'}
                            </span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-slate-600">Reconstructed: </span>
                            <span className={`font-medium ${showReconstructed ? 'text-green-600' : 'text-gray-400'}`}>
                                {showReconstructed ? '✓ Visible' : '✗ Hidden'}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            {/* Reconstruction Quality Warning */}
            {totalVariance2D < 0.7 && (
                <div className="mt-3 bg-yellow-50 border border-yellow-300 rounded p-2 text-xs text-yellow-800">
                    <strong>⚠ Note:</strong> The first 2 PCs explain only {(totalVariance2D * 100).toFixed(1)}% of variance.
                    The 2D visualization may not fully capture the optimization dynamics.
                </div>
            )}
        </div>
    );
};

export default HUDPanel;