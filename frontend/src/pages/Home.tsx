import React, { useState } from "react";
import ControlsPanel from "../components/ControlsPanel";
import HUDPanel from "../components/HUDPanel";
import ConvergencePlot2D from "../components/ConvergencePlot2D";
import Arena3D from "../components/Arena3D";
import { runTraining } from "../api/backend";
import type { TrainRequest, TrainResponse } from "../types/backend";
import "../App.css"

const Home: React.FC = () => {
    const [trainingResult, setTrainingResult] = useState<TrainResponse | null>(null);
    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [trainingParams, setTrainingParams] = useState<TrainRequest | null>(null);

    // View mode state:  normalized vs non-normalized 
    const [isNormalized, setIsNormalized] = useState<boolean>(false);

    // Trajectory display state
    const [showOriginal, setShowOriginal] = useState<boolean>(true);
    const [showReconstructed, setShowReconstructed] = useState<boolean>(true);

    // Fold selection state 
    const [selectedFold, setSelectedFold] = useState<number>(1);

    const handleTrain = async (params: TrainRequest) => {
        setTrainingResult(null);
        setLoading(true);
        setError(null);
        try {
            const result = await runTraining(params);
            setTrainingResult(result);
            setTrainingParams(params);
            setSelectedFold(1);
        } catch (e: any) {
            console.error(e);
            setError(e.message || "Training Failed.");

        } finally {
            setLoading(false);
        }
    };

    return (
        <div>
            <div className="w-full container">
                {/* Left: Controls */}
                <div className="container-item left-item">
                    <ControlsPanel onRun={handleTrain} disabled={loading} />

                    {/* View Toggle Controls */}
                    {trainingResult && (
                        <div className="mt-6 p-4 border-t">
                            <h3 className="font-semibold mb-3">View Controls</h3>

                            {/* Fold Selection */}
                            <div className="mb-4">
                                <label className="block mb-2 text-sm font-medium">
                                    Select Fold  </label>
                                <select
                                    value={selectedFold}
                                    onChange={(e) => setSelectedFold(Number(e.target.value))}
                                    className="w-full border rounded p-2 text-sm"
                                >
                                    {Object.keys(trainingResult.pca || {}).map((foldKey) => {
                                        const foldNum = parseInt(foldKey.split('_')[1]);
                                        return (
                                            <option key={foldKey} value={foldNum}>
                                                Fold {foldNum}
                                            </option>
                                        );
                                    })}
                                </select>
                            </div>

                            {/* Normalized/Non-normalized Toggle */}
                            <div className="mb-4">
                                <label className="block mb-2 text-sm font-medium">
                                    Display Mode
                                </label>
                                <div className="flex gap-2">
                                    <button
                                        onClick={() => setIsNormalized(false)}
                                        className={`flex-1 px-3 py-2 rounded text-sm ${!isNormalized
                                            ? "bg-blue-600 text-white"
                                            : "bg-gray-200 text-gray-700"
                                            }`}
                                    >
                                        Original
                                    </button>
                                    <button
                                        onClick={() => setIsNormalized(true)}
                                        className={`flex-1 px-3 py-2 rounded text-sm ${isNormalized
                                            ? "bg-blue-600 text-white"
                                            : "bg-gray-200 text-gray-700"
                                            }`}
                                    >
                                        Normalized
                                    </button>
                                </div>
                            </div>

                            {/* Trajectory Toggle */}
                            <div>
                                <label className="block mb-2 text-sm font-medium">
                                    Show Trajectories
                                </label>
                                <div className="space-y-2">
                                    <label className="flex items-center">
                                        <input
                                            type="checkbox"
                                            checked={showOriginal}
                                            onChange={(e) => setShowOriginal(e.target.checked)}
                                            className="mr-2"
                                        />
                                        <span className="text-sm">Original Trajectory</span>
                                    </label>
                                    <label className="flex items-center">
                                        <input
                                            type="checkbox"
                                            checked={showReconstructed}
                                            onChange={(e) => setShowReconstructed(e.target.checked)}
                                            className="mr-2"
                                        />
                                        <span className="text-sm">Reconstructed Trajectory</span>
                                    </label>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {error && <div className="text-red-500 p-4">{error}</div>}

                {/* Middle: 3D Visualization Arena */}
                <div className="container-item middle-item">
                    {trainingResult ? (
                        trainingResult.pca ? (
                            <Arena3D
                                explained_variance={trainingResult.pca[`fold_${selectedFold}`].explained_variance}
                                projections={trainingResult.pca[`fold_${selectedFold}`].projections}
                                loss_surface={trainingResult.pca[`fold_${selectedFold}`].loss_surface}
                                isNormalized={isNormalized}
                                showOriginal={showOriginal}
                                showReconstructed={showReconstructed}
                            />
                        ) : (
                            <div className="flex items-center justify-center h-full text-gray-400">
                                PCA results unavailable
                            </div>
                        )
                    ) : (
                        <div className="flex items-center justify-center h-full text-gray-400">
                            Click "Run Training" to visualize 3D Loss Landscape and Optimizer Trajectory
                        </div>
                    )}
                </div>

                {/* Right: 2D Convergence plot */}
                <div className="container-item right-item">
                    {trainingResult ? (
                        trainingResult.pca ? (
                            <ConvergencePlot2D
                                projections={trainingResult.pca[`fold_${selectedFold}`].projections}
                                isNormalized={isNormalized}
                                showOriginal={showOriginal}
                                showReconstructed={showReconstructed}
                            />
                        ) : (
                            <div className="flex items-center justify-center h-full text-gray-400">
                                PCA results unavailable
                            </div>
                        )
                    ) : (
                        <div className="flex items-center justify-center h-full text-gray-400">
                            Click "Run Training" to visualize 2D Loss Convergence Plot
                        </div>
                    )}
                </div>
            </div>
            <div>
                {   
                    trainingResult ? (
                        <HUDPanel
                            trainingResult={trainingResult}
                            selectedFold={selectedFold}
                            isNormalized={isNormalized}
                            showOriginal={showOriginal}
                            showReconstructed={showReconstructed}
                            trainingParams={{
                                epochs: trainingParams?.epochs,
                                batch_size: trainingParams?.batch_size,
                                lr: trainingParams?.lr
                            }}
                        />) : (<div></div>)
                }
            </div>
        </div>
        
    );
}; 

export default Home;