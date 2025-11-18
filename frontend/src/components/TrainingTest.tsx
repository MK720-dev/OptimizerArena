import React, { useState } from "react";
import { runTraining } from "../api/backend";
import type { TrainRequest, TrainResponse } from "../types/backend";
import LossSurfacePlot from "./TrainingPlot"

const TrainingTest: React.FC = () => {
    const [result, setResult] = useState<TrainResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [selectedFold, setSelectedFold] = useState<string>("fold_0");

    const handleTrain = async () => {
        setLoading(true);
        setError(null);

        const req: TrainRequest = {
            dataset_name: "breast_cancer",
            task_type: "binary_classification",
            optimizer_name: "fullbatch_gradientdescent",
            K: 3,
            epochs: 5,
            compute_pca: true,
            k_pca: 2,
        };

        try {
            const res = await runTraining(req);
            setResult(res);
        } catch (err: any) {
            console.error(err);
            setError("Training failed. Check backend logs.");
        } finally {
            setLoading(false);
        }
    }; return (
        <div className="p-6">
            <h2 className="text-xl font-bold mb-4">Optimizer Arena Training</h2>
            <button
                onClick={handleTrain}
                disabled={loading}
                className="px-4 py-2 bg-blue-600 text-white rounded"
            >
                {loading ? "Training..." : "Run Training"}
            </button>

            {error && <p className="text-red-600 mt-2">{error}</p>}

            {/* Once we have results */}
            {result && (
                <div className="mt-6">
                    <h3 className="font-semibold">Training Complete ✅</h3>
                    <p>Optimizer: {result.optimizer}</p>
                    <p>Dataset: {result.dataset}</p>
                    <p>Task: {result.task_type}</p>
                    <p>
                        Avg Val Loss:{" "}
                        {result.fold_results.avg_val_loss?.toFixed(4) ?? "N/A"}
                    </p>

                    {/* Dropdown to select fold */}
                    {result.pca && (
                        <div className="my-3">
                            <label className="mr-2">Select Fold:</label>
                            <select
                                onChange={(e) => setSelectedFold(e.target.value)}
                                value={selectedFold}
                                className="border px-2 py-1 rounded"
                            >
                                {Object.keys(result.pca).map((foldKey) => (
                                    <option key={foldKey} value={foldKey}>
                                        {foldKey}
                                    </option>
                                ))}
                            </select>
                        </div>
                    )}
                    
                    {/* Render 3D Loss Surface Plot */}
                    {result.pca?.[selectedFold]?.loss_surface &&
                        result.pca?.[selectedFold]?.projections && (
                            <LossSurfacePlot
                                surface={result.pca[selectedFold].loss_surface}
                                trajectory={
                                    Object.values(
                                        result.pca[selectedFold].projections
                                    )[0]
                                }
                            />
                        )}
                </div>
            )}
        </div>
    );
};

export default TrainingTest;


