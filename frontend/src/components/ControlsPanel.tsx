import React, { useState } from "react";
import type { TrainRequest } from "../types/backend";
import { DATASETS } from "../types/datasetOptions";   // import dataset mapping
import "../App.css";

interface Props {
    onRun: (params: TrainRequest) => void;
    disabled: boolean;
}

const ControlsPanel: React.FC<Props> = ({ onRun, disabled }) => {
    const [taskType, setTaskType] = useState<TrainRequest["task_type"]>("regression");
    const [hiddenLayers, setHiddenLayers] = useState<number[]>([8, 8]);
    const [newLayer, setNewLayer] = useState<number>(8);
    const [form, setForm] = useState<TrainRequest>({
        dataset_name: "synthetic",
        task_type: "regression",
        optimizer_name: "adam",
        K: 3,
        epochs: 10,
        lr: 0.01,
        compute_pca: true,
        k_pca: 3,
    });

    const addLayer = () => {
        setHiddenLayers(prev => {
            const newLayers = [...prev, newLayer];
            setForm(prevForm => ({
                ...prevForm,
                hidden_layers: newLayers
            }));
            return newLayers;
        });
    };

    const removeLayer = (indexToRemove: number) => {
        console.log("Before removing hidden layer: ", hiddenLayers);
        console.log("Removing item at index: ", indexToRemove);
        setHiddenLayers(prev => {
            const newLayers = prev.filter((_, i) => { return i !== indexToRemove });
            console.log("After removing item at index: ", newLayers);
            setForm(prevForm => ({
                ...prevForm,
                hidden_layers: newLayers
            }));
            return newLayers;
        });
    };

    const updateHiddenLayer = (index: number, neurons: number) => {
        setHiddenLayers(prev => {
            const newLayers = [...prev];
            newLayers[index] = neurons;
            setForm(prevForm => ({
                ...prevForm,
                hidden_layers: newLayers
            }));
            return newLayers;
        });
    };

    const handleChange = (key: keyof TrainRequest, value: any) => {
        setForm((prev) => ({ ...prev, [key]: value }));
    };

    const handleTaskChange = (value: TrainRequest["task_type"]) => {
        setTaskType(value);
        const firstDataset = DATASETS[value][0].value;
        handleChange("task_type", value);
        handleChange("dataset_name", firstDataset);
    };

    return (
        <div className="p-4 bg-white shadow rounded h-full overflow-y-auto">

            <h2 className="text-lg font-semibold mb-4">Network Architecture</h2>

            {/* NEURAL NETWORK ARCHITECTURE */}
            <div className="mb-6">
                <label className="block mb-2">Hidden Layers</label>
                {hiddenLayers.map((neurons, index) => (
                    <div key={index} className="flex gap-2 mb-2">
                        <span className="text-sm">Layer {index + 1}: </span>
                        <input
                            type="number"
                            value={neurons}
                            onChange={(e) => updateHiddenLayer(index, Number(e.target.value))}
                            className="w-20 border rounded p-1"
                            min="1"
                        />
                        <button
                            onClick={() => removeLayer(index)}
                            className="text-red-600 text-sm"
                        >
                            Remove
                        </button>
                    </div>
                ))}
                <div>
                    <label className="block mb-2"># Neurons in New Layer: </label>
                    <input
                        type="number"
                        value={newLayer}
                        onChange={(e) => setNewLayer(Number(e.target.value))}
                        className="w-full mb-6 border rounded p-1"
                    />
                    <button
                        onClick={addLayer}
                        className="text-blue-600 text-sm mt-2"
                    >
                        + Add Layer
                    </button>
                </div>

                <p className="text-xs text-gray-500 mt-2">
                    Architecture: [{hiddenLayers.join(', ')}]
                </p>

            </div>

            <h2 className="text-lg font-semibold mb-4">Training Controls</h2>

            {/* TASK TYPE DROPDOWN */}
            <div className="mb-6">
                <label className="block mb-2">Task Type </label>
                <select
                    value={taskType}
                    onChange={(e) => handleTaskChange(e.target.value as TrainRequest["task_type"])}
                    className="w-full mb-6 border rounded p-1"
                >
                    <option value="regression">Regression</option>
                    <option value="binary_classification">Binary Classification</option>
                    <option value="multiclass_classification">Multi-class Classification</option>
                </select>
            </div>

            {/* DATASET DROPDOWN — FILTERED BASED ON TASK TYPE */}
            <div className="mb-6">
                <label className="block mb-2">Dataset </label>
                <select
                    value={form.dataset_name}
                    onChange={(e) => handleChange("dataset_name", e.target.value)}
                    className="w-full mb-6 border rounded p-1"
                >
                    {DATASETS[taskType].map((ds) => (
                        <option key={ds.value} value={ds.value}>
                            {ds.label}
                        </option>
                    ))}
                </select>
            </div>

            {/*SYNTHETIC REGRESSION VARIANT DROPDOWN*/}
            {taskType === "regression" && (
                <div className="mb-6">
                    <label className="block mb-2">Regression Function Variant </label>
                    <select
                        value={form.func_variant}
                        onChange={(e) => handleChange("func_variant", e.target.value)}
                        className="w-full mb-6 border rounded p-1"
                    >
                        <option value="simple">Simple</option>
                        <option value="medium">Medium</option>
                        <option value="complex">Complex</option>
                    </select>
                </div>
            )}

            
            {/* NUMBER OF FOLDS*/}
            <div className="mb-6">
                <label className="block mb-2">Number of Folds </label>
                <input
                    type="number"
                    value={form.K}
                    onChange={(e) => handleChange("K", +e.target.value)}
                    className="w-full mb-6 border rounded p-1"
                />
            </div>

            {/* OPTIMIZER */}
            <div className="mb-6">
                <label className="block mb-2">Optimizer </label>
                <select
                    value={form.optimizer_name}
                    onChange={(e) => handleChange("optimizer_name", e.target.value)}
                    className="w-full mb-6 border rounded p-1"
                >
                    <option value="fullbatch_gradientdescent">Full-Batch Gradient Descent</option>
                    <option value="minibatch_gradientdescent">Mini-Batch Gradient Descent</option>
                    <option value="sgd">SGD</option>
                    <option value="adam">Adam</option>
                    <option value="bfgs">BFGS</option>
                    <option value="rmsprop">RMSProp</option>
                </select>
            </div>

            {/* EPOCHS */}
            <div className="mb-6">
                <label className="block mb-2">Epochs </label>
                <input
                    type="number"
                    value={form.epochs}
                    onChange={(e) => handleChange("epochs", +e.target.value)}
                    className="w-full mb-6 border rounded p-1"
                />
            </div>

            {/*BATCH SIZE*/}
            <div className="mb-6">
                <label className="block mb-2">Batch Size </label>
                <select
                    value={Number(form.batch_size)}
                    onChange={(e) => handleChange("batch_size", e.target.value === "" ? null : Number(e.target.value))}
                    className="w-full mb-6 border rounded p-1"
                >
                    {/* SGD → always batch size 1 */}
                    {form.optimizer_name.toLowerCase() === "sgd" && (
                        <option value="1"> 1 (SGD uses batch size 1) </option>
                    )}
                    {/* Mini-Batch GD → choose from predefined sizes */}
                    {form.optimizer_name.toLowerCase() === "minibatch_gradientdescent" && (
                        <>
                            <option value="16">16</option>
                            <option value="32">32</option>
                            <option value="64">64</option>
                        </>
                    )}
                    {/* All other optimizers → Full batch only */}
                    {form.optimizer_name.toLowerCase() !== "sgd" &&
                        form.optimizer_name.toLowerCase() !== "minibatch_gradientdescent" && (
                            <option value="">Full Batch</option>
                        )}
                </select>
            </div>

            <div className="mb-6">
                <label className="block mb-2">PCA Components </label>
                <input
                    type="number"
                    value={form.k_pca ? form.k_pca : ""}
                    onChange={(e) => handleChange("k_pca", e.target.value)}
                    className="w-full mb-6 border rounded p-1"
                />
            </div>

            <button
                onClick={() => onRun(form)}
                disabled={disabled}
                className="button-style"
            >
                {disabled ? "Running..." : "Run Training"}
            </button>
        </div>
    );
};

export default ControlsPanel;
