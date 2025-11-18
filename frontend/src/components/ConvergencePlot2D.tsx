import React, { useState } from "react";
import Plot from "react-plotly.js";
import * as Plotly from "plotly.js-dist-min";
import type { PCAProjection } from "../types/backend";

interface Props {
    projections: Record<string, PCAProjection>;
    isNormalized: boolean;
    showOriginal: boolean;
    showReconstructed: boolean;
}
const ConvergencePlot2D: React.FC<Props> = ({
    projections,
    isNormalized,
    showOriginal,
    showReconstructed
}) => {
    // Get the first projection
    const trajectory = projections ? Object.values(projections)[0] : undefined;

    if (!trajectory) {
        return (
            <div className="w-full h-full p-4 bg-white shadow rounded flex items-center justify-center">
                <p className="text-gray-400">No trajectory data available</p>
            </div>
        );
    }

    const data: Plotly.Data[] = [];

    // Determine which loss histories to display based on mode and toggles
    if (showOriginal) {
        const originalLoss = isNormalized ? trajectory.normalized_loss : trajectory.loss;
        const epochs = Array.from({ length: originalLoss.length }, (_, i) => i);

        data.push({
            x: epochs,
            y: originalLoss,
            type: "scatter",
            mode: "lines+markers",
            name: isNormalized ? "Normalized Original Loss" : "Original Loss",
            line: { color: "#ff5733", width: 3 },
            marker: { size: 6 }
        } as Plotly.Data);
    }

    if (showReconstructed) {
        const reconstructedLoss = isNormalized
            ? trajectory.normalized_reconstructed_loss
            : trajectory.reconstructed_loss;
        const epochs = Array.from({ length: reconstructedLoss.length }, (_, i) => i);

        data.push({
            x: epochs,
            y: reconstructedLoss,
            type: "scatter",
            mode: "lines+markers",
            name: isNormalized ? "Normalized Reconstructed Loss" : "Reconstructed Loss",
            line: { color: "#00d9ff", width: 3 },
            marker: { size: 6 }
        } as Plotly.Data);
    }

    const layout: Partial<Plotly.Layout> = {
        autosize: true,
        xaxis: {
            title: { text: 'Epoch' }
        },
        yaxis: {
            title: { text: isNormalized ? 'Normalized Loss' : 'Loss' }
        },
        margin: { t: 40, b: 50, l: 60, r: 20 },
        showlegend: true,
        legend: {
            x: 1,
            y: 1,
            xanchor: 'right',
            yanchor: 'top'
        }
    };

    return (
        <div className="w-full h-full p-4 bg-white shadow rounded flex flex-col">
            <h2 className="text-center text-xl font-bold mb-2">
                {isNormalized ? 'Normalized ' : ''}Loss Convergence
            </h2>
            <Plot
                data={data}
                layout={layout}
                style={{ width: "100%", height: "100%" }}
                config={{ responsive: true }}
            />
        </div>
    );
};

export default ConvergencePlot2D;