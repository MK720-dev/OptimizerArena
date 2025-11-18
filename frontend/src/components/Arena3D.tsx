import React, { useState } from "react";
import type { PCALossResult, LossSurface } from "../types/backend";
import Plot from "react-plotly.js";
import * as Plotly from "plotly.js-dist-min";

interface Arena3DProps extends PCALossResult {
    isNormalized: boolean;
    showOriginal: boolean;
    showReconstructed: boolean;
}

const Arena3D: React.FC<Arena3DProps> = ({
    explained_variance,
    projections,
    loss_surface,
    isNormalized,
    showOriginal,
    showReconstructed
}) => {
    if (!loss_surface) {
        return <p> No loss surface data available. Train First. </p>
    };

    // Select appropriate loss surface based on normalized mode
    const surfaceKey = isNormalized ? "normalized_loss" : "original_loss";
    // Extract surface components
    const { A, B, L }: LossSurface = loss_surface[surfaceKey];

    // Pick first Projection
    const trajectory = projections ? Object.values(projections)[0] : undefined;

    const data: Plotly.Data[] = [
        {
            x: A[0],
            y: B.map(row => row[0]),
            z: L,
            type: "surface",
            colorscale: "Viridis",
            opacity: 0.9,
            showscale: true,
            name: isNormalized ? "Normalized Loss Surface" : "Loss Surface"
        } as Plotly.Data,
    ];

    if (trajectory) {
        // Add original trajectory
        if (showOriginal) {
            const originalLoss = isNormalized ? trajectory.normalized_loss : trajectory.loss;
            data.push({
                x: trajectory.Z_viz.map(p => p[0]),
                y: trajectory.Z_viz.map(p => p[1]),
                z: originalLoss,
                type: "scatter3d",
                mode: "lines+markers",
                line: { color: "red", width: 5 },
                marker: { size: 5, color: "red" },
                name: isNormalized ? "Normalized Original Trajectory" : "Original Trajectory"
            } as Plotly.Data);
        }
        if (showReconstructed) {
            const reconstructedLoss = isNormalized
                ? trajectory.normalized_reconstructed_loss
                : trajectory.reconstructed_loss;

            data.push({
                x: trajectory.Z_viz.map(p => p[0]),
                y: trajectory.Z_viz.map(p => p[1]),
                z: reconstructedLoss,
                type: "scatter3d",
                mode: "lines+markers",
                line: { color: "cyan", width: 5 },
                marker: { size: 5, color: "cyan" },
                name: isNormalized ? "Normalized Reconstructed Trajectory" : "Reconstructed Trajectory"
            } as Plotly.Data);
        }
        
    };

    const layout: Partial<Plotly.Layout> = {
        autosize: true,
        scene: {
            xaxis: {
                title: { text: 'PC1(α)' }
            },
            yaxis: {
                title: { text: 'PC2 (β)' }
            },
            zaxis: {
                title: { text: isNormalized ? 'Normalized Loss' : 'Loss' }
            },
        },
        margin: {t: 40},
        height: 600,
    };
   
    return (
        <div className="w-full h-full p-4 bg-white shadow rounded flex flex-col">
          <h2 className="text-center text-xl font-bold mb-2">
                Optimizer Arena: {isNormalized ? 'Normalized ' : ''}Loss Surface & Trajectory
          </h2>

          <Plot
              data={data}
              layout={layout}
              style={{ width: "100%", height: "100%" }}
              config={{ responsive: true }}
          />
      </div>
    );
}

export default Arena3D;