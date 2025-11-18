import React from 'react';
import Plot from 'react-plotly.js';
import * as Plotly from "plotly.js-dist-min";


interface LossSurfacePlotProps {
    surface: { A: number[][]; B: number[][]; L: number[][] };
    trajectory?: { Z: number[][]; loss?: number[]; reconstructed_loss?: number[]; };
}

const LossSurfacePlot: React.FC<LossSurfacePlotProps> = ({ surface, trajectory }) => {
    const { A, B, L } = surface;

    // Debug logging
    console.log('Surface data:', { A, B, L });
    console.log('A shape:', A.length, 'x', A[0]?.length);
    console.log('B shape:', B.length, 'x', B[0]?.length);
    console.log('L shape:', L.length, 'x', L[0]?.length);

    // Extract 1D coordinate arrays
    // Assuming A and B are meshgrids where:
    // - A[i][j] contains the x-coordinate at grid point (i,j)
    // - B[i][j] contains the y-coordinate at grid point (i,j)

    const xCoords = A[0];  // First row contains all unique x values
    const yCoords = B.map(row => row[0]);  // First column contains all unique y values

    console.log('xCoords:', xCoords);
    console.log('yCoords:', yCoords);
    console.log('Z (loss) grid:', L);

    const data: Plotly.Data[] = [
        {
            x: A[0],
            y: B.map((row) => row[0]),
            z: L,
            type: 'surface',
            colorscale: 'Viridis',
            opacity: 0.85,
            showscale: true,
        },
    ];

    if (trajectory) {
        data.push({
            x: trajectory.Z.map((p) => p[0]),
            y: trajectory.Z.map((p) => p[1]),
            z: trajectory.reconstructed_loss || [],
            mode: 'lines+markers',
            type: 'scatter3d',
            line: { color: 'red', width: 4 },
            marker: { size: 4, color: 'red' },
            name: 'Optimizer Path',
        });
    }

    const layout: Partial<Plotly.Layout> = {
        autosize: true,
        title: { text: 'Optimizer Trajectory on Loss Surface' },
        scene: {
            xaxis: {
                title: { text: 'PC1(α)' }
            },
            yaxis: {
                title: { text: 'PC2 (β)' }
            },
            zaxis: {
                title: { text: 'Loss' }
            },
        },
        height: 600,
    };

    return (
        <div className="mt-6">
            <h3>Loss Surface Info:</h3>
            <h2> Loss values (L):</h2>
            {surface.L.map((row, rowIndex) => (
                <div key={'L-row-${rowIndex}'}>
                    {row.map((value, colIndex) => (
                        <span key={'L-row-${rowIndex}-col-${colIndex}'} style={{ margin: '0.5px' }}>
                            {value.toFixed(4)}
                        </span>
                    ))}
                    <br />
                </div>
            ))}
            <h3>Trajectory points Z</h3>
            {trajectory?.Z.map((row, rowIndex) => (
                <div key={'Z-row-${rowIndex}'}>
                    {row.map((value, colIndex) => (
                        <span key={'Z-row-${rowIndex}-col-${colIndex}'} style={{ margin: '0.5px' }}>
                            {value.toFixed(4)}
                        </span>
                    ))}
                    <br />
                </div>
            ))}
            <Plot
                data={data}
                layout={layout}
                config={{ responsive: true }}
                useResizeHandler
                style={{ width: "100%", height: "100%" }}
                divId="plot"
            />

        </div>
        );
};

export default LossSurfacePlot;

