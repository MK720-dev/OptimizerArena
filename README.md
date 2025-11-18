
# OptimizerArena

OptimizerArena is a research-grade visualization platform for analyzing neural network optimization algorithms through interactive 3D loss-landscape exploration. This document provides a comprehensive technical overview of the system’s mathematical foundations—particularly the role of Principal Component Analysis (PCA) in enabling high-dimensional weight-trajectory visualization—along with backend algorithmic details, frontend architecture, and design decisions that balance mathematical correctness with visual interpretability.

---

## Overview

OptimizerArena extends a custom neural-network training engine with advanced tools for:

* Comparing optimizers (SGD, Momentum, RMSProp, Adam, BFGS, etc.)
* Capturing weight trajectories during training
* Performing PCA on high-dimensional weight vectors
* Sampling 2D slices of the loss landscape
* Rendering high-fidelity 3D visualizations of optimization paths vs. local geometry

The platform is designed for research, teaching, and the empirical study of optimization behavior in deep learning.

---

# 1. Backend Architecture & Training Pipeline

## 1.1 Correct Loss Function Selection

Previous versions incorrectly applied MSE to all tasks, including binary classification. This caused issues such as vanishing gradients and incorrect probabilistic assumptions.

OptimizerArena now **automatically selects the correct loss function**:

| Task Type                 | Loss                  | Activation | Rationale                                                  |
| ------------------------- | --------------------- | ---------- | ---------------------------------------------------------- |
| Regression                | MSE                   | Linear     | Natural for continuous outputs                             |
| Binary Classification     | Binary Cross-Entropy  | Sigmoid    | Proper probabilistic interpretation                        |
| Multiclass Classification | Softmax Cross-Entropy | Softmax    | Stable and mathematically aligned with class probabilities |

Additional backend improvements:

* Stable softmax + cross-entropy derivative formulation
* Automatic loss selection in `NeuralNetwork.set_loss()`
* Gradient clipping for numerical stability

---

## 1.2 BFGS Optimizer Redesign

The earlier implementation incorrectly nested BFGS iterations inside the epoch loop.

### After Fix:

* BFGS performs **one full quasi-Newton update per epoch**
* `max_iter` applies **only to line search convergence**
* Wolfe conditions enforced correctly
* Weight snapshots captured consistently at epoch granularity
* Trajectories correctly reflect the actual optimization path

This yields smoother, more mathematically valid BFGS behavior.

---

## 1.3 Weight Capture Strategy

Weights are stored:

* At **epoch 0**
* At **every epoch afterward**

This ensures:

* High-quality PCA input
* Matching array lengths across losses, weights, projections
* Clear visualization of trajectory evolution through training

---

## 1.4 Global Min–Max Normalization

Before:
Surface, trajectory, and reconstructed trajectory were normalized **independently**, causing scale mismatch and floating trajectories.

Now:

```python
scaled = (value - surface_min) / (surface_max - surface_min)
```

All elements use **one global normalization**, extracted from the loss surface only.

Benefits:

* A unified and mathematically consistent scale
* Accurate trajectory-to-surface alignment
* No misleading distortions

---

# 2. PCA-Enhanced Visualization

Neural networks operate in extremely high-dimensional parameter spaces. PCA provides a principled way to reduce this to 2D or 3D for visualization.

---

## 2.1 The Visualization Challenge

We want to:

* Visualize the shape of the loss landscape
* Visualize optimizer trajectories
* Compare optimizers geometrically

But:

* Weight space is typically thousands to millions of dimensions
* Loss is a function $( L : \mathbb{R}^n \to \mathbb{R} )$
* We cannot sample or visualize beyond 3 dimensions

Thus PCA becomes essential.

---

## 2.2 PCA as the Solution

PCA extracts the dominant directions of variation in the optimizer trajectory.

It provides:

1. A meaningful 2D/3D coordinate system
2. A common frame for comparing different optimizers
3. A 2D plane (PC1–PC2) from which to sample the loss surface
4. Variance-explained metrics to assess projection quality

---

## 2.3 Mathematical Summary

### Step 1 — Collect weight snapshots

$$[
w_0, w_1, \dots, w_T \in \mathbb{R}^n
]$$

### Step 2 — Form the data matrix

$$[
M \in \mathbb{R}^{(T+1)\times n}
]$$

### Step 3 — Center the data

$$[
X = M - \mu
]$$

### Step 4 — Apply SVD

$$[
X = U S V^T
]$$

Right-singular vectors ( V ) are the **principal component directions**.

### Step 5 — Select first ( k ) PCs

Typically ( k = 2 ) or ( k = 3).

### Step 6 — Project weights into PCA space

$$[
z = (w - \mu)V_k
]$$

### Step 7 — Explained variance

$$[
\text{var}_i = \frac{s_i^2}{\sum_j s_j^2}
]$$

---

# 3. Weight Reconstruction & Loss Surface Rendering

## 3.1 Forward Projection

$$[
z = (w - \mu)V_k
]$$

## 3.2 Inverse Reconstruction

$$[
\hat w = \mu + \sum_{i=1}^k z_i v_i
]$$

This reconstructs weights for computing loss values at grid points or trajectory points.

---

## 3.3 Reconstruction Strategies

### **A. Trajectory Reconstruction — Use All PCs (k = larger)**

* Accurate loss evaluation
* Low reconstruction error
* Faithful reproduction of true training behavior

### **B. Loss Surface — Use Only PC1 & PC2**

We define:

$$[
w(\alpha,\beta) = \mu + \alpha v_1 + \beta v_2
]$$

Sampling this plane yields a **dense 2D grid** for the 3D surface plot.

Why only two?

* 3D visualization requires 2D domain + 1D loss
* Higher dimensions cannot be visualized
* PC1 & PC2 capture the dominant variation

---

## 3.4 Why the Trajectory Appears on the Surface

Even though trajectories use more PCs:

* PC1 & PC2 capture most meaningful motion
* Higher PCs often correspond to flat or low-impact directions
* Loss is effectively a function of the first two PCs
* Shared normalization ensures alignment

Small deviations arise from reconstruction error, but are usually imperceptible.

---

# 4. Frontend Architecture

The frontend (React + Three.js) supports:

* Real-time 3D rotation, zoom, and pan
* Animated optimizer paths
* Loss-surface interpolation and color mapping
* Responsive UI with control menus

Communication with the backend occurs via REST endpoints for:

* PCA projections
* Trajectory data
* Reconstructed weights
* Loss surface grids

---

# 5. Quick Start

### **Backend (Terminal 1)**

```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

### **Frontend (Terminal 2)**

```bash
cd frontend
npm install
npm run dev
```

Open the app at:

```
http://localhost:5173
```

---

# 6. Repository Structure

```
OptimizerArena/
│
├── backend/
│   ├── app/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── utils/
│   │   ├── __pycache__/
│   │   ├── main.py
│   │   └── __init__.py
│   │
│   ├── venv/
│   ├── __pycache__/
│   ├── .env
│   ├── .env.example
│   ├── backend.pyproj
│   ├── backend.pyproj.user
│   ├── neural_network.py
│   ├── optimizers.py
│   └── requirements.txt
│
├── frontend/
│   ├── .vscode/
│   ├── node_modules/
│   ├── obj/
│   ├── public/
│   ├── src/
│   │   ├── (React components, hooks, styles, utils, etc.)
│   │   └── ...
│   │
│   ├── .env
│   ├── .env.example
│   ├── .gitignore
│   ├── CHANGELOG.md
│   ├── eslint.config.js
│   ├── frontend.esproj
│   ├── index.html
│   ├── package-lock.json
│   ├── package.json
│   ├── README.md
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   └── vite.config.ts
│
├── .gitignore
├── README.md
└── LICENSE

```


# 7. Citation

If you use OptimizerArena for research or teaching, please cite this repository.

---





