
# OptimizerArena

OptimizerArena is a research-grade visualization platform for analyzing neural network optimization algorithms through interactive 3D loss-landscape exploration. This document provides a comprehensive technical overview of the system’s mathematical foundations—particularly the role of Principal Component Analysis (PCA) in enabling high-dimensional weight-trajectory visualization—along with backend algorithmic details, frontend architecture, and design decisions that balance mathematical correctness with visual interpretability.

---

## Overview

OptimizerArena extends a custom neural-network training engine with advanced tools for:

* Comparing optimizers (SGD, RMSProp, Adam, BFGS, etc.)
* Capturing weight trajectories during training
* Performing PCA on high-dimensional weight vectors
* Sampling 2D slices of the loss landscape
* Rendering high-fidelity 3D visualizations of optimization paths vs. local geometry

The platform is designed for research, teaching, and the empirical study of optimization behavior in deep learning.

## Visual Demo

Quick demo available here:  
👉 [Download / Watch Demo 1](./OptimizerArenaDemo1.mp4)
👉 [Download / Watch Demo 2](./OptimizerArenaDemo2.mp4)

---
# 1. Web App Architecture Overview

                ┌─────────────────────────────────┐
                │       Python Backend            │
                │        (FastAPI API)            │
                │   - Training engine             │
                │   - Optimizers (SGD, BFGS,...)  │
                │   - PCA + Loss Surface          │
                └───────────────┬─────────────────┘
                                │
                                │  HTTP (JSON)
                                │  /train, /pca
                                ▼
                 ┌────────────────────────────┐
                 │    Vite Dev Server         │
                 │  (Node.js Proxy Gateway)   │
                 │                            │
                 │  Proxies: /api/* → FastAPI │
                 └───────────────┬────────────┘
                                 │
                                 │  Local WebSocket + HTTP
                                 ▼
                 ┌───────────────────────────┐
                 │       React Frontend      │
                 │  - Controls Panel UI      │
                 │  - Arena3D Visualization  │
                 │  - Plotly Loss Surface    │
                 │  - Optimizer Trajectory   │
                 └───────────────────────────┘

## FastAPI (Python backend)
Implements all core computation: neural-network training, optimizers, PCA, loss-surface reconstruction.
Exposes routes like /api/train and /api/pca.

## Node.js / Vite dev server
Acts as a reverse proxy, forwarding all /api/* requests to FastAPI during development.
Also serves the bundled React assets.

## React Frontend (Vite client)
Renders UI, collects user parameters, triggers backend training calls, and visualizes:

- optimizer trajectories

- PCA projections

- 3D loss surfaces with Plotly

# 2. Backend Architecture & Training Pipeline

## 2.1 Correct Loss Function Selection

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

## 2.2 BFGS Optimizer Redesign

The earlier implementation incorrectly nested BFGS iterations inside the epoch loop.

### After Fix:

* BFGS performs **one full quasi-Newton update per epoch**
* `max_iter` applies **only to line search convergence**
* Wolfe conditions enforced correctly
* Weight snapshots captured consistently at epoch granularity
* Trajectories correctly reflect the actual optimization path

This yields smoother, more mathematically valid BFGS behavior.

---

## 2.3 Weight Capture Strategy

Weights are stored:

* At **epoch 0**
* At **every epoch afterward**

This ensures:

* High-quality PCA input
* Matching array lengths across losses, weights, projections
* Clear visualization of trajectory evolution through training

---

## 2.4 Global Min–Max Normalization

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

# 3. PCA-Enhanced Visualization

Neural networks operate in extremely high-dimensional parameter spaces. PCA provides a principled way to reduce this to 2D or 3D for visualization.

---

## 3.1 The Visualization Challenge

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

## 3.2 PCA as the Solution

PCA extracts the dominant directions of variation in the optimizer trajectory.

It provides:

1. A meaningful 2D/3D coordinate system
2. A common frame for comparing different optimizers
3. A 2D plane (PC1–PC2) from which to sample the loss surface
4. Variance-explained metrics to assess projection quality

---

## 3.3 Mathematical Summary

### Step 1 — Collect weight snapshots

$$[
w_0, w_1, \dots, w_T \in \mathbb{R}^n
]$$

### Step 2 — Form the data matrix

$$
M \in \mathbb{R}^{(T+1)\times n}
$$

### Step 3 — Center the data

$$
X = M - \mu
$$

### Step 4 — Apply SVD

$$
X = U S V^T
$$

Right-singular vectors ( V ) are the **principal component directions**.

### Step 5 — Select first ( k ) PCs

Typically ( k = 2 ) or ( k = 3).

### Step 6 — Project weights into PCA space

$$
z = (w - \mu)V_k
$$

### Step 7 — Explained variance

$$
\text{var}_i = \frac{s_i^2}{\sum_j s_j^2}
$$

---

# 4. Weight Reconstruction & Loss Surface Rendering

## 4.1 Forward Projection

$$
z = (w - \mu)V_k
$$

## 4.2 Inverse Reconstruction

$$
\hat w = \mu + \sum_{i=1}^k z_i v_i
$$

This reconstructs weights for computing loss values at grid points or trajectory points.

---

## 4.3 Reconstruction Strategies

### **A. Trajectory Reconstruction — Use All PCs (k = larger)**

* Accurate loss evaluation
* Low reconstruction error
* Faithful reproduction of true training behavior

### **B. Loss Surface — Use Only PC1 & PC2**

We define:

$$
w(\alpha,\beta) = \mu + \alpha v_1 + \beta v_2
$$

Sampling this plane yields a **dense 2D grid** for the 3D surface plot.

Why only two?

* 3D visualization requires 2D domain + 1D loss
* Higher dimensions cannot be visualized
* PC1 & PC2 capture the dominant variation

---

## 4.4 Why the Trajectory Appears on the Surface

Even though trajectories use more PCs:

* PC1 & PC2 capture most meaningful motion
* Higher PCs often correspond to flat or low-impact directions
* Loss is effectively a function of the first two PCs
* Shared normalization ensures alignment

Small deviations arise from reconstruction error, but are usually imperceptible.

---

# 5. Frontend Architecture

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

# 6. Datasets overview

OptimizerArena supports multiple datasets spanning regression, binary classification, and multiclass classification, allowing users to evaluate optimizer behavior across a diverse set of learning problems. All datasets are standardized into the format:

```
X : (n_features, n_samples)
y : (1 or C, n_samples)
meta : { input_dim, output_dim, task_type, name }
```

## Synthetic Regression Datasets

Synthetic regression datasets are procedurally generated to provide controlled difficulty levels.
All synthetic inputs follow:

- Input sampling:

	$$𝑋 ∈ 𝑅^2 , 𝑋 ∼ 𝑁(0,1)$$

- Target generation:

	$$𝑦 = 𝑓(𝑋) + 𝜀, 𝜀∼𝑁(0,0.1)$$

Depending on the selected function func_variant ∈ { "simple", "medium", "complex" }, the target values come from one of the following analytic functions:

### Simple Function (Linear)

$$𝑦 = 3𝑥_1 + 2𝑥_2 + 𝜀$$

- Low curvature

- Good for verifying correctness of optimizers

- Produces an almost-convex loss surface

### Medium Function (Mildly Nonlinear)
$$𝑦 = 𝑥_1^2 + sin(𝑥_2) + 𝜀$$

- Introduces moderate nonlinearity

- Contains local curvature variations

- Useful for evaluating adaptive optimizers (Adam, RMSProp)

### Complex Function (Highly Nonlinear)
$$𝑦 = sin(𝑥_1𝑥_2) + 0.5𝑥_1^3 − 𝑥_2^2 + 𝜀$$

- Strong nonlinearity and multimodality

- Produces a rugged loss landscape

- Ideal for contrasting first-order vs. second-order behavior

### Usage in OptimizerArena

Synthetic datasets are configured by:

- dataset_name="synthetic"

- func_variant ∈ {"simple", "medium", "complex"}
## California Housing (Real Regression Dataset)

✔ Task: Regression
✔ Source: Scikit-learn’s fetch_california_housing()
✔ Shape:

Features: 8 (median income, house age, rooms, population, etc.)

Samples: ~20,000

This large real-world dataset allows testing:

- Optimizer scalability

- Sensitivity to feature scaling
 
- Behavior on noisy, heterogeneous data

## Breast Cancer (Binary Classification)

✔ Task: Binary classification
✔ Source: load_breast_cancer()
✔ Labels: 0 (benign), 1 (malignant)
✔ Features: 30

Used to benchmark:

- Binary cross-entropy vs MSE

- Stability of optimizers on real classification tasks

- Decision boundary sharpness for small networks

## Iris Dataset (Multiclass Classification)

✔ Task: Multiclass (3 classes)
✔ Source: load_iris()
✔ Features: 4
✔ Classes: {0, 1, 2}

Used for:

- Testing softmax cross-entropy

- Visualizing multiclass loss geometry

- Understanding optimizer behavior on well-conditioned low-dimensional data

Summary Table
Dataset Name	Task Type	Complexity	Notes
Synthetic (Simple)	Regression	Low	Linear model test
Synthetic (Medium)	Regression	Medium	Mild nonlinearities
Synthetic (Complex)	Regression	High	Rugged loss landscape
California Housing	Regression (real)	Medium	Large real dataset
Breast Cancer	Binary Classification	Low–Med	BCE loss evaluation
Iris	Multiclass Classification	Low	Softmax model test

---

# 7. Quick Start

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

# 8. Repository Structure

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

---

# 9. Future Work

This project opens the door to several meaningful extensions that can greatly expand the analytical depth and usability of OptimizerArena:

## More Advanced Loss-Landscape Modeling
Incorporate richer surrogate models (Gaussian Processes, kernel regression, neural fields) to generate smoother, more accurate 3D surfaces and improve alignment between reconstructed trajectories and the true loss geometry.

## Support for Additional Optimizers
Expand beyond SGD/Adam/RMSProp/BFGS to include Momentum SGD, Nesterov, AdamW, L-BFGS, K-FAC, Shampoo, and other modern second-order optimizers to enable broader comparative studies.

## Trajectory Animation & Multi-Optimizer Visualization
Add animated optimizer avatars, playback controls, and side-by-side trajectories for comparing how different optimizers traverse the same loss landscape.

## Expanded Dataset Library + User Uploads
Support more real-world datasets, richer synthetic functions, and user-uploaded datasets—making the platform robust across regression, binary classification, and multi-class settings.

## Refined PCA & Dimensionality Reduction Tools
Add per-fold PCA, user-selectable number of components, and reconstruction-error diagnostics. Investigate nonlinear alternatives like UMAP, t-SNE, or Isomap for complex trajectory manifolds.

## Unified Diagnostics Dashboard
Integrate loss curves, gradient norms, learning-rate schedules, curvature estimates (e.g., Hessian approximations), and per-epoch statistics into a cohesive analytics panel.

## Improved UI/UX and Visualization Controls
Enhance the interface with a cleaner control panel, dark mode, 2D/3D toggles, export options (PNG/JSON), and a more polished layout for professional use.

## Modular Plugin-Style Architecture
Allow researchers to plug in custom optimizers, datasets, activation functions, or visualization modules—turning OptimizerArena into an extensible research framework.

## Stability, Sensitivity, and Meta-Learning Studies
Use recorded trajectories to study optimizer robustness to initialization, noise, and hyperparameters, and explore meta-learning strategies to automate optimizer selection.

## Higher-Dimensional Landscape Exploration
Explore slicing techniques beyond 2-PC projection: use multi-plane slicing, 3-PC volumetric views, and neighborhood curvature estimation to capture more complex geometry.

---

# 10. References
## Optimization & Neural Network Training

- Nocedal, J., & Wright, S. (2006). Numerical Optimization (2nd ed.). Springer.
— Standard reference for quasi-Newton methods (BFGS, line search, curvature conditions).

- Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR.
— Introduces Adam, the adaptive optimizer used as a baseline.

## Loss Landscape Visualization & PCA Geometry

- Li, H., Xu, Z., Taylor, G., & Goldstein, T. (2018). Visualizing the Loss Landscape of Neural Nets. NeurIPS.
— Foundation for PCA-based projection of high-dimensional weight trajectories.

- Jolliffe, I. T. (2002). Principal Component Analysis. Springer.
— Classical treatment of PCA, eigen-decomposition, and dimensionality reduction.

## Surrogate Modeling & Surface Reconstruction

- Forrester, A., Sobester, A., & Keane, A. (2008). Engineering Design via Surrogate Modelling. Wiley.
— Background for surrogate-based surface approximation in high-dimensional spaces.

## Neural Optimization Dynamics

- Bottou, L., Curtis, F. E., & Nocedal, J. (2018). Optimization Methods for Large-Scale Machine Learning. SIAM Review.
— Analysis of SGD regimes, convergence, and curvature behavior.

## Additional Useful Background

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
— Chapters on optimization, curvature, and training instability.

# 11. Citation

If you use OptimizerArena for research or teaching, please cite this repository.

---
















