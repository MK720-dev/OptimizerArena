export const DATASETS: Record<
    "regression" | "binary_classification" | "multiclass_classification",
    { value: string; label: string }[]
> = {
    regression: [
        { value: "synthetic", label: "Synthetic Function" },
        { value: "california_housing", label: "California Housing" },
    ],

    binary_classification: [
        { value: "breast_cancer", label: "Breast Cancer" },
    ],

    multiclass_classification: [
        { value: "iris", label: "Iris Dataset" },
    ],
};
