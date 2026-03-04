"""
Registry estático de algoritmos de ML y sus hiperparámetros configurables.
Usado por el frontend para renderizar formularios dinámicos
y por el backend para validación.
"""
from typing import Any, Dict, List


# Definición de cada hiperparámetro:
#   name: nombre del parámetro (coincide con el kwarg de sklearn)
#   label: nombre legible para la UI
#   type: "int", "float", "bool", "select"
#   min/max: rango para int/float
#   default: valor por defecto
#   options: para type="select", lista de {label, value}

ALGORITHM_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── Clasificadores ────────────────────────────────────────────────────────
    "random_forest": {
        "display_name": "Random Forest",
        "task_types": ["classification", "regression"],
        "sklearn_class": "sklearn.ensemble.RandomForestClassifier",
        "sklearn_regressor": "sklearn.ensemble.RandomForestRegressor",
        "supports_proba": True,
        "supports_tree_variance": True,
        "hyperparams": [
            {"name": "n_estimators", "label": "Number of Trees", "type": "int", "min": 10, "max": 1000, "default": 100},
            {"name": "max_depth", "label": "Max Depth", "type": "int", "min": 1, "max": 100, "default": 10},
            {"name": "min_samples_split", "label": "Min Samples Split", "type": "int", "min": 2, "max": 50, "default": 2},
            {"name": "min_samples_leaf", "label": "Min Samples Leaf", "type": "int", "min": 1, "max": 50, "default": 1},
            {"name": "max_features", "label": "Max Features", "type": "select", "default": "sqrt",
             "options": [{"label": "√n features", "value": "sqrt"}, {"label": "log2(n)", "value": "log2"}, {"label": "All", "value": None}]},
        ],
    },
    "gradient_boosting": {
        "display_name": "Gradient Boosting",
        "task_types": ["classification", "regression"],
        "sklearn_class": "sklearn.ensemble.GradientBoostingClassifier",
        "sklearn_regressor": "sklearn.ensemble.GradientBoostingRegressor",
        "supports_proba": True,
        "supports_tree_variance": True,
        "hyperparams": [
            {"name": "n_estimators", "label": "Number of Boosting Rounds", "type": "int", "min": 10, "max": 1000, "default": 100},
            {"name": "learning_rate", "label": "Learning Rate", "type": "float", "min": 0.001, "max": 1.0, "default": 0.1},
            {"name": "max_depth", "label": "Max Depth", "type": "int", "min": 1, "max": 20, "default": 3},
            {"name": "subsample", "label": "Subsample Ratio", "type": "float", "min": 0.1, "max": 1.0, "default": 1.0},
        ],
    },
    "logistic_regression": {
        "display_name": "Logistic Regression",
        "task_types": ["classification"],
        "sklearn_class": "sklearn.linear_model.LogisticRegression",
        "supports_proba": True,
        "supports_tree_variance": False,
        "hyperparams": [
            {"name": "C", "label": "Regularization Strength (C)", "type": "float", "min": 0.001, "max": 100.0, "default": 1.0},
            {"name": "max_iter", "label": "Max Iterations", "type": "int", "min": 50, "max": 5000, "default": 100},
            {"name": "solver", "label": "Solver", "type": "select", "default": "lbfgs",
             "options": [{"label": "LBFGS", "value": "lbfgs"}, {"label": "Liblinear", "value": "liblinear"}, {"label": "SAG", "value": "sag"}]},
        ],
    },
    "svm": {
        "display_name": "Support Vector Machine",
        "task_types": ["classification", "regression"],
        "sklearn_class": "sklearn.svm.SVC",
        "sklearn_regressor": "sklearn.svm.SVR",
        "supports_proba": True,  # with probability=True
        "supports_tree_variance": False,
        "hyperparams": [
            {"name": "C", "label": "Regularization (C)", "type": "float", "min": 0.01, "max": 100.0, "default": 1.0},
            {"name": "kernel", "label": "Kernel", "type": "select", "default": "rbf",
             "options": [{"label": "RBF", "value": "rbf"}, {"label": "Linear", "value": "linear"}, {"label": "Polynomial", "value": "poly"}]},
            {"name": "gamma", "label": "Gamma", "type": "select", "default": "scale",
             "options": [{"label": "Scale", "value": "scale"}, {"label": "Auto", "value": "auto"}]},
        ],
    },
    "knn": {
        "display_name": "K-Nearest Neighbors",
        "task_types": ["classification", "regression"],
        "sklearn_class": "sklearn.neighbors.KNeighborsClassifier",
        "sklearn_regressor": "sklearn.neighbors.KNeighborsRegressor",
        "supports_proba": True,
        "supports_tree_variance": False,
        "hyperparams": [
            {"name": "n_neighbors", "label": "Number of Neighbors (K)", "type": "int", "min": 1, "max": 50, "default": 5},
            {"name": "weights", "label": "Weight Function", "type": "select", "default": "uniform",
             "options": [{"label": "Uniform", "value": "uniform"}, {"label": "Distance", "value": "distance"}]},
            {"name": "metric", "label": "Distance Metric", "type": "select", "default": "minkowski",
             "options": [{"label": "Minkowski", "value": "minkowski"}, {"label": "Euclidean", "value": "euclidean"}, {"label": "Manhattan", "value": "manhattan"}]},
        ],
    },
    "decision_tree": {
        "display_name": "Decision Tree",
        "framework": "sklearn",
        "task_types": ["classification", "regression"],
        "sklearn_class": "sklearn.tree.DecisionTreeClassifier",
        "sklearn_regressor": "sklearn.tree.DecisionTreeRegressor",
        "supports_proba": True,
        "supports_tree_variance": False,
        "hyperparams": [
            {"name": "max_depth", "label": "Max Depth", "type": "int", "min": 1, "max": 100, "default": 10},
            {"name": "min_samples_split", "label": "Min Samples Split", "type": "int", "min": 2, "max": 50, "default": 2},
            {"name": "min_samples_leaf", "label": "Min Samples Leaf", "type": "int", "min": 1, "max": 50, "default": 1},
            {"name": "criterion", "label": "Criterion", "type": "select", "default": "gini",
             "options": [{"label": "Gini", "value": "gini"}, {"label": "Entropy", "value": "entropy"}, {"label": "Log Loss", "value": "log_loss"}]},
        ],
    },

    # ── PyTorch Architectures ─────────────────────────────────────────────────
    "mlp": {
        "display_name": "Neural Network (MLP)",
        "framework": "pytorch",
        "task_types": ["classification", "regression"],
        "supports_proba": False,
        "supports_tree_variance": False,
        "hyperparams": [
            {"name": "hidden_layers", "label": "Hidden Layers (comma-sep)", "type": "select", "default": "128,64",
             "options": [
                 {"label": "64", "value": "64"},
                 {"label": "128, 64", "value": "128,64"},
                 {"label": "256, 128, 64", "value": "256,128,64"},
                 {"label": "512, 256, 128", "value": "512,256,128"},
                 {"label": "128, 128", "value": "128,128"},
             ]},
            {"name": "epochs", "label": "Epochs", "type": "int", "min": 5, "max": 500, "default": 50},
            {"name": "learning_rate", "label": "Learning Rate", "type": "float", "min": 0.00001, "max": 0.1, "default": 0.001},
            {"name": "batch_size", "label": "Batch Size", "type": "int", "min": 8, "max": 512, "default": 64},
            {"name": "dropout", "label": "Dropout", "type": "float", "min": 0.0, "max": 0.8, "default": 0.2},
            {"name": "activation", "label": "Activation", "type": "select", "default": "relu",
             "options": [
                 {"label": "ReLU", "value": "relu"},
                 {"label": "LeakyReLU", "value": "leaky_relu"},
                 {"label": "ELU", "value": "elu"},
                 {"label": "GELU", "value": "gelu"},
                 {"label": "Tanh", "value": "tanh"},
             ]},
            {"name": "optimizer", "label": "Optimizer", "type": "select", "default": "adam",
             "options": [
                 {"label": "Adam", "value": "adam"},
                 {"label": "AdamW", "value": "adamw"},
                 {"label": "SGD", "value": "sgd"},
             ]},
            {"name": "weight_decay", "label": "Weight Decay (L2)", "type": "float", "min": 0.0, "max": 0.1, "default": 0.0},
            {"name": "scheduler", "label": "LR Scheduler", "type": "select", "default": "none",
             "options": [
                 {"label": "None", "value": "none"},
                 {"label": "Cosine Annealing", "value": "cosine"},
                 {"label": "Step (×0.1 every ⅓ epochs)", "value": "step"},
                 {"label": "Reduce on Plateau", "value": "reduce_on_plateau"},
             ]},
        ],
    },
    "unet": {
        "display_name": "UNet (Segmentation)",
        "framework": "pytorch",
        "task_types": ["segmentation"],
        "supports_proba": False,
        "supports_tree_variance": False,
        "hyperparams": [
            {"name": "in_channels", "label": "Input Channels", "type": "int", "min": 1, "max": 32, "default": 3},
            {"name": "num_classes", "label": "Number of Classes", "type": "int", "min": 1, "max": 100, "default": 2},
            {"name": "epochs", "label": "Epochs", "type": "int", "min": 5, "max": 300, "default": 30},
            {"name": "learning_rate", "label": "Learning Rate", "type": "float", "min": 0.00001, "max": 0.01, "default": 0.0001},
            {"name": "batch_size", "label": "Batch Size", "type": "int", "min": 1, "max": 64, "default": 4},
            {"name": "optimizer", "label": "Optimizer", "type": "select", "default": "adam",
             "options": [
                 {"label": "Adam", "value": "adam"},
                 {"label": "AdamW", "value": "adamw"},
                 {"label": "SGD", "value": "sgd"},
             ]},
        ],
    },
}


def get_algorithm_info(algorithm: str) -> Dict[str, Any]:
    """Devuelve la info de un algoritmo o lanza ValueError."""
    if algorithm not in ALGORITHM_REGISTRY:
        raise ValueError(
            f"Algoritmo '{algorithm}' no registrado. "
            f"Disponibles: {list(ALGORITHM_REGISTRY.keys())}"
        )
    return ALGORITHM_REGISTRY[algorithm]


def get_all_algorithms() -> List[Dict[str, Any]]:
    """Devuelve la lista de todos los algoritmos con su metadata."""
    result = []
    for key, info in ALGORITHM_REGISTRY.items():
        result.append({
            "id": key,
            "display_name": info["display_name"],
            "framework": info.get("framework", "sklearn"),
            "task_types": info["task_types"],
            "supports_proba": info["supports_proba"],
            "supports_tree_variance": info["supports_tree_variance"],
            "hyperparams": info["hyperparams"],
        })
    return result


def get_default_hyperparams(algorithm: str) -> Dict[str, Any]:
    """Devuelve los hiperparámetros por defecto de un algoritmo."""
    info = get_algorithm_info(algorithm)
    return {hp["name"]: hp["default"] for hp in info["hyperparams"]}

