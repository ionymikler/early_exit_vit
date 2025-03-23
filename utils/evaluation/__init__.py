# Module structure for evaluation utilities

# evaluation/
# ├── __init__.py
# ├── structures.py       # Data structures for evaluation
# ├── metrics.py          # Metric calculation functions
# ├── visualizers.py      # Visualization functions
# └── evaluator.py        # Main evaluation functions

# evaluation/__init__.py
"""
Evaluation module for EEVIT models.

This module provides tools for evaluating EEVIT models, including:
- Data collection during evaluation
- Metric calculation from raw data
- Visualization of evaluation results

Main components:
- structures: Data structures for collecting evaluation results
- metrics: Functions for calculating performance metrics
- visualizers: Functions for creating visualizations
- evaluator: Main evaluation functions
"""

from .structures import BatchResult, ExitCollector, ClassCollector, EvaluationResults
from .metrics import (
    calculate_exit_statistics,
    calculate_class_statistics,
    calculate_confusion_matrix,
    calculate_advanced_metrics,
    build_final_metrics,
)

# from .visualizers import plot_confusion_matrix
from .evaluator import evaluate_pytorch_model, evaluate_onnx_model

__all__ = [
    # Data structures
    "BatchResult",
    "ExitCollector",
    "ClassCollector",
    "EvaluationResults",
    # Metric calculation
    "calculate_exit_statistics",
    "calculate_class_statistics",
    "calculate_confusion_matrix",
    "calculate_advanced_metrics",
    "build_final_metrics",
    # Visualization
    "plot_confusion_matrix",
    # Evaluation functions
    "evaluate_pytorch_model",
    "evaluate_onnx_model",
]
