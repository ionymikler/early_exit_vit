#!/usr/bin/env python
# Made by: Jonathan Mikler on 2025-03-21

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np


@dataclass
class BatchResult:
    """
    Stores the results from evaluating a single batch of samples.
    """

    batch_size: int
    true_labels: np.ndarray
    predicted_classes: np.ndarray
    confidences: np.ndarray
    exit_layer: int  # Layer index or -1 for final layer
    inference_time_ms: float
    is_correct: np.ndarray  # Boolean array of correct/incorrect predictions

    def __post_init__(self):
        # Validate dimensions
        assert len(self.true_labels) == self.batch_size, "true_labels size mismatch"
        assert (
            len(self.predicted_classes) == self.batch_size
        ), "predicted_classes size mismatch"
        assert len(self.confidences) == self.batch_size, "confidences size mismatch"
        assert len(self.is_correct) == self.batch_size, "is_correct size mismatch"


@dataclass
class ExitCollector:
    """
    Collects and stores evaluation data for a specific exit point.
    """

    exit_key: str  # "final" or "exit_X" where X is the exit index
    count: int = 0
    correct: int = 0
    confidences: List[float] = field(default_factory=list)
    inference_times: List[float] = field(default_factory=list)
    accuracies_per_sample: List[int] = field(
        default_factory=list
    )  # 0 or 1 for each sample
    batch_accuracies: List[float] = field(default_factory=list)  # Accuracy per batch

    def update_from_batch(self, batch_result: BatchResult):
        """Update collector with data from a batch result"""
        self.count += batch_result.batch_size
        self.correct += np.sum(batch_result.is_correct)
        self.confidences.extend(batch_result.confidences.tolist())
        self.inference_times.append(batch_result.inference_time_ms)
        self.accuracies_per_sample.extend(batch_result.is_correct.astype(int).tolist())

        # Calculate batch accuracy
        batch_accuracy = 100 * np.sum(batch_result.is_correct) / batch_result.batch_size
        self.batch_accuracies.append(batch_accuracy)


@dataclass
class ClassCollector:
    """
    Collects and stores evaluation data for a specific class.
    """

    class_id: int
    count: int = 0
    correct: int = 0
    inference_times: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    exit_by_layer: Dict[str, int] = field(default_factory=dict)  # Exit key -> count
    accuracies: List[int] = field(default_factory=list)  # 0 or 1 for each sample
    predicted_as: Dict[int, int] = field(
        default_factory=dict
    )  # Predicted class -> count

    def update_from_sample(
        self,
        true_class: int,
        predicted_class: int,
        is_correct: bool,
        confidence: float,
        inference_time: float,
        exit_key: str,
    ):
        """Update collector with data from a single sample"""
        assert true_class == self.class_id, "Class ID mismatch"

        self.count += 1
        self.correct += 1 if is_correct else 0
        self.inference_times.append(inference_time)
        self.confidences.append(confidence)
        self.accuracies.append(1 if is_correct else 0)

        # Update exit layer distribution
        if exit_key not in self.exit_by_layer:
            self.exit_by_layer[exit_key] = 0
        self.exit_by_layer[exit_key] += 1

        # Update confusion matrix data
        if predicted_class not in self.predicted_as:
            self.predicted_as[predicted_class] = 0
        self.predicted_as[predicted_class] += 1


@dataclass
class EvaluationResults:
    """
    Container for all collected evaluation data.
    """

    total_samples: int = 0
    total_correct: int = 0
    exit_collectors: Dict[str, ExitCollector] = field(default_factory=dict)
    class_collectors: Dict[int, ClassCollector] = field(default_factory=dict)
    num_classes: int = 0

    def get_or_create_exit_collector(self, exit_key: str) -> ExitCollector:
        """Get existing collector or create a new one if it doesn't exist"""
        if exit_key not in self.exit_collectors:
            self.exit_collectors[exit_key] = ExitCollector(exit_key=exit_key)
        return self.exit_collectors[exit_key]

    def get_or_create_class_collector(self, class_id: int) -> ClassCollector:
        """Get existing collector or create a new one if it doesn't exist"""
        if class_id not in self.class_collectors:
            self.class_collectors[class_id] = ClassCollector(class_id=class_id)
        return self.class_collectors[class_id]

    @property
    def current_accuracy_avg(self):
        return (
            100 * self.total_correct / self.total_samples
            if self.total_samples > 0
            else 0
        )

    @property
    def current_inference_avg_ms(self):
        if self.total_samples == 0:
            return 0

        # Calculate weighted average in one step
        weighted_sum = sum(
            np.mean(c.inference_times) * c.count
            for c in self.exit_collectors.values()
            if c.inference_times
        )

        return weighted_sum / self.total_samples
