"""
Model calibration for probability estimation.
"""
from typing import Optional, Literal
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)


class ModelCalibrator:
    """
    Calibrate model scores to probabilities.
    """
    
    def __init__(self, method: Literal['isotonic', 'platt'] = 'isotonic'):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('isotonic' or 'platt')
        """
        self.method = method
        self.calibrator = None
        self.fitted = False
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        Fit calibrator on validation data.
        
        Args:
            scores: Model scores
            labels: Binary labels (1 for win, 0 for loss)
        """
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'platt':
            self.calibrator = LogisticRegression()
            scores = scores.reshape(-1, 1)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.calibrator.fit(scores, labels)
        self.fitted = True
        
        logger.info(f"Calibrator fitted using {self.method} method")
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Transform scores to calibrated probabilities.
        
        Args:
            scores: Model scores
        
        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        if self.method == 'platt':
            scores = scores.reshape(-1, 1)
            probs = self.calibrator.predict_proba(scores)[:, 1]
        else:
            probs = self.calibrator.transform(scores)
        
        return probs
    
    def fit_transform(self, scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(scores, labels)
        return self.transform(scores)


def calculate_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate calibration curve.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins
    
    Returns:
        Tuple of (bin_edges, bin_true_prob, bin_pred_prob)
    """
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Calculate statistics for each bin
    bin_true_prob = []
    bin_pred_prob = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_true_prob.append(y_true[mask].mean())
            bin_pred_prob.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_true_prob.append(np.nan)
            bin_pred_prob.append(np.nan)
            bin_counts.append(0)
    
    return bins, np.array(bin_true_prob), np.array(bin_pred_prob), np.array(bin_counts)


def temperature_scaling(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Apply temperature scaling to scores before softmax.
    
    Args:
        scores: Model scores
        temperature: Temperature parameter (>1 = smoother, <1 = sharper)
    
    Returns:
        Scaled scores
    """
    return scores / temperature


def find_optimal_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    temperature_range: tuple = (0.1, 5.0),
    n_steps: int = 50
) -> float:
    """
    Find optimal temperature for calibration.
    
    Args:
        scores: Model scores
        labels: Binary labels
        temperature_range: Range of temperatures to search
        n_steps: Number of steps in search
    
    Returns:
        Optimal temperature
    """
    from sklearn.metrics import log_loss
    
    temperatures = np.linspace(temperature_range[0], temperature_range[1], n_steps)
    best_temp = 1.0
    best_loss = float('inf')
    
    for temp in temperatures:
        scaled_scores = temperature_scaling(scores, temp)
        probs = 1 / (1 + np.exp(-scaled_scores))  # Sigmoid
        
        try:
            loss = log_loss(labels, probs)
            if loss < best_loss:
                best_loss = loss
                best_temp = temp
        except:
            continue
    
    logger.info(f"Optimal temperature: {best_temp:.3f} (loss: {best_loss:.4f})")
    
    return best_temp
