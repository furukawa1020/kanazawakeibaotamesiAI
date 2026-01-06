"""
GPU-accelerated Monte Carlo sampling for Plackett-Luce model.
Optimized for RTX 5060.
"""
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class PlackettLuceSampler:
    """
    GPU-accelerated Monte Carlo sampler for Plackett-Luce model.
    Uses PyTorch with CUDA for maximum performance on RTX 5060.
    """
    
    def __init__(
        self,
        n_samples: int = 50000,
        use_gpu: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize sampler.
        
        Args:
            n_samples: Number of Monte Carlo samples
            use_gpu: Whether to use GPU acceleration
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.random_seed = random_seed
        
        if self.use_gpu:
            self.device = torch.device('cuda')
            logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU for sampling")
        
        # Set random seed
        if random_seed is not None:
            torch.manual_seed(random_seed)
            if self.use_gpu:
                torch.cuda.manual_seed(random_seed)
    
    def sample_top3(self, weights: np.ndarray) -> Dict[Tuple[int, int, int], float]:
        """
        Sample top-3 finishes using Plackett-Luce model.
        
        Args:
            weights: Array of weights for each horse (higher = stronger)
        
        Returns:
            Dictionary mapping (1st, 2nd, 3rd) tuples to probabilities
        """
        n_horses = len(weights)
        
        if n_horses < 3:
            raise ValueError("Need at least 3 horses to sample top-3")
        
        # Convert to torch tensor and move to GPU
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        
        # Ensure weights are positive
        weights_tensor = torch.clamp(weights_tensor, min=1e-10)
        
        # Sample rankings
        logger.debug(f"Sampling {self.n_samples} rankings for {n_horses} horses on {self.device}")
        
        rankings = self._sample_rankings_gpu(weights_tensor, self.n_samples)
        
        # Count top-3 combinations
        top3_counts = {}
        
        # Extract top 3 from each ranking
        top3_rankings = rankings[:, :3].cpu().numpy()
        
        for ranking in top3_rankings:
            key = tuple(ranking)
            top3_counts[key] = top3_counts.get(key, 0) + 1
        
        # Convert counts to probabilities
        top3_probs = {
            combo: count / self.n_samples
            for combo, count in top3_counts.items()
        }
        
        logger.debug(f"Generated {len(top3_probs)} unique top-3 combinations")
        
        return top3_probs
    
    def _sample_rankings_gpu(
        self,
        weights: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """
        Sample complete rankings using Plackett-Luce on GPU.
        
        This is the core GPU-accelerated function that maximizes RTX 5060 usage.
        
        Args:
            weights: Tensor of weights [n_horses]
            n_samples: Number of samples
        
        Returns:
            Tensor of rankings [n_samples, n_horses]
        """
        n_horses = len(weights)
        
        # Expand weights for batch processing
        # Shape: [n_samples, n_horses]
        weights_batch = weights.unsqueeze(0).expand(n_samples, -1).clone()
        
        # Initialize rankings tensor
        rankings = torch.zeros(n_samples, n_horses, dtype=torch.long, device=self.device)
        
        # Sample position by position using Plackett-Luce
        remaining_weights = weights_batch.clone()
        
        for pos in range(n_horses):
            # Calculate probabilities (normalize remaining weights)
            probs = remaining_weights / remaining_weights.sum(dim=1, keepdim=True)
            
            # Sample from categorical distribution
            # Use Gumbel-max trick for efficient GPU sampling
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + 1e-10) + 1e-10)
            log_probs = torch.log(probs + 1e-10)
            
            # Get argmax of log_probs + gumbel_noise
            selected = torch.argmax(log_probs + gumbel_noise, dim=1)
            
            # Store selected horse
            rankings[:, pos] = selected
            
            # Zero out selected horses' weights for next iteration
            remaining_weights[torch.arange(n_samples), selected] = 0
        
        return rankings
    
    def estimate_trifecta_probabilities(
        self,
        weights: np.ndarray,
        top_k: Optional[int] = None
    ) -> Dict[Tuple[int, int, int], float]:
        """
        Estimate trifecta probabilities, optionally limiting to top-K horses.
        
        Args:
            weights: Array of weights for each horse
            top_k: If specified, only consider top-K horses by weight
        
        Returns:
            Dictionary of trifecta probabilities
        """
        if top_k is not None and top_k < len(weights):
            # Get indices of top-K horses
            top_k_indices = np.argsort(weights)[-top_k:][::-1]
            
            # Create mapping from sampled indices to original indices
            index_map = {i: orig_idx for i, orig_idx in enumerate(top_k_indices)}
            
            # Sample with top-K weights
            top_k_weights = weights[top_k_indices]
            sampled_probs = self.sample_top3(top_k_weights)
            
            # Remap indices back to original
            remapped_probs = {
                (index_map[a], index_map[b], index_map[c]): prob
                for (a, b, c), prob in sampled_probs.items()
            }
            
            return remapped_probs
        else:
            return self.sample_top3(weights)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage statistics."""
        if not self.use_gpu:
            return {'gpu_available': False}
        
        return {
            'gpu_available': True,
            'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2
        }


def analytical_plackett_luce_top3(weights: np.ndarray) -> Dict[Tuple[int, int, int], float]:
    """
    Calculate exact Plackett-Luce probabilities for top-3 (small field only).
    
    This is computationally expensive for large fields, use sampling instead.
    
    Args:
        weights: Array of weights for each horse
    
    Returns:
        Dictionary of exact trifecta probabilities
    """
    n = len(weights)
    
    if n > 10:
        logger.warning(f"Analytical calculation with {n} horses is slow. Consider using sampling.")
    
    probs = {}
    
    # Iterate over all possible top-3 combinations
    for i in range(n):
        for j in range(n):
            if j == i:
                continue
            for k in range(n):
                if k == i or k == j:
                    continue
                
                # Calculate P(i=1st, j=2nd, k=3rd)
                # P(i=1st) = w_i / sum(w)
                p1 = weights[i] / weights.sum()
                
                # P(j=2nd | i=1st) = w_j / sum(w \ {i})
                remaining_after_i = weights.copy()
                remaining_after_i[i] = 0
                p2 = weights[j] / remaining_after_i.sum()
                
                # P(k=3rd | i=1st, j=2nd) = w_k / sum(w \ {i, j})
                remaining_after_ij = remaining_after_i.copy()
                remaining_after_ij[j] = 0
                p3 = weights[k] / remaining_after_ij.sum()
                
                probs[(i, j, k)] = p1 * p2 * p3
    
    return probs
