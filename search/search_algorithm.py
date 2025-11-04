"""
Search algorithms for noise search in diffusion models

Implements three search strategies:
1. RandomSearch: Random noise candidates
2. ZeroOrderSearch: Iterative optimization in noise space
3. PathSearch: Noise injection at intermediate denoising steps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Tuple, Dict, Any
from tqdm import tqdm


class RandomSearch:
    """
    Random Search: Generate N random noise candidates, denoise each,
    and select the one with highest verifier score.
    """
    def __init__(self, n_candidates: int = 4):
        """
        Initialize Random Search
        
        Args:
            n_candidates: Number of random noise candidates to try
        """
        self.n_candidates = n_candidates
        self.nfes = 0  # Track number of function evaluations
    
    def search(self, 
               noise_shape: Tuple[int, ...],
               denoise_fn: Callable,
               verifier_fn: Callable,
               device: str = 'cuda',
               verbose: bool = True,
               **kwargs) -> Tuple[torch.Tensor, float]:
        """
        Perform random search
        
        Args:
            noise_shape: Shape of the noise tensor (B, C, H, W)
            denoise_fn: Function that takes noise and returns denoised images
            verifier_fn: Function that scores denoised images
            device: Device to run on
            verbose: Whether to show progress bar
            **kwargs: Additional arguments for denoise_fn and verifier_fn
            
        Returns:
            Best noise candidate and its score
        """
        batch_size = noise_shape[0]
        best_noise = None
        best_score = float('-inf')
        
        candidates_scores = []
        
        # Create progress bar
        iterator = range(self.n_candidates)
        if verbose:
            iterator = tqdm(iterator, desc=f"Random Search ({self.n_candidates} candidates)", leave=False)
        
        for i in iterator:
            # Generate random noise
            noise = torch.randn(noise_shape, device=device)
            
            # Denoise
            with torch.no_grad():
                denoised = denoise_fn(noise, show_progress=(i == 0), **kwargs)  # Show progress for first candidate
                self.nfes += 1
            
            # Score
            score = verifier_fn(denoised, **kwargs)
            candidates_scores.append(score)
            
            # Update best
            if score > best_score:
                best_score = score
                best_noise = noise.clone()
        
        return best_noise, best_score
    
    def reset_nfes(self):
        """Reset NFE counter"""
        self.nfes = 0


class ZeroOrderSearch:
    """
    Zero-Order Search: Iterative optimization in noise space
    
    Explores the neighborhood of a pivot noise vector and iteratively
    refines it based on verifier scores.
    """
    def __init__(self, 
                 n_neighbors: int = 4,
                 lambda_radius: float = 0.95,
                 n_iterations: int = 10,
                 verbose: bool = False):
        """
        Initialize Zero-Order Search
        
        Args:
            n_neighbors: Number of neighbors to sample at each iteration
            lambda_radius: Radius of the neighborhood (0-1)
            n_iterations: Number of search iterations
            verbose: Whether to print progress
        """
        self.n_neighbors = n_neighbors
        self.lambda_radius = lambda_radius
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.nfes = 0
    
    def search(self,
               initial_noise: torch.Tensor,
               denoise_fn: Callable,
               verifier_fn: Callable,
               device: str = 'cuda',
               verbose: Optional[bool] = None,
               **kwargs) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """
        Perform zero-order search
        
        Args:
            initial_noise: Initial noise vector (B, C, H, W)
            denoise_fn: Function that denoises noise
            verifier_fn: Function that scores denoised images
            device: Device to run on
            verbose: Whether to show progress (overrides instance setting)
            **kwargs: Additional arguments
            
        Returns:
            Best noise, best score, and search history
        """
        # Use provided verbose or instance setting
        if verbose is None:
            verbose = self.verbose
        
        current_noise = initial_noise.clone()
        best_noise = initial_noise.clone()
        best_score = float('-inf')
        
        history = {
            'scores': [],
            'candidates_per_iter': []
        }
        
        # Create outer progress bar for iterations
        iterator = range(self.n_iterations)
        if verbose:
            iterator = tqdm(iterator, desc=f"Zero-Order Search", total=self.n_iterations, leave=True)
        
        for iteration in iterator:
            # Sample neighbors around current noise
            neighbors = self._sample_neighbors(current_noise, device)
            
            iteration_scores = []
            best_candidate = None
            best_candidate_score = float('-inf')
            
            # Create inner progress bar for neighbors
            neighbor_iterator = neighbors
            if verbose:
                neighbor_iterator = tqdm(neighbors, desc=f"  Iter {iteration+1}/{self.n_iterations}", leave=False, total=len(neighbors))
            
            for neighbor_idx, neighbor in enumerate(neighbor_iterator):
                # Denoise neighbor
                with torch.no_grad():
                    # Show denoising progress for each neighbor
                    denoised = denoise_fn(neighbor, show_progress=(neighbor_idx == 0), **kwargs)
                    self.nfes += 1
                
                # Score
                score = verifier_fn(denoised, **kwargs)
                iteration_scores.append(score)
                
                # Track best candidate in this iteration
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = neighbor.clone()
                
                # Update neighbor progress bar with score
                if verbose:
                    neighbor_iterator.set_postfix({'score': f'{score:.4f}'})
            
            history['scores'].append(iteration_scores)
            history['candidates_per_iter'].append(len(neighbors))
            
            # Update pivot if we found a better candidate
            if best_candidate_score > best_score:
                best_score = best_candidate_score
                best_noise = best_candidate.clone()
                current_noise = best_candidate.clone()
            
            # Update progress bar with current best score
            if verbose:
                iterator.set_postfix({
                    'best_score': f'{best_score:.4f}',
                    'iter_best': f'{best_candidate_score:.4f}'
                })
                print(f"\n  Iteration {iteration+1}/{self.n_iterations} complete!")
                print(f"    Best this iteration: {best_candidate_score:.4f}")
                print(f"    Overall best: {best_score:.4f}")
        
        return best_noise, best_score, history
    
    def _sample_neighbors(self, pivot: torch.Tensor, device: str) -> list:
        """
        Sample neighbors around a pivot noise vector
        
        Args:
            pivot: Center noise vector (B, C, H, W)
            device: Device to create tensors on
            
        Returns:
            List of neighbor noise tensors
        """
        neighbors = []
        
        for _ in range(self.n_neighbors):
            # Generate random perturbation
            perturbation = torch.randn_like(pivot) * (1 - self.lambda_radius)
            
            # Add to pivot within radius
            neighbor = pivot + perturbation
            neighbors.append(neighbor)
        
        return neighbors
    
    def reset_nfes(self):
        """Reset NFE counter"""
        self.nfes = 0


class PathSearch:
    """
    Path Search: Inject noise at intermediate denoising steps
    
    Explores multiple paths by injecting noise at different points in
    the denoising trajectory.
    """
    def __init__(self,
                 n_paths: int = 4,
                 injection_step: int = 400,  # Inject noise at step 400 (out of 1000)
                 noise_scale: float = 0.1,
                 verbose: bool = False):
        """
        Initialize Path Search
        
        Args:
            n_paths: Number of paths to explore
            injection_step: Timestep at which to inject noise
            noise_scale: Scale of injected noise
            verbose: Whether to print progress
        """
        self.n_paths = n_paths
        self.injection_step = injection_step
        self.noise_scale = noise_scale
        self.verbose = verbose
        self.nfes = 0
    
    def search(self,
               initial_noise: torch.Tensor,
               denoise_fn: Callable,
               verifier_fn: Callable,
               timesteps: int = 1000,
               device: str = 'cuda',
               verbose: Optional[bool] = None,
               **kwargs) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """
        Perform path search
        
        Args:
            initial_noise: Initial noise (B, C, H, W)
            denoise_fn: Function that denoises (should support intermediate injection)
            verifier_fn: Function that scores images
            timesteps: Total number of denoising steps
            device: Device to run on
            verbose: Whether to show progress (overrides instance setting)
            **kwargs: Additional arguments
            
        Returns:
            Best noise, best score, and search history
        """
        # Use provided verbose or instance setting
        if verbose is None:
            verbose = self.verbose
        
        best_noise = initial_noise.clone()
        best_score = float('-inf')
        
        history = {
            'scores': [],
            'injection_points': []
        }
        
        # Create progress bar
        iterator = range(self.n_paths)
        if verbose:
            iterator = tqdm(iterator, desc=f"Path Search", total=self.n_paths, leave=True)
        
        for path_idx in iterator:
            # Create modified denoise function with noise injection
            def denoise_with_injection(noise, injection_step=self.injection_step):
                # This is a placeholder - actual implementation would require
                # tracking the denoising trajectory
                result = denoise_fn(noise, show_progress=(path_idx == 0), **kwargs)
                self.nfes += 1
                return result
            
            # Generate variation by adding noise to initial noise
            noise_variation = torch.randn_like(initial_noise) * self.noise_scale
            perturbed_noise = initial_noise + noise_variation
            
            # Denoise with injection
            with torch.no_grad():
                denoised = denoise_with_injection(perturbed_noise)
            
            # Score
            score = verifier_fn(denoised, **kwargs)
            history['scores'].append(score)
            history['injection_points'].append(self.injection_step)
            
            # Update best
            if score > best_score:
                best_score = score
                best_noise = perturbed_noise.clone()
            
            # Update progress bar
            if verbose:
                iterator.set_postfix({'best_score': f'{best_score:.4f}'})
        
        return best_noise, best_score, history
    
    def reset_nfes(self):
        """Reset NFE counter"""
        self.nfes = 0


class GradientBasedSearch:
    """
    Gradient-Based Search: Use gradients through verifier to optimize noise
    
    This is an extension beyond the paper's zero-order methods.
    """
    def __init__(self,
                 n_iterations: int = 20,
                 lr: float = 0.01,
                 verbose: bool = False):
        """
        Initialize Gradient-Based Search
        
        Args:
            n_iterations: Number of optimization iterations
            lr: Learning rate for noise optimization
            verbose: Whether to print progress
        """
        self.n_iterations = n_iterations
        self.lr = lr
        self.verbose = verbose
        self.nfes = 0
    
    def search(self,
               initial_noise: torch.Tensor,
               denoise_fn: Callable,
               verifier_fn: Callable,
               device: str = 'cuda',
               **kwargs) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """
        Perform gradient-based search
        
        Args:
            initial_noise: Initial noise vector
            denoise_fn: Denoising function
            verifier_fn: Verifier function (must be differentiable)
            device: Device to run on
            **kwargs: Additional arguments
            
        Returns:
            Optimized noise, best score, and history
        """
        noise = initial_noise.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([noise], lr=self.lr)
        
        history = {
            'scores': [],
            'grad_norms': []
        }
        
        best_noise = noise.clone()
        best_score = float('-inf')
        
        for iteration in range(self.n_iterations):
            if self.verbose:
                print(f"Gradient-Based Search Iteration {iteration + 1}/{self.n_iterations}")
            
            optimizer.zero_grad()
            
            # Denoise
            denoised = denoise_fn(noise, **kwargs)
            self.nfes += 1
            
            # Score (assuming verifier is differentiable)
            # For verifiers that aren't, use surrogate differentiable metrics
            score = verifier_fn(denoised, **kwargs)
            
            # Maximize score = minimize negative score
            loss = -score
            
            # Backward pass
            loss.backward()
            
            # Track gradient norm
            grad_norm = noise.grad.norm().item()
            history['grad_norms'].append(grad_norm)
            
            # Update
            optimizer.step()
            
            # Track best
            current_score = score.item() if isinstance(score, torch.Tensor) else score
            history['scores'].append(current_score)
            
            if current_score > best_score:
                best_score = current_score
                best_noise = noise.clone().detach()
            
            if self.verbose:
                print(f"  Score: {current_score:.4f}, Grad Norm: {grad_norm:.4f}")
        
        return best_noise, best_score, history
    
    def reset_nfes(self):
        """Reset NFE counter"""
        self.nfes = 0

