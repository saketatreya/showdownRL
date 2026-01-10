#!/usr/bin/env python3
"""
Win Predictor Model for Potential-Based Reward Shaping.

Φ(s) = P(win | state) trained on Pokemon Showdown ladder replays.
Used for reward shaping: R' = R_terminal + γ * Φ(s') - Φ(s)
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WinPredictor(nn.Module):
    """
    Neural network that predicts P(win | state).
    
    Architecture: MLP with dropout for regularization.
    Input: State features (128-dim from ReplayParser)
    Output: Probability of winning (0-1)
    """
    
    def __init__(
        self, 
        input_dim: int = 744,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.2
    ):
        """
        Args:
            input_dim: Size of input feature vector
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns LOGITS (unnormalized scores).
        Use torch.sigmoid(output) to get probabilities.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, 1)
        """
        return self.network(x)
    
    def predict(self, state: np.ndarray) -> float:
        """
        Predict win probability for a single state.
        
        Args:
            state: Feature vector of shape (input_dim,)
            
        Returns:
            Win probability (0-1)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            logits = self(x)
            prob = torch.sigmoid(logits).item()
        return prob
    
    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Predict win probabilities for a batch of states.
        
        Args:
            states: Feature vectors of shape (batch_size, input_dim)
            
        Returns:
            Win probabilities of shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(states)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            logits = self(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims
        }, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'WinPredictor':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            input_dim=checkpoint['input_dim'],
            hidden_dims=checkpoint['hidden_dims']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded from {path}")
        return model


class WinPredictorDataset(torch.utils.data.Dataset):
    """Dataset for training the win predictor."""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: State features of shape (n_samples, feature_dim)
            labels: Win labels of shape (n_samples,), 1=win, 0=loss
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class WinPredictorTrainer:
    """Trainer for the win predictor model."""
    
    def __init__(
        self,
        model: WinPredictor,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.BCELoss()
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        # Metrics history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.model(features)
            loss = self.criterion(predictions, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * len(features)
        
        return total_loss / len(dataloader.dataset)
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item() * len(features)
                
                # Accuracy
                predicted_labels = (predictions > 0.5).float()
                correct += (predicted_labels == labels).sum().item()
                total += len(labels)
        
        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 20,
        early_stopping_patience: int = 5,
        save_path: Optional[str] = None
    ) -> dict:
        """
        Full training loop with early stopping.
        
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Val Acc={val_acc:.4f}"
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if save_path:
                    self.model.save(save_path)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss,
            'final_val_accuracy': self.val_accuracies[-1]
        }


def prepare_data(
    replay_dir: str,
    val_split: float = 0.1,
    max_replays: Optional[int] = None
) -> Tuple[WinPredictorDataset, WinPredictorDataset]:
    """
    Prepare datasets from parsed replays.
    
    Args:
        replay_dir: Directory containing replay JSON files
        val_split: Fraction of replays for validation
        max_replays: Maximum number of replays to use
        
    Returns:
        (train_dataset, val_dataset)
    """
    import json
    from src.replay_parser import ReplayParser
    
    replay_path = Path(replay_dir)
    parser = ReplayParser()
    
    all_features = []
    all_labels = []
    
    replay_files = list(replay_path.rglob("*.json"))
    replay_files = [f for f in replay_files if f.name != "progress.json"]
    
    if max_replays:
        replay_files = replay_files[:max_replays]
    
    logger.info(f"Processing {len(replay_files)} replays...")
    
    for replay_file in replay_files:
        try:
            with open(replay_file, 'r') as f:
                replay = json.load(f)
            
            states = parser.parse_replay(replay)
            
            for state in states:
                all_features.append(state.features)
                all_labels.append(1.0 if state.did_p1_win else 0.0)
                
        except Exception as e:
            logger.debug(f"Error processing {replay_file}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No valid states extracted from replays")
    
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    logger.info(f"Total samples: {len(features)}")
    logger.info(f"Win rate in data: {labels.mean():.3f}")
    
    # Split by replay (not by state) to avoid data leakage
    n_total = len(features)
    n_val = int(n_total * val_split)
    
    # Shuffle
    indices = np.random.permutation(n_total)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_dataset = WinPredictorDataset(features[train_indices], labels[train_indices])
    val_dataset = WinPredictorDataset(features[val_indices], labels[val_indices])
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train win predictor model")
    parser.add_argument("--replay-dir", type=str, default="data/replays",
                        help="Directory containing replays")
    parser.add_argument("--output", type=str, default="data/win_predictor.pt",
                        help="Output model path")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--max-replays", type=int, default=None,
                        help="Max replays to use")
    
    args = parser.parse_args()
    
    # Prepare data
    train_dataset, val_dataset = prepare_data(
        args.replay_dir,
        max_replays=args.max_replays
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = WinPredictor(input_dim=128)
    
    # Train
    trainer = WinPredictorTrainer(model, learning_rate=args.lr, device=device)
    history = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        save_path=args.output
    )
    
    print(f"\nTraining complete!")
    print(f"Best val loss: {history['best_val_loss']:.4f}")
    print(f"Final val accuracy: {history['final_val_accuracy']:.4f}")
