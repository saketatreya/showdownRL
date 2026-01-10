#! /usr/bin/env python3
"""
Train the Win Predictor model (Robust & Memory Efficient).
Uses Chunked Processing to handle 100k+ replays without OOM.
"""
import glob
import json
import logging
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

from src.replay_parser import ReplayParser
from src.win_predictor import WinPredictor

# Configuration
REPLAY_DIR = "data/replays"
CHECKPOINT_DIR = "data/checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/checkpoint_latest.pt"
BEST_MODEL_PATH = "models/win_predictor.pt"
METRICS_PATH = "data/training_metrics.json"
CACHE_DIR = "data/cache_chunks" # Changes to valid dir name
BATCH_SIZE = 512 # Increased batch size for efficiency
EPOCHS = 20 # 20 epochs on full dataset is plenty
LEARNING_RATE = 1e-4
CHUNK_SIZE = 5000 # Process 5000 replays per chunk (~250k states)

# State sampling filters to avoid learning trivial patterns
MIN_GAME_LENGTH = 10  # Skip forfeits/disconnects (< 10 turns)
GAME_PERCENT_CUTOFF = 0.60  # Only use first 60% of each game's states

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/train_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_metrics(status, **kwargs):
    """Write metrics to JSON for the monitoring script."""
    metrics = {
        "status": status,
        "timestamp": time.time(),
        **kwargs
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

def save_chunk(chunk_id, features, labels):
    """Save a chunk of data to disk."""
    if not features:
        return
    
    path = f"{CACHE_DIR}/chunk_{chunk_id}.pt"
    # Compact saving
    torch.save({
        'features': torch.tensor(np.array(features, dtype=np.float32)),
        'labels': torch.tensor(np.array(labels, dtype=np.float32)).unsqueeze(1)
    }, path)
    logger.info(f"Saved chunk {chunk_id} to {path} ({len(features)} samples)")

class DataProcessor:
    """Handles the parsing of replays into cached chunks."""
    def __init__(self, replay_files, parser):
        self.replay_files = replay_files
        self.parser = parser
        
    def run(self):
        chunks = glob.glob(f"{CACHE_DIR}/chunk_*.pt")
        if len(chunks) > 0:
            logger.info(f"Found {len(chunks)} existing chunks in {CACHE_DIR}. Skipping processing.")
            return chunks

        logger.info(f"Processing {len(self.replay_files)} replays into chunks...")
        update_metrics("Parsing", progress=0, total=len(self.replay_files))
        
        current_features = []
        current_labels = []
        chunk_counter = 0
        processed_count = 0
        
        for i, file_path in enumerate(tqdm(self.replay_files, desc="Parsing Replays")):
            if i % 1000 == 0:
                update_metrics("Parsing", progress=i, total=len(self.replay_files))
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                states, did_p1_win = self.parser.parse_replay_raw(data)
                
                if not states:
                    continue
                
                # Get max turn in this game
                max_turn = max(s.turn for s in states) if states else 0
                
                # Skip short games (forfeits/disconnects)
                if max_turn < MIN_GAME_LENGTH:
                    continue
                
                # Calculate turn cutoff (first 60% of game)
                turn_cutoff = int(max_turn * GAME_PERCENT_CUTOFF)
                
                for state in states:
                    # Skip late-game states
                    if state.turn > turn_cutoff:
                        continue
                    
                    # P1 Perspective
                    f1 = self.parser._state_to_features(state, perspective=1)
                    current_features.append(f1)
                    current_labels.append(1.0 if did_p1_win else 0.0)
                    
                    # P2 Perspective (Symmetry)
                    f2 = self.parser._state_to_features(state, perspective=2)
                    current_features.append(f2)
                    current_labels.append(0.0 if did_p1_win else 1.0)
                
                processed_count += 1
                
                # Check chunk size
                if processed_count >= CHUNK_SIZE:
                    save_chunk(chunk_counter, current_features, current_labels)
                    chunk_counter += 1
                    current_features = []
                    current_labels = []
                    processed_count = 0
                    
            except Exception as e:
                # logger.debug(f"Error {file_path}: {e}")
                pass
        
        # Save remainder
        if current_features:
            save_chunk(chunk_counter, current_features, current_labels)
            
        return glob.glob(f"{CACHE_DIR}/chunk_*.pt")

class ChunkedDataset(IterableDataset):
    """
    Streams data from disk chunks. 
    Loads one chunk at a time into memory, shuffles it, and yields samples.
    """
    def __init__(self, chunk_paths: List[str], shuffle_chunks=True):
        self.chunk_paths = chunk_paths
        self.shuffle_chunks = shuffle_chunks
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        # If multiple workers, split chunks among them
        if worker_info is not None:
             per_worker = int(np.ceil(len(self.chunk_paths) / float(worker_info.num_workers)))
             worker_id = worker_info.id
             iter_start = worker_id * per_worker
             iter_end = min(iter_start + per_worker, len(self.chunk_paths))
             my_chunks = self.chunk_paths[iter_start:iter_end]
        else:
             my_chunks = self.chunk_paths
        
        if self.shuffle_chunks:
            random.shuffle(my_chunks)
            
        for path in my_chunks:
            try:
                data = torch.load(path)
                features = data['features']
                labels = data['labels']
                
                # Shuffle within chunk
                indices = torch.randperm(len(features))
                features = features[indices]
                labels = labels[indices]
                
                for i in range(len(features)):
                    yield features[i], labels[i]
                    
            except Exception as e:
                logger.error(f"Failed to load chunk {path}: {e}")

def train_model(model, train_loader, val_loader, start_epoch, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    
    best_val_loss = float('inf')
    
    # Resume Checkpoint
    if start_epoch > 0 and os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Restoring optimizer state needed...")
        try:
           checkpoint = torch.load(CHECKPOINT_PATH)
           optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
           scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
           best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        except:
           logger.warning("Could not load optimizer state, starting fresh optimizer.")
    
    logger.info(f"Starting training from epoch {start_epoch+1}...")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        batches = 0
        
        update_metrics("Training", epoch=epoch+1, total_epochs=EPOCHS, train_loss=0, val_loss=0)
        
        # Training Loop
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for X, y in progress_bar:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # fast accuracy
            preds = (outputs > 0).float()
            acc = (preds == y).float().mean()
            train_acc += acc.item()
            batches += 1
            
            # Update bar less frequently to save CPU
            if batches % 10 == 0:
                 current_loss = train_loss / batches
                 current_acc = train_acc / batches
                 progress_bar.set_postfix({'loss': current_loss, 'acc': current_acc})
                 
            # Update external metrics every 100 batches for live monitoring
            if batches % 100 == 0:
                update_metrics(
                    "Training", 
                    epoch=epoch+1, 
                    total_epochs=EPOCHS, 
                    train_loss=train_loss / batches, 
                    val_loss=0.0, # Not computed yet
                    val_acc=0.0
                )
        
        avg_train_loss = train_loss / max(1, batches)
        avg_train_acc = train_acc / max(1, batches)
        
        # Validation Loop
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': min(val_loss, best_val_loss),
            'val_loss': val_loss
        }, CHECKPOINT_PATH)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            logger.info(f"Saved best model to {BEST_MODEL_PATH}")
            
        update_metrics(
            "Training", 
            epoch=epoch+1, 
            total_epochs=EPOCHS, 
            train_loss=avg_train_loss, 
            val_loss=val_loss,
            val_acc=val_acc
        )

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            
            total_loss += loss.item()
            preds = (outputs > 0).float()
            acc = (preds == y).float().mean()
            total_acc += acc.item()
            batches += 1
            
    if batches == 0: return 0.0, 0.0
    return total_loss / batches, total_acc / batches

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Collect and Process Data (Chunked)
    replay_files = glob.glob(f"{REPLAY_DIR}/**/*.json", recursive=True)
    replay_files = [Path(p) for p in replay_files if "progress.json" not in p.lower()]
    
    if not replay_files:
        logger.error("No replays found!")
        return

    parser = ReplayParser()
    processor = DataProcessor(replay_files, parser)
    chunk_paths = processor.run()
    
    if not chunk_paths:
         logger.error("No data chunks created!")
         return
         
    # 2. Split Chunks into Train/Val
    random.seed(42)
    random.shuffle(chunk_paths)
    split_idx = int(len(chunk_paths) * 0.9)
    train_chunks = chunk_paths[:split_idx]
    val_chunks = chunk_paths[split_idx:]
    
    # 3. Create iterable datasets
    train_ds = ChunkedDataset(train_chunks, shuffle_chunks=True)
    val_ds = ChunkedDataset(val_chunks, shuffle_chunks=False)
    
    # num_workers > 0 works with IterableDataset to load chunks in parallel backgrounds
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=1)
    
    # 4. Initialize Model
    model = WinPredictor(input_dim=parser.FEATURE_DIM).to(device)
    start_epoch = 0
    
    if os.path.exists(CHECKPOINT_PATH):
        try:
            checkpoint = torch.load(CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            logger.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to resume: {e}")
            
    # 5. Train
    if start_epoch < EPOCHS:
        train_model(model, train_loader, val_loader, start_epoch, device)
    else:
        logger.info("Training complete.")

if __name__ == "__main__":
    main()
