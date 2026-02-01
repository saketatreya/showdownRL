#!/usr/bin/env python3
"""
Live Training Monitor for Win Predictor.
Reads data/training_metrics.json and displays a TUI.
"""
import json
import time
import os
import sys
from datetime import datetime

METRICS_PATH = "data/training_metrics.json"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_metrics():
    try:
        with open(METRICS_PATH, 'r') as f:
            return json.load(f)
    except:
        return None

def format_time(seconds):
    if not seconds: return "Unknown"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def main():
    print("Waiting for training metrics...")
    
    start_time = None
    
    while True:
        metrics = load_metrics()
        
        if metrics:
            clear_screen()
            status = metrics.get('status', 'Unknown')
            
            print(f"╔{'═'*50}╗")
            print(f"║ Win Predictor Training Monitor                   ║")
            print(f"╠{'═'*50}╣")
            print(f"║ Status: {status:<41}║")
            
            if status == "Parsing":
                prog = metrics.get('progress', 0)
                total = metrics.get('total', 0)
                pct = (prog / total * 100) if total > 0 else 0
                bar_len = 30
                filled = int(bar_len * pct / 100)
                bar = '█' * filled + '░' * (bar_len - filled)
                print(f"║ Progress: [{bar}] {pct:>5.1f}% ║")
                print(f"║ Files: {prog}/{total}                           ║")
                
            elif status == "Training":
                epoch = metrics.get('epoch', 0)
                total_epochs = metrics.get('total_epochs', 0)
                train_loss = metrics.get('train_loss', 0)
                val_loss = metrics.get('val_loss', 0)
                val_acc = metrics.get('val_acc', 0)
                
                print(f"║ Epoch: {epoch}/{total_epochs}                              ║")
                print(f"║ Train Loss: {train_loss:.4f}                          ║")
                print(f"║ Val Loss:   {val_loss:.4f}                          ║")
                print(f"║ Val Acc:    {val_acc:.4f} ({val_acc*100:.1f}%)               ║")
                
                # Simple visual history could go here if we tracked it in metrics
            
            elif status == "Completed":
                print(f"║ Training Completed Successfully!                 ║")
            
            elif status == "Error":
                msg = metrics.get('message', 'Unknown Error')
                print(f"║ Error: {msg:<42}║")

            print(f"╚{'═'*50}╝")
            print(f"\nLast Updated: {datetime.now().strftime('%H:%M:%S')}")
            print("Press Ctrl+C to exit monitor (Training continues)")
            
        time.sleep(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting monitor.")
