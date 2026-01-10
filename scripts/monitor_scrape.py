#!/usr/bin/env python3
"""
Live monitoring script for replay scraper.
Shows real-time progress with a nice dashboard.

Usage:
    python scripts/monitor_scrape.py
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def count_replays(replay_dir: Path) -> dict:
    """Count replays on disk by date."""
    counts = {}
    total = 0
    
    for date_dir in replay_dir.iterdir():
        if date_dir.is_dir():
            count = len([f for f in date_dir.glob("*.json")])
            if count > 0:
                counts[date_dir.name] = count
                total += count
    
    return {"by_date": counts, "total": total}


def get_progress(replay_dir: Path) -> dict:
    """Load progress.json and compute stats."""
    progress_file = replay_dir / "progress.json"
    
    if not progress_file.exists():
        return None
    
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
        
        data['scraped_count'] = len(data.get('scraped', []))
        data['failed_count'] = len(data.get('failed', []))
        
        return data
    except Exception as e:
        return None


def format_time(seconds: float) -> str:
    """Format seconds to human readable."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_log_tail(log_file: Path, lines: int = 5) -> list:
    """Get last N lines from log file."""
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            return [l.strip() for l in all_lines[-lines:] if l.strip()]
    except:
        return []


def main():
    replay_dir = Path("data/replays")
    log_file = Path("data/scrape_log.txt")
    target_count = 100000
    
    start_time = None
    last_count = 0
    
    print("🔍 Monitoring replay scrape... (Ctrl+C to exit)")
    print()
    
    while True:
        try:
            clear_screen()
            
            progress = get_progress(replay_dir)
            disk_stats = count_replays(replay_dir)
            
            print("╔════════════════════════════════════════════════════════╗")
            print("║          🎮 POKEMON SHOWDOWN REPLAY SCRAPER            ║")
            print("╠════════════════════════════════════════════════════════╣")
            
            if progress:
                scraped = progress.get('scraped_count', 0)
                failed = progress.get('failed_count', 0)
                skipped = progress.get('skipped_low_rating', 0)
                min_rating = progress.get('min_rating', 0)
                last_update = progress.get('last_update', 'Unknown')
                
                pending = len(progress.get('pending', []))
                
                # Calculate rate
                if start_time is None:
                    start_time = time.time()
                    last_count = scraped
                
                elapsed = time.time() - start_time
                current_rate = (scraped - last_count) / max(elapsed, 1)
                
                if current_rate > 0:
                    remaining = target_count - scraped
                    eta_seconds = remaining / current_rate
                    eta = format_time(eta_seconds)
                else:
                    eta = "Calculating..."
                
                pct = (scraped / target_count) * 100
                bar_width = 40
                filled = int(bar_width * pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                
                print(f"║  Progress: [{bar}] {pct:.1f}%")
                print(f"║")
                print(f"║  📥 Downloaded: {scraped:,} / {target_count:,}")
                print(f"║  ⏳ Queued:     {pending:,}")
                print(f"║  ❌ Failed:     {failed:,}")
                if min_rating > 0:
                    print(f"║  ⏭️  Skipped:    {skipped:,} (rating < {min_rating})")
                print(f"║")
                print(f"║  ⚡ Rate:     {current_rate:.1f} replays/sec")
                print(f"║  ⏰ ETA:      {eta}")
                print(f"║  🕐 Updated:  {last_update}")
            else:
                print("║  ⏳ Waiting for scraper to start...")
                print("║")
                print("║  Files on disk: ", disk_stats['total'])
            
            print("╠════════════════════════════════════════════════════════╣")
            print("║  📋 Recent Log Entries:                                ║")
            print("╠════════════════════════════════════════════════════════╣")
            
            log_lines = get_log_tail(log_file, 5)
            for line in log_lines:
                # Truncate long lines
                if len(line) > 54:
                    line = line[:51] + "..."
                print(f"║  {line:<54} ║")
            
            if not log_lines:
                print("║  (No log entries yet)                                  ║")
            
            print("╠════════════════════════════════════════════════════════╣")
            
            if disk_stats['by_date']:
                print("║  📅 Replays by Date:                                   ║")
                sorted_dates = sorted(disk_stats['by_date'].keys(), reverse=True)[:5]
                for date in sorted_dates:
                    count = disk_stats['by_date'][date]
                    print(f"║    {date}: {count:,} replays")
            
            print("╚════════════════════════════════════════════════════════╝")
            print()
            print("Press Ctrl+C to exit")
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
