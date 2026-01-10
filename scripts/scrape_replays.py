#!/usr/bin/env python3
"""
Replay Scraper for Pokemon Showdown.
Downloads replays from replay.pokemonshowdown.com for training the win predictor.

Usage:
    python scripts/scrape_replays.py --count 100000 --output data/replays
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import aiohttp
import aiofiles
from tqdm.asyncio import tqdm_asyncio


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ReplayMetadata:
    """Metadata for a single replay."""
    id: str
    format: str
    players: List[str]
    uploadtime: int
    rating: Optional[int] = None


@dataclass
class Replay:
    """Full replay data."""
    id: str
    format: str
    players: List[str]
    uploadtime: int
    rating: Optional[int]
    log: str
    winner: Optional[str]
    

class ReplayScraper:
    """Scrapes Gen9 Random Battle replays from Pokemon Showdown."""
    
    BASE_URL = "https://replay.pokemonshowdown.com"
    SEARCH_ENDPOINT = "/search.json"
    
    # Rate limiting
    REQUESTS_PER_SECOND = 5
    MAX_CONCURRENT = 10
    
    def __init__(self, format: str = "gen9randombattle", output_dir: str = "data/replays", min_rating: int = 0):
        self.format = format
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_rating = min_rating
        
        # Track progress
        self.scraped_ids: set = set()
        self.failed_ids: set = set()
        self.skipped_low_rating: int = 0
        self.pending_ids: list = []  # IDs collected but not yet downloaded
        self.last_before_timestamp: Optional[int] = None  # For pagination resume
        
        # Load existing progress
        self._load_progress()
    
    def _load_progress(self):
        """Load previously scraped replay IDs to resume."""
        progress_file = self.output_dir / "progress.json"
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                data = json.load(f)
                self.scraped_ids = set(data.get('scraped', []))
                self.failed_ids = set(data.get('failed', []))
                self.pending_ids = data.get('pending', [])
                self.skipped_low_rating = data.get('skipped_low_rating', 0)
                self.last_before_timestamp = data.get('last_before_timestamp')
            logger.info(f"Resumed: {len(self.scraped_ids)} scraped, {len(self.pending_ids)} pending, {len(self.failed_ids)} failed")
    
    def _save_progress(self, pending_ids: Optional[list] = None):
        """Save progress for resumption."""
        progress_file = self.output_dir / "progress.json"
        with open(progress_file, 'w') as f:
            json.dump({
                'scraped': list(self.scraped_ids),
                'failed': list(self.failed_ids),
                'pending': pending_ids if pending_ids is not None else self.pending_ids,
                'skipped_low_rating': self.skipped_low_rating,
                'min_rating': self.min_rating,
                'last_before_timestamp': self.last_before_timestamp,
                'last_update': datetime.now().isoformat()
            }, f)
    
    async def get_replay_list(
        self, 
        session: aiohttp.ClientSession,
        before: Optional[int] = None,
        limit: int = 51
    ) -> List[ReplayMetadata]:
        """
        Get list of recent replays for the format.
        
        Args:
            session: aiohttp session
            before: Unix timestamp to get replays before this time
            limit: Max replays per request (API limit is ~51)
            
        Returns:
            List of replay metadata
        """
        url = f"{self.BASE_URL}{self.SEARCH_ENDPOINT}"
        params = {"format": self.format}
        if before:
            params["before"] = before
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    replays = []
                    for item in data:
                        replays.append(ReplayMetadata(
                            id=item['id'],
                            format=item.get('format', self.format),
                            players=item.get('players', []),
                            uploadtime=item.get('uploadtime', 0),
                            rating=item.get('rating')
                        ))
                    return replays
                else:
                    logger.warning(f"Search API returned {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching replay list: {e}")
            return []
    
    async def download_replay(
        self, 
        session: aiohttp.ClientSession, 
        replay_id: str,
        semaphore: asyncio.Semaphore
    ) -> Optional[Replay]:
        """
        Download a single replay.
        
        Args:
            session: aiohttp session
            replay_id: Replay ID to download
            semaphore: Rate limiting semaphore
            
        Returns:
            Replay object or None if failed
        """
        if replay_id in self.scraped_ids:
            return None
        
        async with semaphore:
            url = f"{self.BASE_URL}/{replay_id}.json"
            
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract winner from log
                        winner = None
                        log = data.get('log', '')
                        if '|win|' in log:
                            for line in log.split('\n'):
                                if line.startswith('|win|'):
                                    winner = line.split('|win|')[1].strip()
                                    break
                        
                        replay = Replay(
                            id=replay_id,
                            format=data.get('format', self.format),
                            players=data.get('players', []),
                            uploadtime=data.get('uploadtime', 0),
                            rating=data.get('rating'),
                            log=log,
                            winner=winner
                        )
                        
                        return replay
                    elif response.status == 404:
                        self.failed_ids.add(replay_id)
                        return None
                    else:
                        logger.warning(f"Replay {replay_id} returned {response.status}")
                        return None
                        
            except Exception as e:
                logger.error(f"Error downloading {replay_id}: {e}")
                return None
    
    async def save_replay(self, replay: Replay):
        """Save replay to disk."""
        # Organize by date for easier management
        date_str = datetime.fromtimestamp(replay.uploadtime).strftime('%Y-%m-%d')
        date_dir = self.output_dir / date_str
        date_dir.mkdir(exist_ok=True)
        
        filepath = date_dir / f"{replay.id}.json"
        
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(asdict(replay), indent=2))
        
        self.scraped_ids.add(replay.id)
    
    async def scrape(self, target_count: int = 100000):
        """
        Main scraping loop using Producer-Consumer pattern.
        """
        logger.info(f"Starting scrape: target={target_count}, already have={len(self.scraped_ids)}")
        
        # Queue for replays found but not yet downloaded
        # Limit queue size to prevent memory explosion if collection is much faster than download
        queue = asyncio.Queue(maxsize=1000)
        
        # State
        self.running = True
        self.collected_count = 0
        
        # Create session
        connector = aiohttp.TCPConnector(limit=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Stats for progress bar
            # We start with what we have on disk + pending from last run
            initial_count = len(self.scraped_ids)
            
            # Start consumers (downloaders) FIRST so they can drain the queue
            consumers = [
                asyncio.create_task(self._consumer_worker(session, queue))
                for _ in range(self.MAX_CONCURRENT)
            ]
            
            # If we had pending IDs from a previous run, preload them into the queue
            # Now safe because consumers are running
            if self.pending_ids:
                logger.info(f"Queuing {len(self.pending_ids)} pending replays from previous run...")
                for item in self.pending_ids:
                    # Handle both dict and object (pending is dicts)
                    if isinstance(item, dict):
                        # Convert back to metadata-like object/dict for queue
                        await queue.put(item) # Put dict directly
                    else:
                        await queue.put(item)
                # Clear pending since they are now in queue
                self.pending_ids = []
            
            # Start producer (collector)
            producer = asyncio.create_task(
                self._producer_worker(session, queue, target_count)
            )
            
            # Monitoring loop - save progress periodically
            last_save = time.time()
            
            try:
                # Wait for producer to finish (it stops when target reached)
                await producer
                
                # Signal consumers to stop
                for _ in range(self.MAX_CONCURRENT):
                    await queue.put(None)
                
                # Wait for consumers to finish processing queue
                await asyncio.gather(*consumers)
                
            except asyncio.CancelledError:
                self.running = False
                logger.info("Scrape cancelled, saving progress...")
            finally:
                # Save any remaining items in queue to pending_ids
                remaining = []
                while not queue.empty():
                    item = queue.get_nowait()
                    if item is not None:
                        remaining.append(item)
                
                if remaining:
                    logger.info(f"Saving {len(remaining)} queued items to pending...")
                    self.pending_ids = remaining
                
                self._save_progress()
                logger.info(f"Scraping ended. Total: {len(self.scraped_ids)}")

    async def _producer_worker(self, session, queue, target_count):
        """Collects replay IDs and puts them in queue."""
        before_timestamp = self.last_before_timestamp
        
        # Update progress bar logic handled by monitoring script mostly, 
        # but we can log milestones
        
        while self.running and (len(self.scraped_ids) + queue.qsize()) < target_count:
            # Check if queue is full - backpressure
            if queue.full():
                await asyncio.sleep(0.5)
                continue
                
            batch = await self.get_replay_list(session, before=before_timestamp)
            
            if not batch:
                logger.warning("No more replays available via API")
                break
            
            count_added = 0
            for r in batch:
                if r.id in self.scraped_ids:
                    continue
                
                # Filter by rating
                if self.min_rating > 0 and (r.rating is None or r.rating < self.min_rating):
                    self.skipped_low_rating += 1
                    continue
                
                # Check if already in queue (not efficient for large queues, but queue is small)
                # We can skip this check if we trust the API to move backwards in time
                
                # Add to queue as dict (lighter weight)
                await queue.put(asdict(r))
                count_added += 1
                self.collected_count += 1
            
            if batch:
                before_timestamp = min(r.uploadtime for r in batch)
                self.last_before_timestamp = before_timestamp
            
            # Rate limit for search API
            await asyncio.sleep(1 / self.REQUESTS_PER_SECOND)
            
            # Periodic saves handled by parent or consumers?
            # Actually consumers update 'scraped_ids', so main thread monitoring handles saves
            
        logger.info("Producer finished collecting IDs")

    async def _consumer_worker(self, session, queue):
        """Consumes IDs from queue and downloads them."""
        while self.running:
            item = await queue.get()
            
            if item is None:
                # Sentinel to stop
                queue.task_done()
                break
            
            replay_id = item['id']
            
            # Double check (might have been scraped by another worker or prev run)
            if replay_id in self.scraped_ids:
                queue.task_done()
                continue
                
            # Download
            semaphore = asyncio.Semaphore(1) # Unused since we limit consumers count
            # Actually we shouldn't pass semaphore if we limit workers, but let's keep sig
            
            try:
                replay = await self.download_replay(session, replay_id, semaphore)
                if replay:
                    await self.save_replay(replay)
            except Exception as e:
                logger.error(f"Worker error on {replay_id}: {e}")
                
            queue.task_done()
            
            # Small sleep to be nice
            # With 10 workers, 10 reqs/sec total max
            await asyncio.sleep(0.1)
            
            # Save progress periodically (global check?)
            if len(self.scraped_ids) % 100 == 0:
                self._save_progress()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scraping statistics."""
        replay_files = list(self.output_dir.rglob("*.json"))
        replay_files = [f for f in replay_files if f.name != "progress.json"]
        
        return {
            'total_scraped': len(self.scraped_ids),
            'total_failed': len(self.failed_ids),
            'files_on_disk': len(replay_files),
            'output_dir': str(self.output_dir)
        }


async def main():
    parser = argparse.ArgumentParser(description="Scrape Pokemon Showdown replays")
    parser.add_argument("--count", type=int, default=100000, 
                        help="Number of replays to scrape")
    parser.add_argument("--output", type=str, default="data/replays",
                        help="Output directory")
    parser.add_argument("--format", type=str, default="gen9randombattle",
                        help="Battle format to scrape")
    parser.add_argument("--min-rating", type=int, default=0,
                        help="Minimum rating to include (0 = no filter)")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats and exit")
    
    args = parser.parse_args()
    
    scraper = ReplayScraper(format=args.format, output_dir=args.output, min_rating=args.min_rating)
    
    if args.min_rating > 0:
        logger.info(f"Filtering for replays with rating >= {args.min_rating}")
    
    if args.stats:
        stats = scraper.get_stats()
        print(json.dumps(stats, indent=2))
        return
    
    await scraper.scrape(target_count=args.count)
    
    stats = scraper.get_stats()
    print("\nFinal stats:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
