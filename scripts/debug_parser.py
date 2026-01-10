import sys
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.replay_parser import ReplayParser

logging.basicConfig(level=logging.DEBUG)

def main():
    path = "data/replays/2025-10-03/gen9randombattle-2453962376.json"
    print(f"Parsing {path}...")
    
    with open(path) as f:
        data = json.load(f)
    
    parser = ReplayParser()
    try:
        states = parser.parse_replay(data)
        print(f"Result: {len(states)} states found.")
        if len(states) == 0:
            print("Why 0?")
            log = data.get('log', '')
            winner = data.get('winner')
            print(f"Log length: {len(log)}")
            print(f"Winner metadata: {winner}")
            
            # Check manual extraction
            found_winner = None
            if log:
                for line in log.split('\n'):
                    if line.startswith('|win|'):
                        found_winner = line.split('|')[2]
                        break
            print(f"Extracted winner from log: {found_winner}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
