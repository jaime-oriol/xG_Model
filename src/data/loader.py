"""
Load and parse StatsBomb shot events
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Get project root (3 levels up from this file: src/data/loader.py -> src/data -> src -> root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

def load_shot_events(data_dir: str = "data/raw", use_cache: bool = True) -> pd.DataFrame:
    """
    Load all shot events from StatsBomb data

    Args:
        data_dir: Directory with raw StatsBomb data (relative to project root)
        use_cache: If True, use cached pickle if available

    Returns:
        DataFrame with one row per shot
    """
    cache_file = PROJECT_ROOT / "data/processed/shots_cache.pkl"

    # Try cache first
    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        return pd.read_pickle(cache_file)

    # Load from JSONs
    print("Loading from JSONs (this will take ~5 min)...")
    data_path = PROJECT_ROOT / data_dir
    events_dir = data_path / "events"

    shots = []

    for event_file in events_dir.glob("*.json"):
        if event_file.name == "summary.json":
            continue

        with open(event_file) as f:
            events = json.load(f)

        for event in events:
            if event.get('type', {}).get('name') != 'Shot':
                continue

            shot_data = _extract_shot_data(event)
            shots.append(shot_data)

    df = pd.DataFrame(shots)

    # Clean data
    initial_count = len(df)

    # Filter penalties (constant xG ~0.78, sesgan coeficientes)
    df = df[df['shot_type'] != 'Penalty']

    # Filter shots hacia portería incorrecta (X < 60)
    df = df[df['x'] >= 60]

    # Filter shots sin coordenadas válidas
    df = df[df['x'].notna() & df['y'].notna()]

    df = df.reset_index(drop=True)

    filtered_count = initial_count - len(df)
    print(f"Filtered {filtered_count} shots (penalties, X<60, invalid coords)")
    print(f"Clean shots: {len(df)}")

    # Save cache
    cache_file.parent.mkdir(exist_ok=True)
    df.to_pickle(cache_file)
    print(f"Saved cache: {cache_file}")

    return df

def _extract_shot_data(event: Dict) -> Dict:
    """Extract relevant fields from shot event"""
    shot = event.get('shot', {})
    location = event.get('location', [None, None])
    end_location = shot.get('end_location', [None, None, None])

    return {
        # IDs
        'match_id': event.get('match_id'),
        'event_id': event.get('id'),

        # Location
        'x': location[0],
        'y': location[1],
        'end_x': end_location[0] if len(end_location) > 0 else None,
        'end_y': end_location[1] if len(end_location) > 1 else None,
        'end_z': end_location[2] if len(end_location) > 2 else None,

        # Shot characteristics
        'body_part': shot.get('body_part', {}).get('name'),
        'technique': shot.get('technique', {}).get('name'),
        'shot_type': shot.get('type', {}).get('name'),

        # Outcome
        'outcome': shot.get('outcome', {}).get('name'),
        'is_goal': 1 if shot.get('outcome', {}).get('name') == 'Goal' else 0,

        # Reference
        'statsbomb_xg': shot.get('statsbomb_xg'),

        # Optional fields
        'first_time': shot.get('first_time', False),
        'under_pressure': event.get('under_pressure', False),
        'one_on_one': shot.get('one_on_one', False),
        'aerial_won': shot.get('aerial_won', False),

        # Freeze frame (for later phases)
        'has_freeze_frame': 'freeze_frame' in shot,
        'freeze_frame': shot.get('freeze_frame'),
    }

def load_shot_events_with_360(data_dir: str = "data/raw", use_cache: bool = True) -> pd.DataFrame:
    """
    Load shot events with 360 data merged

    Args:
        data_dir: Directory with raw StatsBomb data (relative to project root)
        use_cache: If True, use cached pickle if available

    Returns:
        DataFrame with shots + 360 features (where available)
    """
    cache_file = PROJECT_ROOT / "data/processed/shots_360_cache.pkl"

    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        return pd.read_pickle(cache_file)

    # Load base shots
    df = load_shot_events(data_dir=data_dir, use_cache=use_cache)

    # Load 360 data
    print("Loading 360 data...")
    data_path = PROJECT_ROOT / data_dir
    three_sixty_dir = data_path / "three-sixty"

    if not three_sixty_dir.exists():
        print("No 360 data directory found")
        df['visible_area_360'] = None
        df['freeze_frame_360'] = None
        return df

    # Build mapping of event_id -> 360 data
    three_sixty_map = {}

    for match_file in three_sixty_dir.glob("*.json"):
        with open(match_file) as f:
            events_360 = json.load(f)

        for event_360 in events_360:
            # Skip if not a dict (malformed data)
            if not isinstance(event_360, dict):
                continue

            event_uuid = event_360.get('event_uuid')
            if event_uuid:
                three_sixty_map[event_uuid] = {
                    'visible_area': event_360.get('visible_area'),
                    'freeze_frame': event_360.get('freeze_frame'),
                }

    print(f"Loaded 360 data for {len(three_sixty_map)} events")

    # Merge 360 data with shots
    df['visible_area_360'] = None
    df['freeze_frame_360'] = None

    matched = 0
    for idx, row in df.iterrows():
        event_id = row['event_id']
        if event_id in three_sixty_map:
            df.at[idx, 'visible_area_360'] = three_sixty_map[event_id]['visible_area']
            df.at[idx, 'freeze_frame_360'] = three_sixty_map[event_id]['freeze_frame']
            matched += 1

    print(f"Matched 360 data for {matched} / {len(df)} shots ({matched/len(df)*100:.1f}%)")

    # Rename base coordinates
    df['shot_x'] = df['x']
    df['shot_y'] = df['y']

    # Save cache
    cache_file.parent.mkdir(exist_ok=True)
    df.to_pickle(cache_file)
    print(f"Saved cache: {cache_file}")

    return df
