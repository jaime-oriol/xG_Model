"""
Load and parse StatsBomb shot events
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict

def load_shot_events(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Load all shot events from StatsBomb data

    Returns:
        DataFrame with one row per shot
    """
    data_path = Path(data_dir)
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
