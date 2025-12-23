"""Debug script to check data loading"""
import sys
sys.path.append('.')

from src.data.loader import load_shot_events

print("Force reloading from JSONs...")
df = load_shot_events(use_cache=False)

print(f"\n✓ Loaded {len(df)} shots")
print(f"\nColumns ({len(df.columns)}):")
print(df.columns.tolist()[:20])

if 'shot_type' in df.columns:
    print(f"\nshot_type values:")
    print(df['shot_type'].value_counts())
else:
    print("\n❌ ERROR: 'shot_type' column missing!")
    print(f"All columns: {df.columns.tolist()}")
