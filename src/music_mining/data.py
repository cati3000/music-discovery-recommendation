"""Dataset loading and validation."""

from pathlib import Path

import pandas as pd

from .config import AUDIO_FEATURES


def load_tracks(path: str | Path) -> pd.DataFrame:
    """Load, validate, and clean a Spotify tracks CSV."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    aliases = {
        "track_name": ("track_name", "name"),
        "artists": ("artists", "track_artist"),
        "track_id": ("track_id", "id"),
        "popularity": ("popularity", "track_popularity"),
        "album_name": ("album_name", "track_album_name"),
    }
    for canonical, candidates in aliases.items():
        if canonical not in frame.columns:
            for candidate in candidates:
                if candidate in frame.columns:
                    frame[canonical] = frame[candidate]
                    break

    missing = [column for column in AUDIO_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required audio feature columns: "
            + ", ".join(missing)
        )

    frame = frame.copy()
    for column in AUDIO_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=AUDIO_FEATURES).reset_index(drop=True)

    if "track_id" in frame.columns:
        frame = frame.drop_duplicates(subset="track_id").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No valid tracks remain after cleaning the dataset.")
    return frame
