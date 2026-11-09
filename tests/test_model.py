import pandas as pd
import pytest

from music_mining.config import AUDIO_FEATURES
from music_mining.model import MusicDiscoveryModel


def make_tracks(count=12):
    rows = []
    for index in range(count):
        row = {feature: float(index + 1) for feature in AUDIO_FEATURES}
        row.update(track_id=f"id-{index}", track_name=f"Song {index}", artists="Artist")
        rows.append(row)
    return pd.DataFrame(rows)


def test_fit_adds_clusters_and_pca_coordinates():
    result = MusicDiscoveryModel(n_clusters=3).fit(make_tracks())
    assert {"cluster", "pca_1", "pca_2"} <= set(result.tracks.columns)
    assert len(result.tracks["cluster"].unique()) == 3


def test_recommendation_excludes_query_track():
    model = MusicDiscoveryModel(n_clusters=3).fit(make_tracks())
    recommendations = model.recommend("Song 0", n=3)
    assert len(recommendations) == 3
    assert "Song 0" not in recommendations["track_name"].tolist()


def test_invalid_query_is_explicit():
    model = MusicDiscoveryModel(n_clusters=3).fit(make_tracks())
    with pytest.raises(ValueError, match="No track matched"):
        model.recommend("missing song")
