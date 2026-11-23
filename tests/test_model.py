import pandas as pd
import pytest

from music_mining.config import AUDIO_FEATURES
from music_mining.data import load_tracks
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


def test_fit_rejects_missing_numeric_feature_values():
    tracks = make_tracks()
    tracks.loc[0, "tempo"] = float("nan")

    with pytest.raises(ValueError, match="finite numeric"):
        MusicDiscoveryModel(n_clusters=3).fit(tracks)


def test_track_id_query_takes_priority_over_partial_name_match():
    tracks = make_tracks()
    tracks.loc[1, "track_name"] = "ID-12"
    model = MusicDiscoveryModel(n_clusters=3).fit(tracks)

    recommendations = model.recommend("id-1", n=1)

    assert "Song 1" not in recommendations["track_name"].tolist()


def test_load_tracks_supports_spotify_playlist_schema(tmp_path):
    path = tmp_path / "spotify.csv"
    frame = make_tracks()
    frame = frame.rename(
        columns={
            "track_id": "id",
            "track_name": "name",
            "artists": "track_artist",
            "popularity": "track_popularity",
        }
    )
    frame.to_csv(path, index=False)

    loaded = load_tracks(path)

    assert "track_name" in loaded
    assert "artists" in loaded
    assert len(loaded) == len(frame)


def test_recommend_tracks_supports_up_to_three_selected_tracks():
    tracks = make_tracks(12)
    model = MusicDiscoveryModel(n_clusters=3).fit(tracks)

    recommendations = model.recommend_tracks(["id-0", "id-1"], n=4)

    assert len(recommendations) == 4
    assert "Song 0" not in recommendations["track_name"].tolist()
    assert "Song 1" not in recommendations["track_name"].tolist()
