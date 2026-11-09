# Music Discovery and Recommendation

A clean Python data-mining project for discovering patterns in Spotify audio
features, grouping tracks with K-Means, visualizing them with PCA, and
generating content-based recommendations with cosine similarity.

## Project structure

```text
data/          Put the downloaded Spotify CSV here
outputs/       Generated clusters, metrics, and plots
src/music_mining/
  config.py    Feature definitions
  data.py      CSV loading and validation
  model.py     Scaling, K-Means, PCA, evaluation, recommendations
  cli.py       Command-line interface
tests/         Automated tests
```

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, activation is optional. Run the
environment's executables directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\music-discovery.exe analyze data\dataset.csv
```

Download the dataset from the [Spotify tracks dataset repository](https://github.com/sai-chaitanya-reddy/spotify-tracks-dataset)
and save its CSV in `data/`. The downloaded file may be named
`spotify-tracks-dataset-detailed.csv`; use the actual filename in the commands
below.

## Run the analysis

```powershell
.\.venv\Scripts\music-discovery.exe analyze data\spotify-tracks-dataset-detailed.csv --output-dir outputs --clusters 5
```

This writes `clustered_tracks.csv`, `cluster_evaluation.csv`, and
`clusters_pca.png`. The analysis standardizes every audio feature before
clustering, preventing tempo and loudness from dominating the distance metric.
The evaluation file includes inertia (elbow method) and silhouette score for
candidate cluster counts.

## Compare cluster choices in the terminal

To compare `k=2` and `k=6` on all available tracks without changing the
dashboard:

```powershell
.\.venv\Scripts\music-discovery.exe compare data\spotify-tracks-dataset-detailed.csv
```

This prints inertia, silhouette, Calinski-Harabasz, Davies-Bouldin, runtime,
and the percentage of tracks in the smallest cluster. Higher silhouette and
Calinski-Harabasz scores are better; lower inertia and Davies-Bouldin scores
are better.

## Open the visual dashboard

After analysis, create an interactive dashboard:

```powershell
.\.venv\Scripts\music-discovery.exe dashboard outputs --open
```

The dashboard includes a zoomable music map, cluster-size chart, audio-feature
heatmap, and cluster-evaluation chart. Hover over points to see track names,
click legend entries to isolate profiles, and zoom into dense regions.

## Get recommendations

```powershell
.\.venv\Scripts\music-discovery.exe recommend data\spotify-tracks-dataset-detailed.csv "Blinding Lights" -n 10
```

The query can be a track name, artist, or Spotify track ID. Recommendations
are based on cosine similarity in the standardized nine-feature space.

## Test

```powershell
py -m pytest
```
