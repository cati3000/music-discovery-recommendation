"""Command-line entry point for the music discovery project."""

import argparse
from pathlib import Path
from time import perf_counter
import webbrowser

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import load_tracks
from .config import AUDIO_FEATURES
from .model import MusicDiscoveryModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Cluster and evaluate tracks.")
    analyze.add_argument("dataset", type=Path)
    analyze.add_argument("--output-dir", type=Path, default=Path("outputs"))
    analyze.add_argument("--clusters", type=int, default=5)
    analyze.add_argument("--sample-size", type=int, default=10000)

    compare = subparsers.add_parser(
        "compare", help="Compare selected cluster counts in the command window."
    )
    compare.add_argument("dataset", type=Path)
    compare.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[2, 6],
        help="Cluster counts to compare (default: 2 6).",
    )
    compare.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Rows to use; 0 uses the complete dataset.",
    )

    recommend = subparsers.add_parser("recommend", help="Recommend similar tracks.")
    recommend.add_argument("dataset", type=Path)
    recommend.add_argument("query", help="Track name, artist, or track ID.")
    recommend.add_argument("--clusters", type=int, default=5)
    recommend.add_argument("-n", type=int, default=10)

    dashboard = subparsers.add_parser(
        "dashboard", help="Create an interactive visual dashboard."
    )
    dashboard.add_argument("output_dir", type=Path, default=Path("outputs"), nargs="?")
    dashboard.add_argument("--open", action="store_true", help="Open the dashboard in a browser.")
    return parser


def run_analysis(args: argparse.Namespace) -> None:
    tracks = load_tracks(args.dataset)
    if args.sample_size > 0 and len(tracks) > args.sample_size:
        tracks = tracks.sample(args.sample_size, random_state=42).reset_index(drop=True)

    model = MusicDiscoveryModel(n_clusters=args.clusters).fit(tracks)
    evaluation = model.evaluate(tracks)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.tracks.to_csv(output_dir / "clustered_tracks.csv", index=False)
    evaluation.scores.to_csv(output_dir / "cluster_evaluation.csv", index=False)

    plt.figure(figsize=(9, 6))
    for cluster, group in model.tracks.groupby("cluster"):
        plt.scatter(group["pca_1"], group["pca_2"], s=8, alpha=0.45, label=f"Cluster {cluster}")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("Spotify tracks in PCA space")
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(output_dir / "clusters_pca.png", dpi=160)
    plt.close()
    print(evaluation.scores.to_string(index=False))


def run_recommendation(args: argparse.Namespace) -> None:
    model = MusicDiscoveryModel(n_clusters=args.clusters).fit(load_tracks(args.dataset))
    columns = [column for column in ("similarity", "track_name", "artists", "cluster") if column in model.tracks]
    print(model.recommend(args.query, args.n)[columns].to_string(index=False))


def run_comparison(args: argparse.Namespace) -> None:
    """Print clustering quality metrics for selected values of k."""
    if len(args.clusters) < 2 or any(count < 2 for count in args.clusters):
        raise ValueError("Provide at least two cluster counts, each at least 2.")
    tracks = load_tracks(args.dataset)
    if args.sample_size > 0 and len(tracks) > args.sample_size:
        tracks = tracks.sample(args.sample_size, random_state=42).reset_index(drop=True)

    start = perf_counter()
    model = MusicDiscoveryModel(n_clusters=args.clusters[0])
    metrics = model.evaluate(tracks, cluster_counts=args.clusters).scores
    elapsed = perf_counter() - start
    metrics.insert(0, "rows", len(tracks))
    metrics["runtime_seconds"] = round(elapsed, 2)
    metrics["smallest_cluster_pct"] = metrics["smallest_cluster_pct"].round(2)
    print(metrics.to_string(index=False))
    print("\nMetric interpretation:")
    print("- Higher silhouette and Calinski-Harabasz scores are better.")
    print("- Lower Davies-Bouldin score and inertia are better.")
    print("- smallest_cluster_pct shows whether a solution creates tiny clusters.")


def run_dashboard(args: argparse.Namespace) -> None:
    """Build a self-contained interactive dashboard from analysis outputs."""
    output_dir = args.output_dir
    tracks_path = output_dir / "clustered_tracks.csv"
    evaluation_path = output_dir / "cluster_evaluation.csv"
    if not tracks_path.is_file() or not evaluation_path.is_file():
        raise FileNotFoundError(
            f"Expected {tracks_path.name} and {evaluation_path.name} in {output_dir}. "
            "Run the analyze command first."
        )

    tracks = pd.read_csv(tracks_path)
    evaluation = pd.read_csv(evaluation_path)
    display_tracks = tracks.sample(min(20000, len(tracks)), random_state=42)
    profiles = tracks.groupby("cluster")[AUDIO_FEATURES].mean()
    sizes = tracks["cluster"].value_counts().sort_index()

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "The Music Map",
            "How large is each music profile?",
            "What defines each profile?",
            "Choosing the number of clusters",
        ),
        specs=[[{"type": "scatter"}, {"type": "bar"}], [{"type": "heatmap"}, {"type": "xy"}]],
        vertical_spacing=0.16,
        horizontal_spacing=0.1,
    )
    for cluster in sorted(display_tracks["cluster"].unique()):
        group = display_tracks[display_tracks["cluster"] == cluster]
        figure.add_trace(
            go.Scattergl(
                x=group["pca_1"],
                y=group["pca_2"],
                mode="markers",
                name=f"Profile {cluster}",
                text=group.get("track_name", pd.Series("", index=group.index)),
                customdata=group.get("artists", pd.Series("", index=group.index)),
                hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
                marker={"size": 5, "opacity": 0.55},
            ),
            row=1,
            col=1,
        )
    figure.add_trace(
        go.Bar(
            x=[f"Profile {cluster}" for cluster in sizes.index],
            y=sizes.values,
            text=[f"{size / len(tracks):.1%}" for size in sizes.values],
            textposition="auto",
            marker_color="#7c3aed",
            name="Tracks",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Heatmap(
            z=profiles.values,
            x=profiles.columns,
            y=[f"Profile {cluster}" for cluster in profiles.index],
            colorscale="Viridis",
            hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
            showscale=False,
        ),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=evaluation["n_clusters"],
            y=evaluation["silhouette_score"],
            mode="lines+markers",
            name="Silhouette score",
            line={"color": "#f97316", "width": 3},
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=evaluation["n_clusters"],
            y=evaluation["inertia"],
            mode="lines+markers",
            name="Inertia",
            yaxis="y2",
            line={"color": "#2563eb", "width": 2, "dash": "dot"},
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    figure.update_xaxes(title_text="Principal component 1", row=1, col=1)
    figure.update_yaxes(title_text="Principal component 2", row=1, col=1)
    figure.update_yaxes(title_text="Number of tracks", row=1, col=2)
    figure.update_xaxes(title_text="Audio feature", row=2, col=1, tickangle=-35)
    figure.update_xaxes(title_text="Number of clusters (k)", row=2, col=2)
    figure.update_yaxes(title_text="Silhouette", row=2, col=2)
    figure.update_layout(
        title={"text": "Music Discovery Lab", "x": 0.5, "xanchor": "center"},
        template="plotly_dark",
        height=950,
        margin={"t": 100, "l": 60, "r": 60, "b": 70},
        legend={"orientation": "h", "y": 1.04, "x": 0},
    )
    best = evaluation.loc[evaluation["silhouette_score"].idxmax()]
    cluster_count = len(profiles)
    explanation = (
        f"<h2>What are we seeing?</h2>"
        f"<p>Each dot is a song. Nearby dots have similar audio characteristics. "
        f"The model found <b>{cluster_count} musical profiles</b> in the analyzed dataset.</p>"
        f"<p>The strongest statistical separation occurs at <b>k={int(best['n_clusters'])}</b> "
        f"with a silhouette score of <b>{best['silhouette_score']:.3f}</b>. "
        "The heatmap shows which audio features make each profile distinctive.</p>"
        "<p><b>Try this:</b> hover over points in the Music Map, click legend items "
        "to isolate profiles, and zoom into dense regions.</p>"
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Music Discovery Lab</title>
<style>
body {{ margin:0; background:#111827; color:#e5e7eb; font:16px system-ui, sans-serif; }}
main {{ max-width:1400px; margin:auto; padding:24px; }}
h1 {{ font-size:42px; margin:0 0 8px; color:#f9fafb; }}
.subtitle {{ color:#a5b4fc; margin:0 0 20px; }}
.card {{ background:#1f2937; border-radius:16px; padding:20px; box-shadow:0 8px 30px #0005; }}
footer {{ color:#9ca3af; font-size:13px; margin-top:14px; }}
</style></head><body><main>
<h1>Music Discovery Lab</h1>
<p class="subtitle">Finding hidden structure in {len(tracks):,} songs using data mining</p>
<section class="card">{explanation}</section>
<section class="card">{figure.to_html(full_html=False, include_plotlyjs=True)}</section>
<footer>Features: danceability, energy, acousticness, instrumentalness, speechiness, tempo, loudness, valence, popularity.</footer>
</main></body></html>"""
    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")
    print(f"Dashboard written to {dashboard_path}")
    if args.open:
        webbrowser.open(dashboard_path.resolve().as_uri())


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "analyze":
        run_analysis(args)
    elif args.command == "recommend":
        run_recommendation(args)
    elif args.command == "compare":
        run_comparison(args)
    else:
        run_dashboard(args)
