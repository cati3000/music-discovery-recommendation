"""Small local web application for music recommendations."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .data import load_tracks
from .model import MusicDiscoveryModel


PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Music Discovery Lab</title>
  <style>
    :root { color-scheme: dark; --bg:#09111f; --card:#172236; --accent:#8b5cf6; --text:#edf2ff; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:16px system-ui,sans-serif; }
    main { max-width:1100px; margin:auto; padding:32px 20px 60px; }
    h1 { margin:0 0 8px; font-size:clamp(32px,6vw,56px); }
    .sub { color:#a9b8d4; margin-bottom:28px; }
    .search { display:flex; gap:10px; }
    input,button { border-radius:12px; border:1px solid #3a4963; padding:14px 16px; font:inherit; }
    input { flex:1; background:#0d1829; color:white; }
    button { background:var(--accent); color:white; border:0; cursor:pointer; font-weight:700; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    .choices { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:10px; margin-top:14px; }
    .choice { text-align:left; background:#0d1829; border:1px solid #3a4963; color:white; }
    .choice.selected { border-color:#a78bfa; background:#30275a; }
    .selected-list { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
    .pill { background:#30275a; border:1px solid #7056bd; border-radius:999px; padding:8px 12px; }
    .pill button { padding:0 0 0 8px; background:none; color:#c4b5fd; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin-top:24px; }
    .card { background:var(--card); border-radius:16px; padding:16px; box-shadow:0 12px 30px #0003; }
    .card h3 { margin:0 0 8px; }
    .meta { color:#b7c5df; font-size:14px; }
    .score { float:right; color:#c4b5fd; }
    #status { margin-top:14px; color:#a9b8d4; }
  </style>
</head>
<body><main>
  <h1>Music Discovery Lab</h1>
  <p class="sub">Choose up to three songs, then discover tracks with similar audio characteristics.</p>
  <div class="search">
    <input id="query" placeholder="Try a track name, artist, or track ID" aria-label="Music query">
    <button id="search">Search</button>
  </div>
  <div id="status"></div>
  <div id="choices" class="choices"></div>
  <div id="selected" class="selected-list"></div>
  <button id="recommend" disabled style="margin-top:16px">Get recommendations</button>
  <div id="results" class="grid"></div>
</main>
<script>
const selections = new Map();
async function searchTracks() {
  const query = document.getElementById("query").value.trim();
  if (!query) return;
  document.getElementById("status").textContent = "Searching...";
  const response = await fetch("/api/search?q=" + encodeURIComponent(query));
  const data = await response.json();
  if (!response.ok) {
    document.getElementById("status").textContent = data.error || "No matches found.";
    document.getElementById("choices").innerHTML = "";
    return;
  }
  document.getElementById("status").textContent = `Choose a track (${data.results.length} matches)`;
  document.getElementById("choices").innerHTML = data.results.map(track =>
    `<button class="choice ${selections.has(track.track_id) ? "selected" : ""}" data-id="${escapeHtml(track.track_id)}">
      <strong>${escapeHtml(track.track_name || "Unknown track")}</strong><br>
      <span class="meta">${escapeHtml(track.artists || "Unknown artist")}</span>
    </button>`
  ).join("");
  document.querySelectorAll(".choice").forEach(button => button.addEventListener("click", () => toggleSelection(
    button.dataset.id,
    button.querySelector("strong").textContent,
    button.querySelector(".meta").textContent
  )));
}
function toggleSelection(id, name, artist) {
  if (selections.has(id)) selections.delete(id);
  else if (selections.size < 3) selections.set(id, {name, artist});
  else {
    document.getElementById("status").textContent = "You can choose up to three tracks.";
    return;
  }
  renderSelections();
  searchTracks();
}
function renderSelections() {
  document.getElementById("selected").innerHTML = [...selections].map(([id, track]) =>
    `<span class="pill">${escapeHtml(track.name)} — ${escapeHtml(track.artist)}
      <button aria-label="Remove ${escapeHtml(track.name)}" data-remove="${escapeHtml(id)}">×</button>
    </span>`
  ).join("");
  document.querySelectorAll("[data-remove]").forEach(button => button.addEventListener("click", () => {
    selections.delete(button.dataset.remove);
    renderSelections();
  }));
  document.getElementById("recommend").disabled = selections.size === 0;
}
async function discover() {
  if (!selections.size) return;
  document.getElementById("status").textContent = "Finding similar tracks...";
  const ids = [...selections.keys()].map(id => "id=" + encodeURIComponent(id)).join("&");
  const response = await fetch("/api/recommend?" + ids + "&n=12");
  const data = await response.json();
  if (!response.ok) {
    document.getElementById("status").textContent = data.error || "No matches found.";
    document.getElementById("results").innerHTML = "";
    return;
  }
  document.getElementById("status").textContent = data.message;
  document.getElementById("results").innerHTML = data.results.map(track =>
    `<article class="card">
      <span class="score">${(track.similarity * 100).toFixed(1)}% match</span>
      <h3>${escapeHtml(track.track_name || "Unknown track")}</h3>
      <div class="meta">${escapeHtml(track.artists || "Unknown artist")}</div>
      <div class="meta">Profile ${track.cluster} · ${track.playlist_genre || ""}</div>
    </article>`
  ).join("");
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, char =>
    ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" })[char]);
}
document.getElementById("search").addEventListener("click", searchTracks);
document.getElementById("recommend").addEventListener("click", discover);
document.getElementById("query").addEventListener("keydown", event => {
  if (event.key === "Enter") searchTracks();
});
</script></body></html>"""


class MusicApp:
    def __init__(self, dataset: Path, n_clusters: int = 5, sample_size: int = 30000) -> None:
        tracks = load_tracks(dataset)
        if sample_size > 0 and len(tracks) > sample_size:
            tracks = tracks.sample(sample_size, random_state=42).reset_index(drop=True)
        self.model = MusicDiscoveryModel(n_clusters=n_clusters).fit(tracks)
        self.track_count = len(tracks)


def create_handler(app: MusicApp):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if parsed.path == "/api/search":
                params = parse_qs(parsed.query)
                query = params.get("q", [""])[0].strip()
                matches = app.model._find_query_matches(query)
                records = matches.head(8)
                columns = [column for column in ("track_id", "track_name", "artists") if column in records]
                self.send_json(200, {
                    "results": records[columns].to_dict(orient="records"),
                })
                return
            if parsed.path == "/api/recommend":
                params = parse_qs(parsed.query)
                try:
                    count = int(params.get("n", ["10"])[0])
                    track_ids = params.get("id", [])
                    if track_ids:
                        recommendations = app.model.recommend_tracks(track_ids, count)
                    else:
                        query = params.get("q", [""])[0].strip()
                        recommendations = app.model.recommend(query, count)
                except (ValueError, RuntimeError) as error:
                    self.send_json(404 if "No track matched" in str(error) else 400, {"error": str(error)})
                    return

                columns = [
                    "similarity", "track_name", "artists", "track_id",
                    "track_popularity", "cluster", "playlist_genre",
                ]
                available = [column for column in columns if column in recommendations.columns]
                records = recommendations[available].to_dict(orient="records")
                for record in records:
                    record["similarity"] = float(record["similarity"])
                    if "track_popularity" in record:
                        record["popularity"] = record.pop("track_popularity")
                self.send_json(200, {
                    "message": f"{len(records)} recommendations from {app.track_count:,} tracks",
                    "results": records,
                })
                return
            self.send_json(404, {"error": "Not found"})

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("data/spotify dataset.csv"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=30000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = MusicApp(args.dataset, args.clusters, args.sample_size)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(app))
    print(f"Music Discovery Lab running at http://{args.host}:{args.port}")
    server.serve_forever()
