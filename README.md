# Foresight Forge

This project implements a self‑running forecasting service that daily ingests news sources,
summarises new information, generates probabilistic predictions, and publishes a Markdown
newsletter. All data and artefacts are stored as plain text in this repository for transparency.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a local `.env` file with your API credentials.
   Copy the example template and then edit it:
   ```bash
   cp .env.example .env
   ```
   Open `.env` in your editor and replace the placeholders:
   ```ini
   OPENAI_API_KEY=your_openai_api_key_here
   GITHUB_TOKEN=your_github_token_here
   ```
3. Populate `sources.yaml` with one URL per line of a news/RSS feed you want to track.
4. (Optional) To publish the dashboard via GitHub Pages, enable Pages in repo settings and select the `docs/` folder on the `main` branch.

## Usage

Run the daily cycle manually:
```bash
python forecast.py run-daily
```

You can also invoke individual stages:
```bash
python forecast.py ingest
python forecast.py summarise
python forecast.py predict
python forecast.py record
python forecast.py discover   # weekly source discovery
python forecast.py self-update [--pr]  # merge candidates and optionally open a PR
python forecast.py dashboard      # generate minimal HTML dashboard
```

Comments or review replies for a given date can be added with your editor,
or piped directly on the command line. In both cases the system will also
generate an AI reply based on the day’s summary:
```bash
# launch editor to type your comment interactively
python forecast.py comment --date YYYY-MM-DD

# or pipe in a comment in one shot (no editor)
python forecast.py comment --date YYYY-MM-DD <<< "I love the predictions—what about link X?"
```

## Scheduling

A GitHub Actions workflow (`.github/workflows/daily.yml`) runs the daily pipeline once per day,
and a separate workflow (`.github/workflows/weekly.yml`) performs weekly source discovery
and automatic self‑updates.

## Directory structure

```plain
.
├── .env.example     # template for your local API keys (ignored by git)
├── sources.yaml     # list of RSS or HTTP sources
├── forecast.py      # main CLI application
├── requirements.txt # Python dependencies
├── .github/
│   └── workflows/
│       └── daily.yml # scheduled CI workflow
├── raw/             # daily raw pulls (JSON)
├── summaries/       # daily summaries (MD)
├── newsletters/     # daily newsletters (MD)
├── comments/        # review comments (MD)
├── discover/        # candidate new sources
└── docs/            # generated dashboard HTML
```plaintext
