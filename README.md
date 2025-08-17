# Foresight Forge

This project implements a self‑running forecasting service that daily ingests news sources,
summarises new information, generates probabilistic predictions, and publishes a Markdown
newsletter. All data and artefacts are stored as plain text in this repository for transparency.

## Project name: “Foresight Forge”

The name **Foresight Forge** was chosen to capture both the forward-looking aim
of the project (foresight: anticipating future trends) and the idea of an active,
hands-on workshop (forge: where raw materials are refined into finished tools).
It suggests a place where disparate pieces of information (raw news, data, feedback)
are hammered and tempered into clear, actionable predictions.

Other variants considered were **Future Forge**, **Forecast Foundry**, and **Insight Forge**,
but “Foresight Forge” felt the most distinctive and memorable.

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

Run the daily cycle manually (it will skip any stage with no new work):
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
```

### Enhanced Lookback Features

The system now includes historical context and prediction tracking:

**Historical Context in Predictions:**
- The `predict` command now includes the last 7 days of prediction performance
- The model can learn from its own historical accuracy and confidence calibration
- Reviews of previous predictions are included in the context

**Prediction Outcome Tracking:**
```bash
# Mark a prediction as correct/incorrect
python forecast.py mark-outcome --prediction-index 0 --outcome correct
python forecast.py mark-outcome --prediction-index 1 --outcome incorrect
```

**Prediction Updates in Newsletters:**
- The daily newsletter now includes a "Prediction Updates" section
- Shows which previous predictions have been marked as correct/incorrect
- Automatically added when running `run-daily`

**Review Previous Predictions:**
```bash
python forecast.py review  # Reviews yesterday's predictions against today's news
```

**Source Management:**
```bash
python forecast.py cleanup-sources  # Manually trigger source cleanup
python forecast.py discover         # Find new candidate sources
python forecast.py self-update      # Apply source changes
```
## Brain scheduler

You can use a lightweight scheduler to decide whether to run the daily pipeline. The command below
will output `run-digest` if the pipeline hasn't been run today (and record today’s run), or `skip`
if it has already run:

```bash
python forecast.py brain
```

To tie this into your workflow/CI, use the `brain` command, which now also reports a
meta-action plan (sources to add/remove, prompt‑tuning flag) in JSON
alongside the `run-digest`/`skip` decision:

```bash
# Example output:
# run-digest
# {
#   "add_sources": ["https://newfeed.example.com/rss"],
#   "remove_sources": [],
#   "tune_prompts": false
# }
out=$(python forecast.py brain)
echo "$out"
if [ "$(echo "$out" | head -n1)" = "run-digest" ]; then
  python forecast.py run-scheduled
else
  echo "Skipping daily pipeline (already ran today)"
fi
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

### Brain‑gated daily runs

Daily runs are now gated by the built‑in scheduler ("brain"). The CI calls `python forecast.py run-scheduled`,
which runs the full pipeline only if the brain indicates it hasn’t run yet today, and then applies any proposed
source add/remove changes. This prevents duplicate runs and provides a central policy hook.

To preview the decision and meta‑plan locally:

```bash
python forecast.py brain
```

### Strict output constraints (config‑driven)

LLM steps support constrained outputs for reliability using JSON Schemas and Lark grammars. Configuration lives in
`foresight.config.yaml` and is versioned with the repo. Defaults:

```yaml
strict:
  summarise: true   # JSON schema
  predict: true     # JSON schema
  review: true      # JSON schema
  vet: false        # Lark grammar (candidate list)
  sources: false    # Lark grammar (REMOVE:/ADD:)
```

Flip any toggle to `true`/`false` and commit to adjust behavior. The code falls back safely if the SDK rejects
constraints, so runs won’t fail if constraints aren’t supported in a given environment.

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
```
By default, the AI reply will see the entire prior conversation history for that day
(all earlier comments and replies in `comments/YYYY-MM-DD.md`), plus the newsletter summary,
before drafting its response, so follow‑up comments build on the existing thread.

## Roadmap & future enhancements

Below are some planned features and ideas not yet implemented:

- **Reader‑driven suggestions:**  Parse “Suggestion: https://…” lines in reader comments to auto‑harvest new feed URLs.
- **AI source scouting:**  Add a `scout-sources` CLI command that prompts the AI to recommend high‑quality RSS/newsletter feeds based on the daily summary.
- **Web form / Slack bot integration:**  Provide a lightweight web UI or Slack bot so community members can suggest feeds without using the CLI.
    - **Automated onboarding of reader-suggested feeds:**  Extend `self-update` to merge reader-suggested URLs directly alongside scraped candidates in a reviewable PR.
