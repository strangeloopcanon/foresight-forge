#!/usr/bin/env python3
"""
Main CLI for the Foresight Forge daily forecasting pipeline.
"""
import os
import json
import sys
import datetime

import click
import yaml
import feedparser
import requests
# Centralise the model selection so we can easily switch in one place.
import openai
from openai import OpenAI, OpenAIError

# ---------------------------------------------------------------------------
# OpenAI model name used throughout the pipeline. By centralising this in a
# single constant, we avoid having to touch every call site when switching
# models in the future.
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL = os.getenv("FORESIGHT_LLM_MODEL", "gpt-3.5-turbo")
from git import Repo
import glob
import re
import shutil
from dotenv import load_dotenv


def load_sources(path="sources.yaml"):  # noqa: E302
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return yaml.safe_load(f) or []


@click.group()
def cli():
    """Forecasting pipeline commands."""
    pass


def _should_run_digest():
    """
    Return True if the daily digest should run (not run yet today), False otherwise.
    Does NOT update the scheduler state - that should be done after successful completion.
    """
    state_file = "brain_state.json"
    today = datetime.date.today().isoformat()
    if os.path.exists(state_file):
        state = json.load(open(state_file))
        last = state.get("last_run")
    else:
        state = {}
        last = None

    # # Advanced scheduling logic (commented out for now):
    # # 1) Random jitter: occasionally run early
    # # import random
    # # if random.random() < 0.1:
    # #     run = True
    # # 2) Volume threshold: only run if enough new items
    # # if os.path.exists(f"raw/{today}.json"):
    # #     items = json.load(open(f"raw/{today}.json"))
    # #     if len(items) < 5:
    # #         run = False
    # # 3) AI-driven decision: use LLM to decide based on context
    # # run = ai_scheduler_decision(...)

    # Basic scheduling: run if not already run today
    run = (last != today)

    # Placeholder for future source-discovery or topic-trigger logic (step 3)
    # # if new_topic_spike:
    # #     run = True

    return run

def _brain_vet_candidates(candidates, existing_sources):
    """Use LLM to vet candidate sources and return only high-quality ones."""
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        return []
    
    openai.api_key = key
    
    prompt = (
        "You are an expert analyst for Foresight Forge, a project that forecasts financial, economic, "
        "scientific, and geopolitical trends. Your task is to review the following list of potential "
        "new RSS feed domains and select ONLY the ones that are HIGHLY USEFUL AND INTERESTING for forecasting. "
        "Be EXTREMELY selective - only approve domains that are:\n"
        "1. Major established news outlets with high editorial standards\n"
        "2. Respected financial/economics publications\n"
        "3. Well-known scientific journals or research institutions\n"
        "4. High-quality blogs by recognized experts in relevant fields\n"
        "5. Government or international organization sources\n\n"
        "Reject personal blogs, small websites, or any domains that don't clearly meet these standards. "
        "Look for sources that provide unique insights, data, or analysis that would be valuable for forecasting.\n\n"
        "Format your response as a simple list of approved domains, one per line, like '- example.com/rss'. "
        "Do not include justifications or any other text. If no domains meet the high standards, respond with 'NONE'.\n\n"
        "Candidate Domains:\n" + "\n".join(sorted(candidates))
    )
    
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
        )
        approved_text = resp.choices[0].message.content.strip()
        
        if approved_text.upper() == 'NONE':
            return []
        
        # Parse the response to get the final list of URLs
        approved_urls = [line.lstrip('- ').strip() for line in approved_text.split('\n') if line.strip()]
        return [u for u in approved_urls if u not in existing_sources]
        
    except Exception as e:
        click.echo(f"Error during brain vetting: {e}")
        return []

def _brain_source_review(sources):
    """Use LLM to periodically review and optimize the source list."""
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        return [], []
    
    openai.api_key = key
    
    prompt = (
        "You are an expert analyst for Foresight Forge, a project that forecasts financial, economic, "
        "scientific, and geopolitical trends. Review the current RSS sources and suggest improvements:\n\n"
        "Current Sources:\n" + "\n".join(f"- {s}" for s in sources) + "\n\n"
        "Tasks:\n"
        "1. Identify sources that should be REMOVED (broken, low-quality, redundant, or no longer relevant)\n"
        "2. Suggest NEW sources that would be valuable additions (major outlets, respected publications, expert blogs)\n\n"
        "Be AGGRESSIVE about cleaning up. Remove sources that are:\n"
        "- Personal blogs or low-quality websites\n"
        "- Duplicate or redundant sources\n"
        "- Sources that haven't provided valuable insights\n"
        "- Broken or non-functional feeds\n"
        "- Sources that are too niche or irrelevant to forecasting\n"
        "- Sources that primarily produce clickbait or low-quality content\n\n"
        "Only keep sources that are:\n"
        "- Major established news outlets with high editorial standards\n"
        "- Respected financial/economics publications\n"
        "- Well-known scientific journals or research institutions\n"
        "- High-quality blogs by recognized experts in relevant fields\n"
        "- Government or international organization sources\n\n"
        "Format your response as:\n"
        "REMOVE:\n- source1.com/rss\n- source2.com/rss\n\n"
        "ADD:\n- newsource1.com/rss\n- newsource2.com/rss\n\n"
        "Be very selective. Only suggest removing sources that are clearly problematic, and only suggest adding sources that are clearly high-quality and relevant."
    )
    
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=800,
        )
        review_text = resp.choices[0].message.content.strip()
        
        # Parse the response
        to_remove = []
        to_add = []
        
        lines = review_text.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line == "REMOVE:":
                current_section = "remove"
            elif line == "ADD:":
                current_section = "add"
            elif line.startswith('- ') and current_section:
                url = line[2:].strip()
                if current_section == "remove":
                    to_remove.append(url)
                elif current_section == "add":
                    to_add.append(url)
        
        return to_add, to_remove
        
    except Exception as e:
        click.echo(f"Error during brain source review: {e}")
        return [], []

def _mark_run_completed():
    """Mark today's run as completed in the brain state."""
    state_file = "brain_state.json"
    today = datetime.date.today().isoformat()
    if os.path.exists(state_file):
        state = json.load(open(state_file))
    else:
        state = {}
    state["last_run"] = today
    with open(state_file, "w") as f:
        json.dump(state, f)

def _brain_decision():
    """
    High-level meta decision: whether to run today's digest,
    plus propose new sources to add/remove and whether to tune prompts.
    """
    run = _should_run_digest()
    # load current sources list
    sources = load_sources()
    
    # Brain-powered source review: periodically evaluate and update sources
    today = datetime.date.today().isoformat()
    last_review_file = "brain_source_review.json"
    
    # Check if we should do a source review (every 7 days, but more aggressive cleanup)
    should_review = False
    if os.path.exists(last_review_file):
        review_state = json.load(open(last_review_file))
        last_review = review_state.get("last_review")
        if last_review:
            days_since = (datetime.date.today() - datetime.date.fromisoformat(last_review)).days
            should_review = days_since >= 7
    else:
        should_review = True
    
    if should_review:
        click.echo("Brain conducting periodic source review...")
        to_add, to_remove = _brain_source_review(sources)
        
        # Log the cleanup actions
        if to_remove:
            click.echo(f"Sources to remove: {len(to_remove)}")
            for source in to_remove:
                click.echo(f"  - {source}")
        
        if to_add:
            click.echo(f"Sources to add: {len(to_add)}")
            for source in to_add:
                click.echo(f"  + {source}")
        
        # Update review timestamp
        with open(last_review_file, "w") as f:
            json.dump({"last_review": today}, f)
    else:
        # gather latest candidate sources from discover/ folder
        candidates = []
        files = sorted(glob.glob('discover/*-candidates.md'))
        if files:
            latest = files[-1]
            for line in open(latest):
                if line.startswith('-'):
                    candidates.append(line.lstrip('- ').strip())

        # Only add high-quality candidates that pass brain review
        to_add = []
        if candidates:
            to_add = _brain_vet_candidates(candidates, sources)
        to_remove = []

    # ------------- Parse user comments for explicit source suggestions ---------
    today = datetime.date.today().isoformat()
    comment_path = f"comments/{today}.md"
    if os.path.exists(comment_path):
        comment_text = open(comment_path).read()
        url_regex = re.compile(r"https?://[^\s>]+", re.I)
        for url in url_regex.findall(comment_text):
            if url not in sources and url not in to_add:
                to_add.append(url)
    to_remove = []
    tune_prompts = False

    # # AI-driven meta-scheduler placeholder:
    # # Prompt an LLM: given recent summaries, source list, performance metrics,
    # # decide which sources to add, which to drop, and whether to adjust prompts.
    # # Parse the model's response into to_add, to_remove, tune_prompts.

    return {
        'run': run,
        'add_sources': to_add,
        'remove_sources': to_remove,
        'tune_prompts': tune_prompts,
    }

@cli.command()
def brain():
    """
    Decide whether to run today's digest based on last run date.
    Outputs "run-digest" (and records today's run) or "skip".
    """
    # Run decision and meta-actions (sources to add/remove, prompt tuning)
    decision = _brain_decision()
    # Basic run/skip output for scheduled pipelines
    click.echo("run-digest" if decision.get("run") else "skip")
    # Meta-action plan as JSON
    plan = {
        "add_sources": decision.get("add_sources", []),
        "remove_sources": decision.get("remove_sources", []),
        "tune_prompts": decision.get("tune_prompts", False),
    }
    click.echo(json.dumps(plan, indent=2))

@cli.command(name="run-scheduled")
def run_scheduled():
    """
    Run the full daily pipeline only if the brain scheduler indicates it's time.
    """
    decision = _brain_decision()

    if decision.get('run'):
        click.echo("Running scheduled daily pipeline")
        try:
            run_daily.callback()
            _mark_run_completed()
            click.echo("Daily pipeline completed successfully")
        except Exception as e:
            click.echo(f"Daily pipeline failed: {e}")
            # Don't mark as completed if it failed
    else:
        click.echo("Skipping scheduled daily pipeline; already ran today")

    # If the brain proposed new sources or removals, apply them automatically
    if decision.get('add_sources') or decision.get('remove_sources'):
        click.echo(f"Brain proposed {len(decision.get('add_sources', []))} new source(s) and {len(decision.get('remove_sources', []))} removal(s).")
        try:
            ctx = click.get_current_context()
            ctx.invoke(self_update, pr=False)  # Direct merge, no PR
        except Exception as e:
            click.echo(f"Failed to apply brain's source changes: {e}")
# Register ingest as a Click sub-command so that run_daily can reliably invoke
# it via `ingest.callback()` (the underlying function reference Click attaches).
# This was previously missing, causing AttributeError during the daily pipeline.
@cli.command()
def ingest():
    """Ingest new items from all sources."""
    state_file = "state.json"
    if os.path.exists(state_file):
        state = json.load(open(state_file))
    else:
        state = {"seen": []}
    seen = set(state.get("seen", []))
    new_items = []
    for url in load_sources():
        try:
            # Use requests with timeout to prevent hanging on slow feeds
            import requests
            from io import BytesIO
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ForesightForgeBot/1.0; +https://example.com)'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Parse the RSS content
            feed = feedparser.parse(BytesIO(response.content))
        except Exception as e:
            # Log and continue so a single failing feed does not abort the entire pipeline.
            click.echo(f"⚠️  Failed to fetch {url}: {e}")
            continue
        for entry in feed.entries:
            eid = entry.get("id", entry.get("link"))
            if eid and eid not in seen:
                seen.add(eid)
                new_items.append({
                    "id": eid,
                    "title": entry.get("title"),
                    "link": entry.get("link"),
                    "published": entry.get("published", ""),
                })
    if new_items:
        date = datetime.date.today().isoformat()
        os.makedirs("raw", exist_ok=True)
        out = f"raw/{date}.json"
        with open(out, "w") as f:
            json.dump(new_items, f, indent=2)
        click.echo(f"Wrote {len(new_items)} new items to {out}")
    else:
        click.echo("No new items to ingest.")
    state["seen"] = list(seen)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


@cli.command()
def summarise():
    """Summarise today's raw items into bullet points."""
    date = datetime.date.today().isoformat()
    infile = f"raw/{date}.json"
    if not os.path.exists(infile):
        click.echo("No raw data found for today; skipping summarise.")
        return
    items = json.load(open(infile))
    # Limit to first 100 items to avoid context length issues
    items = items[:100]
    text = "\n".join(f"- {i['title']} ({i['link']})" for i in items)
    prompt = (
        "Condense the following items into concise bullet points capturing only the most important "
        "financial, economic, scientific, or geopolitical insights. Be succinct and avoid fluff:\n" + text
    )
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; skipping summarise.")
        return
    openai.api_key = key
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        click.echo(f"Error during summarise: {e}")
        return
    os.makedirs("summaries", exist_ok=True)
    out = f"summaries/{date}.md"
    with open(out, "w") as f:
        f.write(summary)
    click.echo(f"Wrote summary to {out}")


@cli.command()
def predict():
    """Generate predictions from today's summary with historical context."""
    date = datetime.date.today().isoformat()
    infile = f"summaries/{date}.md"
    if not os.path.exists(infile):
        click.echo("No summary found for today; skipping predict.")
        return
    summary = open(infile).read()
    
    # Get historical prediction context (last 7 days)
    historical_context = _get_prediction_history(7)
    
    prompt = (
        "You are an expert forecasting analyst. Below is your recent prediction performance, "
        "followed by today's news summary. Use this historical context to improve your predictions.\n\n"
        f"HISTORICAL PREDICTION PERFORMANCE (Last 7 days):\n{historical_context}\n\n"
        "TODAY'S NEWS SUMMARY:\n{summary}\n\n"
        "From the summary above, generate at least FIVE clear, testable financial, market, or economic "
        "predictions. Each prediction must include an explicit confidence percentage (e.g., 65%). "
        "Consider your historical performance when calibrating confidence levels. "
        "Format either as a numbered list or bullet list.\n"
    )
    
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; skipping predict.")
        return
    openai.api_key = key
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=300,
        )
        preds = resp.choices[0].message.content.strip()
    except Exception as e:
        click.echo(f"Error during predict: {e}")
        return

    # --- Persist structured prediction log ---------------------------------
    # Store the raw model output as well as a best-effort parsed structure so that
    # future jobs (e.g. an evaluator) can score accuracy without having to
    # re-parse the newsletters. We intentionally keep this simple JSON file next
    # to other artefacts to avoid introducing external storage.

    os.makedirs("predictions", exist_ok=True)
    pred_log_path = f"predictions/{date}.json"

    def _parse_predictions(text: str):  # noqa: E302
        """Return list[{text, confidence}] parsed from the model output."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        out = []
        for ln in lines:
            # strip leading bullets / numbering
            ln_clean = re.sub(r"^[-*\d+.\s]+", "", ln)
            # extract a confidence like "60%" or "60 %" (default None)
            m = re.search(r"(\d{1,3})\s*%", ln_clean)
            conf = int(m.group(1)) if m else None
            out.append({"text": ln_clean, "confidence": conf, "outcome": None})
        return out

    pred_entries = _parse_predictions(preds)
    with open(pred_log_path, "w") as fp:
        json.dump({"date": date, "predictions": pred_entries}, fp, indent=2)
    click.echo(f"Wrote structured predictions to {pred_log_path}")
    os.makedirs("newsletters", exist_ok=True)
    out = f"newsletters/{date}.md"
    with open(out, "w") as f:
        f.write(f"# Daily Newsletter {date}\n\n")
        f.write("## Summary\n\n")
        f.write(summary + "\n\n")
        f.write("## Predictions\n\n")
        f.write(preds + "\n")
    click.echo(f"Wrote newsletter to {out}")


def _get_prediction_history(days_back=7):
    """Get historical prediction performance for context."""
    today = datetime.date.today()
    history = []
    
    for i in range(1, days_back + 1):
        date = (today - datetime.timedelta(days=i)).isoformat()
        pred_file = f"predictions/{date}.json"
        review_file = f"reviews/{date}-review.md"
        
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
                predictions = pred_data.get('predictions', [])
                
                # Get review if available
                review_summary = ""
                if os.path.exists(review_file):
                    with open(review_file, 'r') as rf:
                        review_content = rf.read()
                        # Extract analysis section
                        if "## Analysis" in review_content:
                            analysis_start = review_content.find("## Analysis") + 11
                            analysis_end = review_content.find("##", analysis_start)
                            if analysis_end == -1:
                                analysis_end = len(review_content)
                            review_summary = review_content[analysis_start:analysis_end].strip()
                
                history.append({
                    'date': date,
                    'predictions': predictions,
                    'review': review_summary
                })
    
    if not history:
        return "No historical prediction data available."
    
    # Build context string
    context_parts = []
    for entry in history:
        pred_text = "\n".join([
            f"- {p['text']} (Confidence: {p.get('confidence', 'N/A')}%)"
            for p in entry['predictions']
        ])
        
        context_parts.append(f"Date: {entry['date']}")
        context_parts.append(f"Predictions:\n{pred_text}")
        if entry['review']:
            context_parts.append(f"Review: {entry['review'][:200]}...")
        context_parts.append("")
    
    return "\n".join(context_parts)


@cli.command()
def record():
    """Commit all changes for today's run to git, if in a repository."""
    try:
        repo = Repo(os.getcwd(), search_parent_directories=True)
    except Exception:
        click.echo("No git repository found; skipping record step.")
        return
    repo.git.add(all=True)
    date = datetime.date.today().isoformat()
    repo.index.commit(f"Daily update: {date}")
    click.echo("Recorded changes to git commit.")


def _add_prediction_updates_to_newsletter(date):
    """Add prediction updates section to today's newsletter."""
    newsletter_file = f"newsletters/{date}.md"
    if not os.path.exists(newsletter_file):
        return
    
    # Get recent predictions to check for updates
    today = datetime.date.today()
    updates = []
    
    # Check last 7 days of predictions
    for i in range(1, 8):
        pred_date = (today - datetime.timedelta(days=i)).isoformat()
        pred_file = f"predictions/{pred_date}.json"
        
        if os.path.exists(pred_file):
            with open(pred_file, 'r') as f:
                pred_data = json.load(f)
                predictions = pred_data.get('predictions', [])
                
                # Check if any predictions have outcomes marked
                for pred in predictions:
                    if pred.get('outcome') is not None:
                        updates.append({
                            'date': pred_date,
                            'prediction': pred['text'],
                            'confidence': pred.get('confidence'),
                            'outcome': pred['outcome']
                        })
    
    if not updates:
        return
    
    # Add updates section to newsletter
    with open(newsletter_file, 'r') as f:
        content = f.read()
    
    updates_section = "\n## Prediction Updates\n\n"
    for update in updates:
        status_emoji = "✅" if update['outcome'] == 'correct' else "❌" if update['outcome'] == 'incorrect' else "⏳"
        updates_section += f"{status_emoji} **{update['date']}**: {update['prediction']} (Confidence: {update['confidence']}%)\n\n"
    
    # Insert before the end of the file
    if "## Predictions" in content:
        # Insert after predictions section
        content = content.replace("## Predictions\n\n", "## Predictions\n\n" + updates_section)
    else:
        # Add at the end
        content += updates_section
    
    with open(newsletter_file, 'w') as f:
        f.write(content)
    
    click.echo(f"Added prediction updates to {newsletter_file}")


@cli.command()
@click.option("--date", default=None, help="Date (YYYY-MM-DD) to comment on; defaults to today.")
def comment(date):
    """Add a review comment or reply for a given date."""
    d = date or datetime.date.today().isoformat()
    infile = f"summaries/{d}.md"
    if not os.path.exists(infile):
        click.echo(f"No summary for {d}; cannot comment.")
        sys.exit(1)
    # collect the user's comment: either piped in via stdin or via editor
    if not sys.stdin.isatty():
        comment = sys.stdin.read().strip()
        if not comment:
            click.echo("No comment entered via stdin.")
            return
    else:
        initial = "# Enter your comment below. Lines starting with '#' are ignored.\n"
        text = click.edit(initial)
        if not text:
            click.echo("No comment entered.")
            return
        # strip out comment lines
        comment = "\n".join(l for l in text.splitlines() if not l.startswith("#")).strip()
        if not comment:
            click.echo("No comment entered.")
            return
    # prepare comment file and capture existing conversation for context
    os.makedirs("comments", exist_ok=True)
    out = f"comments/{d}.md"
    if os.path.exists(out):
        history = open(out).read().strip()
    else:
        history = ""
    # append the user's new comment
    with open(out, "a") as f:
        f.write(f"**Comment:** {comment}\n\n")
    click.echo(f"Appended comment to {out}")

    # attempt an AI-generated reply, including prior conversation and summary
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; skipping AI reply.")
        return
    openai.api_key = key
    summary_file = f"summaries/{d}.md"
    if not os.path.exists(summary_file):
        click.echo(f"No summary found at {summary_file}; cannot compose reply.")
        return
    summary = open(summary_file).read()

    # build the prompt with history if present
    intro = (
        "You are a helpful assistant continuing a discussion thread.\n"
        "Here is the prior conversation:\n" + history + "\n\n"
    ) if history else "You are a helpful assistant.\n"
    prompt = (
        f"{intro}Here is the newsletter summary for {d}:\n{summary}\n\n"
        f"A reader commented:\n{comment}\n\n"
        "Please draft a polite, constructive reply to this comment, referencing any earlier points as needed."
    )
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=200,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        click.echo(f"Error generating AI reply: {e}")
        return
    # append the assistant's reply
    with open(out, "a") as f:
        f.write(f"**Reply:** {reply}\n\n")
    click.echo(f"Appended AI reply to {out}")


@cli.command()
def review():
    """Review previous predictions against current news and provide insights."""
    # Find the most recent prediction file
    pred_files = sorted(glob.glob('predictions/*.json'))
    if not pred_files:
        click.echo("No prediction files found; skipping review.")
        return
    
    # Get yesterday's predictions
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    yesterday_pred_file = f'predictions/{yesterday}.json'
    
    if not os.path.exists(yesterday_pred_file):
        click.echo(f"No predictions found for {yesterday}; skipping review.")
        return
    
    # Load yesterday's predictions
    with open(yesterday_pred_file, 'r') as f:
        pred_data = json.load(f)
    
    predictions = pred_data.get('predictions', [])
    if not predictions:
        click.echo("No predictions found in yesterday's file.")
        return
    
    # Get today's news for context
    today = datetime.date.today().isoformat()
    today_raw_file = f'raw/{today}.json'
    
    if not os.path.exists(today_raw_file):
        click.echo(f"No raw data found for {today}; will review without current context.")
        today_news = []
    else:
        with open(today_raw_file, 'r') as f:
            today_news = json.load(f)
    
    # Build review prompt
    pred_text = "\n".join([
        f"- {p['text']} (Confidence: {p.get('confidence', 'N/A')}%)"
        for p in predictions
    ])
    
    news_text = "\n".join([
        f"- {item['title']} ({item['link']})"
        for item in today_news[:50]  # Limit to first 50 items
    ])
    
    prompt = (
        "You are an expert forecasting analyst reviewing yesterday's predictions against today's news.\n\n"
        f"YESTERDAY'S PREDICTIONS:\n{pred_text}\n\n"
        f"TODAY'S NEWS CONTEXT:\n{news_text}\n\n"
        "Please provide a brief analysis (2-3 paragraphs) covering:\n"
        "1. Which predictions seem to be playing out or gaining evidence\n"
        "2. Which predictions appear to be off-track or missing key factors\n"
        "3. Overall assessment of prediction quality and confidence calibration\n"
        "4. Any patterns or insights that could improve future predictions\n\n"
        "Be constructive and specific. Focus on learning opportunities."
    )
    
    # Get LLM review
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; skipping review.")
        return
    
    openai.api_key = key
    
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=600,
        )
        review_text = resp.choices[0].message.content.strip()
        
        # Save the review
        os.makedirs('reviews', exist_ok=True)
        review_file = f'reviews/{yesterday}-review.md'
        with open(review_file, 'w') as f:
            f.write(f"# Prediction Review: {yesterday}\n\n")
            f.write(f"## Predictions Reviewed\n\n")
            f.write(pred_text + "\n\n")
            f.write(f"## Analysis\n\n")
            f.write(review_text + "\n\n")
            f.write(f"## News Context\n\n")
            f.write(f"Based on {len(today_news)} news items from {today}\n")
        
        click.echo(f"Wrote prediction review to {review_file}")
        
        # Also append to a running log
        log_file = 'reviews/prediction-review-log.md'
        with open(log_file, 'a') as f:
            f.write(f"\n## {yesterday}\n\n")
            f.write(review_text + "\n\n")
            f.write("---\n\n")
        
        click.echo(f"Updated review log at {log_file}")
        
    except Exception as e:
        click.echo(f"Error during prediction review: {e}")


@cli.command()
def run_daily():
    """Run the full daily pipeline (ingest → summarise → predict → review → record)."""
    ingest.callback()
    summarise.callback()
    predict.callback()
    review.callback()  # Review yesterday's predictions
    
    # Add prediction updates to newsletter
    date = datetime.date.today().isoformat()
    _add_prediction_updates_to_newsletter(date)
    
    dashboard.callback()
    record.callback()


@cli.command()
@click.option("--date", default=None, help="Date (YYYY-MM-DD) to mark outcomes for; defaults to yesterday.")
@click.option("--prediction-index", type=int, required=True, help="Index of prediction to mark (0-based).")
@click.option("--outcome", type=click.Choice(['correct', 'incorrect', 'pending']), required=True, help="Outcome of the prediction.")
def mark_outcome(date, prediction_index, outcome):
    """Mark the outcome of a specific prediction."""
    if date is None:
        date = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    
    pred_file = f"predictions/{date}.json"
    if not os.path.exists(pred_file):
        click.echo(f"No predictions found for {date}")
        return
    
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    
    predictions = pred_data.get('predictions', [])
    if prediction_index >= len(predictions):
        click.echo(f"Prediction index {prediction_index} out of range (max: {len(predictions) - 1})")
        return
    
    # Mark the outcome
    predictions[prediction_index]['outcome'] = outcome
    
    # Save updated predictions
    with open(pred_file, 'w') as f:
        json.dump(pred_data, f, indent=2)
    
    prediction_text = predictions[prediction_index]['text']
    click.echo(f"Marked prediction '{prediction_text[:50]}...' as {outcome}")


@cli.command()
def discover(since_days):
    """Scan recent entries and propose new source domains into discover/ folder."""
    today = datetime.date.today()
    cutoff = today - datetime.timedelta(days=since_days)
    # existing sources to avoid duplicates
    existing = {re.sub(r"https?://", "", s).strip('/') for s in load_sources()}
    candidates = set()
    for path in glob.glob("raw/*.json"):
        try:
            date_str = os.path.splitext(os.path.basename(path))[0]
            d = datetime.date.fromisoformat(date_str)
        except Exception:
            continue
        if d < cutoff:
            continue
        for item in json.load(open(path)):
            link = item.get('link', '')
            # extract domain
            m = re.match(r'https?://([^/]+)', link)
            if m:
                domain = m.group(1)
                if domain not in existing:
                    candidates.add(domain)
    if not candidates:
        click.echo("No new candidate sources found.")
        return

    # Use an LLM to vet the candidates
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; skipping discover.")
        return
    openai.api_key = key

    prompt = (
        "You are an expert analyst for Foresight Forge, a project that forecasts financial, economic, "
        "scientific, and geopolitical trends. Your task is to review the following list of potential "
        "new RSS feed domains and select ONLY the ones that are HIGHLY USEFUL AND INTERESTING for forecasting. "
        "Be EXTREMELY selective - only approve domains that are:\n"
        "1. Major established news outlets with high editorial standards\n"
        "2. Respected financial/economics publications\n"
        "3. Well-known scientific journals or research institutions\n"
        "4. High-quality blogs by recognized experts in relevant fields\n"
        "5. Government or international organization sources\n\n"
        "Reject personal blogs, small websites, or any domains that don't clearly meet these standards. "
        "Look for sources that provide unique insights, data, or analysis that would be valuable for forecasting.\n\n"
        "Format your response as a simple list of approved domains, one per line, like '- example.com/rss'. "
        "Do not include justifications or any other text. If no domains meet the high standards, respond with 'NONE'.\n\n"
        "Candidate Domains:\n" + "\n".join(sorted(list(candidates)))
    )

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=500,
        )
        approved_text = resp.choices[0].message.content.strip()
        # Parse the response to get the final list of URLs
        approved_urls = [line.lstrip('- ').strip() for line in approved_text.split('\n') if line.strip()]

    except Exception as e:
        click.echo(f"Error during discover's AI vetting: {e}")
        # Fallback to the old behavior in case of an error
        approved_urls = [f"https://{dom}/rss" for dom in sorted(candidates)]


    if not approved_urls:
        click.echo("AI vetting resulted in no new approved sources.")
        return

    os.makedirs('discover', exist_ok=True)
    out_file = f"discover/{today.isoformat()}-candidates.md"
    with open(out_file, 'w') as f:
        f.write("# Candidate sources (AI-vetted)\n")
        for url in approved_urls:
            f.write(f"- {url}\n")
    click.echo(f"Wrote {len(approved_urls)} AI-vetted candidates to {out_file}")


@cli.command()
@click.option('--pr', is_flag=True, help='Create a pull request instead of merging to main.')
def self_update(pr):
    """Apply brain's source decisions (additions/removals) to sources.yaml."""
    # Get brain's decisions
    decision = _brain_decision()
    to_add = decision.get('add_sources', [])
    to_remove = decision.get('remove_sources', [])
    
    if not to_add and not to_remove:
        click.echo('No source changes proposed by brain.')
        return
    
    sources = load_sources()
    
    # Apply removals first
    if to_remove:
        sources = [s for s in sources if s not in to_remove]
        click.echo(f'Removed {len(to_remove)} sources: {", ".join(to_remove)}')
    
    # Apply additions
    if to_add:
        sources.extend(to_add)
        click.echo(f'Added {len(to_add)} sources: {", ".join(to_add)}')
    
    if not to_add and not to_remove:
        click.echo('No changes to apply.')
        return
    with open('sources.yaml', 'w') as f:
        yaml.safe_dump(sources, f, sort_keys=False)
    
    # Commit changes directly to main
    try:
        repo = Repo(os.getcwd(), search_parent_directories=True)
        repo.git.add('sources.yaml')
        date = datetime.date.today().isoformat()
        repo.index.commit(f'Brain auto-update sources: {date}')
        click.echo(f'Applied brain\'s source changes: {len(to_add)} added, {len(to_remove)} removed')
    except Exception as e:
        click.echo(f'Applied changes to sources.yaml but git commit failed: {e}')


@cli.command()
def dashboard():
    """Generate a simple HTML dashboard listing all logged predictions by date."""
    nl_files = sorted(glob.glob('newsletters/*.md'))
    if not nl_files:
        click.echo('No newsletter found; skipping dashboard.')
        return

    def _parse_predictions_md(md_text):  # noqa: E302
        """Extract predictions lines from newsletter Markdown -> list[str]."""
        lines = md_text.splitlines()
        out = []
        # find heading '## Predictions'
        inside = False
        for ln in lines:
            if ln.strip().lower().startswith('## predictions'):
                inside = True
                continue
            if inside:
                if ln.startswith('#'):  # next heading -> stop
                    break
                if ln.strip():
                    out.append(ln.strip())
        return out

    sections = []
    for nl_path in reversed(nl_files):  # newest first
        date = os.path.basename(nl_path).replace('.md', '')
        # try JSON first
        json_path = f'predictions/{date}.json'
        if os.path.exists(json_path):
            data = json.load(open(json_path))
            preds_data = data.get('predictions', [])
        else:
            # parse from MD
            preds_lines = _parse_predictions_md(open(nl_path).read())
            preds_data = []
            current = None
            for ln in preds_lines:
                ln = ln.strip()
                if not ln:
                    continue
                if 'prediction:' in ln.lower():
                    # start new prediction entry
                    text = re.sub(r'^[-*\d+.\s]+', '', ln)
                    current = {'text': text, 'confidence': None}
                    preds_data.append(current)
                elif 'confidence' in ln.lower() and current is not None:
                    m = re.search(r'(\d{1,3})\s*%?', ln)
                    if m:
                        current['confidence'] = int(m.group(1))
                else:
                    # fallback: treat as standalone line
                    clean = re.sub(r'^[-*\d+.\s]+', '', ln)
                    m = re.search(r'(\d{1,3})\s*%?', clean)
                    conf = int(m.group(1)) if m else None
                    preds_data.append({'text': clean, 'confidence': conf})

        # Build HTML list for predictions
        preds_html = '<ul>' + ''.join(
            f"<li>{p['text']}" + (f" — {p['confidence']}%" if p.get('confidence') is not None else '') + '</li>'
            for p in preds_data) + '</ul>'

        # stats: number of raw items ingested
        raw_path = f'raw/{date}.json'
        n_items = len(json.load(open(raw_path))) if os.path.exists(raw_path) else '--'

        # ensure the newsletter Markdown is available under docs/ for direct linking
        dst_dir = os.path.join('docs', 'newsletters')
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, os.path.basename(nl_path))
        if not os.path.exists(dst_path):
            shutil.copyfile(nl_path, dst_path)

        nl_link = f"<a href='newsletters/{os.path.basename(nl_path)}'>newsletter</a>"
        sections.append(f"<h2>{date} — {n_items} items — {nl_link}</h2>\n{preds_html}")

    body_content = "<hr>\n".join(sections)
    html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head><meta charset='utf-8'><title>Foresight Forge - Prediction Log</title></head>\n"
        "<body>\n"
        "<h1>Foresight Forge - Prediction Log</h1>\n"
        f"{body_content}\n"
        "</body>\n"
        "</html>\n"
    )

    os.makedirs('docs', exist_ok=True)
    out = 'docs/index.html'
    with open(out, 'w') as f:
        f.write(html)
    click.echo(f'Wrote dashboard to {out}')


@cli.command()
def cleanup_sources():
    """Manually trigger source cleanup and optimization."""
    sources = load_sources()
    click.echo(f"Current sources: {len(sources)}")
    
    to_add, to_remove = _brain_source_review(sources)
    
    if to_remove:
        click.echo(f"\nSources to remove ({len(to_remove)}):")
        for source in to_remove:
            click.echo(f"  - {source}")
    else:
        click.echo("\nNo sources to remove.")
    
    if to_add:
        click.echo(f"\nSources to add ({len(to_add)}):")
        for source in to_add:
            click.echo(f"  + {source}")
    else:
        click.echo("\nNo sources to add.")
    
    # Apply changes if any
    if to_remove or to_add:
        click.echo("\nApplying changes...")
        try:
            ctx = click.get_current_context()
            ctx.invoke(self_update, pr=False)  # Direct merge, no PR
            click.echo("Source cleanup completed successfully!")
        except Exception as e:
            click.echo(f"Failed to apply source changes: {e}")
    else:
        click.echo("\nNo changes needed.")





if __name__ == "__main__":
    # load environment variables from .env if present
    # load environment variables from .env, override any existing values
    load_dotenv(override=True)
    cli()
