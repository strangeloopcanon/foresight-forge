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
import openai
from openai import OpenAI, OpenAIError
from git import Repo
import glob
import re
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
        feed = feedparser.parse(url)
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
        click.echo("No raw data found for today; run ingest first.")
        sys.exit(1)
    items = json.load(open(infile))
    text = "\n".join(f"- {i['title']} ({i['link']})" for i in items)
    prompt = (
        "Condense the following items into ≤10 clear bullet points:\n" + text
    )
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; cannot summarise.")
        sys.exit(1)
    openai.api_key = key
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        click.echo(f"Error during summarise: {e}")
        sys.exit(1)
    os.makedirs("summaries", exist_ok=True)
    out = f"summaries/{date}.md"
    with open(out, "w") as f:
        f.write(summary)
    click.echo(f"Wrote summary to {out}")


@cli.command()
def predict():
    """Generate predictions from today's summary."""
    date = datetime.date.today().isoformat()
    infile = f"summaries/{date}.md"
    if not os.path.exists(infile):
        click.echo("No summary found for today; run summarise first.")
        sys.exit(1)
    summary = open(infile).read()
    prompt = (
        "From the summary below, generate at least three testable predictions with explicit"
        " confidence levels (as percentages):\n" + summary
    )
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; cannot predict.")
        sys.exit(1)
    openai.api_key = key
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        preds = resp.choices[0].message.content.strip()
    except Exception as e:
        click.echo(f"Error during predict: {e}")
        sys.exit(1)
    os.makedirs("newsletters", exist_ok=True)
    out = f"newsletters/{date}.md"
    with open(out, "w") as f:
        f.write(f"# Daily Newsletter {date}\n\n")
        f.write("## Summary\n\n")
        f.write(summary + "\n\n")
        f.write("## Predictions\n\n")
        f.write(preds + "\n")
    click.echo(f"Wrote newsletter to {out}")


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


@cli.command()
@click.option("--date", default=None, help="Date (YYYY-MM-DD) to comment on; defaults to today.")
def comment(date):
    """Add a review comment or reply for a given date."""
    d = date or datetime.date.today().isoformat()
    infile = f"summaries/{d}.md"
    if not os.path.exists(infile):
        click.echo(f"No summary for {d}; cannot comment.")
        sys.exit(1)
    text = click.edit("# Enter your comment below. Lines starting with '#' are ignored.\n")
    if not text:
        click.echo("No comment entered.")
        return
    # remove commented lines
    comment = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    os.makedirs("comments", exist_ok=True)
    out = f"comments/{d}.md"
    with open(out, "a") as f:
        f.write(f"{comment}\n")
    click.echo(f"Appended comment to {out}")


@cli.command()
def run_daily():
    """Run the full daily pipeline (ingest → summarise → predict → record)."""
    ingest.callback()
    summarise.callback()
    predict.callback()
    dashboard.callback()
    record.callback()


@cli.command()
@click.option("--since-days", default=7, help="How many days of raw data to scan for candidates.")
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
    os.makedirs('discover', exist_ok=True)
    out_file = f"discover/{today.isoformat()}-candidates.md"
    with open(out_file, 'w') as f:
        f.write("# Candidate sources\n")
        for dom in sorted(candidates):
            f.write(f"- https://{dom}/rss\n")
    click.echo(f"Wrote {len(candidates)} candidates to {out_file}")


@cli.command()
@click.option('--pr', is_flag=True, help='Create a pull request instead of merging to main.')
def self_update(pr):
    """Merge candidates from discover/ into sources.yaml and optionally open a PR."""
    # load candidates from latest file
    files = sorted(glob.glob('discover/*-candidates.md'))
    if not files:
        click.echo('No candidate file found; run discover first.')
        sys.exit(1)
    latest = files[-1]
    lines = [l.strip() for l in open(latest) if l.startswith('-')]
    new_urls = [l.lstrip('- ').strip() for l in lines]
    if not new_urls:
        click.echo('No URLs in candidate file to add.')
        return
    sources = load_sources()
    to_add = [u for u in new_urls if u not in sources]
    if not to_add:
        click.echo('No new URLs to merge into sources.yaml.')
        return
    sources.extend(to_add)
    with open('sources.yaml', 'w') as f:
        yaml.safe_dump(sources, f, sort_keys=False)
    repo = Repo(os.getcwd(), search_parent_directories=True)
    date = datetime.date.today().isoformat()
    branch = f'auto/update-sources-{date}'
    # create or reuse branch for source updates
    existing_branches = [b.name for b in repo.branches]
    if branch in existing_branches:
        repo.git.checkout(branch)
    else:
        repo.git.checkout('-b', branch)
    repo.git.add('sources.yaml')
    repo.index.commit(f'Auto-update sources: {date}')
    repo.git.push('--set-upstream', 'origin', branch)
    click.echo(f'Merged {len(to_add)} new sources into sources.yaml on branch {branch}')
    if pr:
        # create GitHub pull request
        url = repo.remotes.origin.url
        m = re.search(r'github\.com[:/](.+)/(.+)(?:\.git)?$', url)
        if not m:
            click.echo('Cannot parse GitHub repo URL; PR not created.')
            return
        owner, name = m.group(1), m.group(2)
        body = {
            'title': f'Auto-update sources list ({date})',
            'head': branch,
            'base': 'main',
            'body': f'Merging {len(to_add)} new candidate sources from {latest}.',
        }
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            click.echo('GITHUB_TOKEN not set; cannot create PR.')
            return
        resp = requests.post(
            f'https://api.github.com/repos/{owner}/{name}/pulls',
            json=body,
            headers={'Authorization': f'token {token}'},
        )
        if resp.status_code == 201:
            pr_url = resp.json().get('html_url')
            click.echo(f'Pull request created: {pr_url}')
        else:
            click.echo(f'Failed to create PR: {resp.status_code} {resp.text}')


@cli.command()
def dashboard():
    """Generate a minimal dashboard HTML displaying the latest newsletter."""
    files = sorted(glob.glob('newsletters/*.md'))
    if not files:
        click.echo('No newsletter found; run predict first.')
        sys.exit(1)
    latest = files[-1]
    content = open(latest).read()
    os.makedirs('docs', exist_ok=True)
    out = 'docs/index.html'
    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Foresight Forge Dashboard</title></head>
<body>
  <h1>Latest Newsletter: {os.path.basename(latest).replace('.md','')}</h1>
  <pre>
{content}
  </pre>
  <hr><p><a href="../newsletters">Full history of newsletters</a></p>
</body>
</html>
"""
    with open(out, 'w') as f:
        f.write(html)
    click.echo(f'Wrote dashboard to {out}')


if __name__ == "__main__":
    # load environment variables from .env if present
    # load environment variables from .env, override any existing values
    load_dotenv(override=True)
    cli()
