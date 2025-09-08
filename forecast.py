#!/usr/bin/env python3
"""
Main CLI for the Foresight Forge daily forecasting pipeline.
"""
import os
import json
import sys
import datetime
import uuid
import hashlib

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

DEFAULT_LLM_MODEL = os.getenv("FORESIGHT_LLM_MODEL", "gpt-5")
# GPT-5 reasoning models only support default temperature=1.0; set 1 by default
DEFAULT_TEMPERATURE = float(os.getenv("FORESIGHT_LLM_TEMPERATURE", "1"))
from git import Repo
import glob
import re
import shutil
from dotenv import load_dotenv
try:
    import tiktoken  # optional, used for token estimation
    _TIKTOKEN_ENC = tiktoken.get_encoding('cl100k_base')
except Exception:  # pragma: no cover
    _TIKTOKEN_ENC = None


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

def _sanitize_openai_env():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        clean = key.strip().strip('"').strip("'")
        if clean != key:
            os.environ["OPENAI_API_KEY"] = clean


CONFIG_PATH = os.getenv("FORESIGHT_CONFIG", "foresight.config.yaml")
_STRICT_DEFAULTS = {
    'summarise': True,
    'predict': True,
    'review': True,
    'vet': False,
    'sources': False,
}
_CONFIG_CACHE = None


def _load_config():
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    cfg = {}
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r') as f:
                data = yaml.safe_load(f) or {}
                if isinstance(data, dict):
                    cfg = data
    except Exception:
        cfg = {}
    _CONFIG_CACHE = cfg
    return _CONFIG_CACHE


def _strict_for(step: str) -> bool:
    cfg = _load_config()
    strict = (cfg.get('strict') or {}) if isinstance(cfg.get('strict'), dict) else {}
    if step in strict and isinstance(strict[step], bool):
        return strict[step]
    return _STRICT_DEFAULTS.get(step, False)


def _looks_like_feed_url(url: str) -> bool:
    """Heuristic: does this URL look like an RSS/Atom endpoint?"""
    if not isinstance(url, str):
        return False
    u = url.lower()
    # Common feed patterns
    if u.endswith('.rss') or u.endswith('.xml'):
        return True
    if '/rss' in u or '/feeds/' in u or '/feed/' in u:
        return True
    if 'format=rss' in u:
        return True
    return False


def _probe_is_feed(url: str, timeout: int = 10) -> bool:
    """Try fetching the URL and detect RSS/Atom by content-type or signature.

    Returns True if content-type indicates XML/Atom/RSS or the payload contains
    typical tags (<rss, <feed, <?xml), else False. Swallows errors.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; ForesightForgeBot/1.0; +https://example.com)'
        }
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        # Some feeds respond 200 only to GET (HEAD may 403/405)
        ct = (resp.headers.get('content-type') or '').lower()
        if 'xml' in ct or 'atom' in ct or 'rss' in ct:
            return True
        # Peek first bytes
        chunk = next(resp.iter_content(chunk_size=2048), b'')
        sniff = chunk.decode('utf-8', errors='ignore').strip()[:2048]
        if sniff.startswith('<?xml') or '<rss' in sniff or '<feed' in sniff or '<rdf' in sniff:
            return True
    except Exception:
        return False
    finally:
        try:
            resp.close()
        except Exception:
            pass
    return False


def _filter_feed_urls(urls):
    """Return only URLs that are likely valid RSS/Atom feeds.

    Strategy: accept if heuristic matches OR probe confirms. This balances
    speed and robustness while avoiding HTML calendars.
    """
    out = []
    for u in urls or []:
        if _looks_like_feed_url(u) or _probe_is_feed(u):
            out.append(u)
    return out


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for prompt sizing.

    Uses tiktoken if available (cl100k_base), else a 4 chars/token heuristic.
    """
    if not text:
        return 0
    try:
        if _TIKTOKEN_ENC is not None:
            return len(_TIKTOKEN_ENC.encode(text))
    except Exception:
        pass
    return max(1, len(text) // 4)


def _summary_json_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "summary_schema",
            "schema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "bullets": {
                        "type": "array",
                        "minItems": 8,
                        "maxItems": 15,
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "minLength": 1, "maxLength": 180},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string", "enum": [
                                        "macro", "markets", "rates", "energy", "tech", "geopolitics", "science", "other"
                                    ]}
                                },
                                "link": {"type": "string"}
                            },
                            "required": ["text", "tags"],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["bullets"],
                "additionalProperties": False
            },
            "strict": True
        }
    }


def _predict_json_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "predict_schema",
            "schema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "predictions": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 15,
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "minLength": 5, "maxLength": 500},
                                "category": {"type": "string", "enum": [
                                    "macro", "markets", "rates", "energy", "tech", "geopolitics", "other"
                                ]},
                                "horizon_days": {"type": "integer", "minimum": 1, "maximum": 3650},
                                "deadline": {"type": "string"},
                                "confidence_pct": {"type": "integer", "minimum": 0, "maximum": 100},
                                "log_odds": {"type": "number"},
                                "verification_criteria": {"type": "string", "minLength": 5},
                                "evidence": {"type": "array", "items": {"type": "string"}},
                                "monitoring_signals": {"type": "array", "items": {"type": "string"}},
                                "supersedes_id": {"type": "string"},
                                "rationale": {"type": "string"}
                            },
                            "required": [
                                "text", "category", "horizon_days", "deadline", "confidence_pct",
                                "verification_criteria", "evidence", "monitoring_signals"
                            ],
                            "additionalProperties": False
                        }
                    }
                },
                "required": ["predictions"],
                "additionalProperties": False
            },
            "strict": True
        }
    }


def _review_json_schema():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "review_schema",
            "schema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "assessments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "status": {"type": "string", "enum": [
                                    "correct", "incorrect", "pending", "needs-clarification"
                                ]},
                                "status_rationale": {"type": "string"}
                            },
                            "required": ["id", "status"],
                            "additionalProperties": False
                        }
                    },
                    "analysis": {"type": "string"}
                },
                "required": ["assessments", "analysis"],
                "additionalProperties": False
            },
            "strict": True
        }
    }


def _grammar_vet_list():
    return {
        "type": "grammar",
        "grammar": r"""
start: NONE | list
list: item+
item: "- " TEXT NEWLINE?
NONE: "NONE"
TEXT: /[^\n]+/
NEWLINE: /\n/
%ignore /[\t ]+/
"""
    }


def _grammar_remove_add():
    return {
        "type": "grammar",
        "grammar": r"""
start: "REMOVE:" NEWLINE items "ADD:" NEWLINE items
items: ("- " TEXT NEWLINE)*
TEXT: /[^\n]+/
NEWLINE: /\n/
%ignore /[\t ]+/
"""
    }


def _llm_respond(
    prompt,
    max_tokens,
    *,
    system=None,
    response_format=None,  # "json" string, or response_format dict for schemas/grammar
    temperature=None,
):
    """Unified LLM call with caching, retries, system message and JSON mode.

    - Uses Responses API with medium reasoning and optional JSON output.
    - Falls back to Chat Completions.
    - Caches by hash of inputs under .cache/llm/.
    """
    _sanitize_openai_env()
    client = OpenAI()
    temp = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
    model = DEFAULT_LLM_MODEL
    is_gpt5 = str(model).startswith("gpt-5")

    # Cache key
    cache_dir = os.path.join('.cache', 'llm')
    os.makedirs(cache_dir, exist_ok=True)
    key_raw = json.dumps({
        'model': model,
        'prompt': prompt,
        'system': system or '',
        'max_tokens': max_tokens,
        'response_format': response_format or 'text',
        'temperature': temp,
    }, sort_keys=True).encode('utf-8')
    key = hashlib.sha256(key_raw).hexdigest()
    cache_path = os.path.join(cache_dir, key + ('.json' if response_format == 'json' else '.txt'))
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return f.read()

    last_err = None
    for attempt in range(3):
        try:
            # Prefer Responses API
            base_kwargs = {
                'model': model,
                'reasoning': {"effort": "medium"},
            }
            if not is_gpt5:
                base_kwargs['max_output_tokens'] = max_tokens
            # GPT-5 reasoning models often disallow non-default temperature; omit for GPT-5
            if not is_gpt5 and temp is not None:
                base_kwargs['temperature'] = temp
            if system:
                base_kwargs['instructions'] = system
            base_kwargs['input'] = prompt

            kwargs = dict(base_kwargs)
            added_json_mode = False
            if isinstance(response_format, dict):
                kwargs['response_format'] = response_format
                added_json_mode = True
            elif response_format == 'json':
                kwargs['response_format'] = {"type": "json_object"}
                added_json_mode = True

            try:
                resp = client.responses.create(**kwargs)
            except TypeError as te:
                if added_json_mode and 'response_format' in str(te):
                    # Retry without response_format for older SDK compatibility
                    resp = client.responses.create(**base_kwargs)
                else:
                    raise
            text = getattr(resp, "output_text", None)
            # Broader extraction across possible SDK shapes
            if not text:
                try:
                    # Try common nested path: output[*].content[*].text
                    parts = []
                    for item in getattr(resp, "output", []) or []:
                        for c in getattr(item, "content", []) or []:
                            t = getattr(c, "text", None)
                            if isinstance(t, str):
                                parts.append(t)
                    if parts:
                        text = "\n".join(parts).strip()
                except Exception:
                    pass
            if text:
                with open(cache_path, 'w') as f:
                    f.write(text)
                return text
            else:
                last_err = RuntimeError('Empty model output')
        except Exception as e:
            last_err = e
        # Fallback to Chat Completions (avoid for GPT-5; use Responses API only)
        if is_gpt5:
            continue
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=temp,
            )
            text = resp.choices[0].message.content.strip()
            if text:
                with open(cache_path, 'w') as f:
                    f.write(text)
                return text
        except Exception as e:
            last_err = e
        # backoff
        import time
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM request failed after retries: {last_err}")

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
        system = (
            "You are Foresight Forge. Output exactly as specified: a plain list of '- domain', or 'NONE'."
        )
        rf = _grammar_vet_list() if _strict_for('vet') else None
        approved_text = _llm_respond(prompt, max_tokens=500, system=system, response_format=rf)
        
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
        system = (
            "You are Foresight Forge. Output exactly with 'REMOVE:' and 'ADD:' sections as specified; no extra text."
        )
        rf = _grammar_remove_add() if _strict_for('sources') else None
        review_text = _llm_respond(prompt, max_tokens=800, system=system, response_format=rf)
        
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
@click.option("--date", "date_opt", default=None, help="Date (YYYY-MM-DD) to summarise; defaults to today.")
def summarise(date_opt=None):
    """Summarise ALL raw items for a date into bullets via chunked map-reduce with priority feeds."""
    date = date_opt or datetime.date.today().isoformat()
    infile = f"raw/{date}.json"
    if not os.path.exists(infile):
        click.echo("No raw data found for today; skipping summarise.")
        return
    items = json.load(open(infile))

    # Prioritise key economic indicator feeds, then keep remaining in original order
    priority_domains = [
        'bls.gov',
        'apps.bea.gov', 'bea.gov',
        'census.gov',
        'federalreserve.gov',
        'eia.gov',
        'conference-board.org',
        'home.treasury.gov',
    ]

    def _score(link: str) -> int:
        try:
            u = (link or '').lower()
        except Exception:
            return 0
        score = 0
        # Strong priority for PFEI sources
        if 'bls.gov' in u or 'apps.bea.gov' in u or 'federalreserve.gov' in u:
            score += 3
        if 'census.gov' in u or 'eia.gov' in u or 'conference-board.org' in u or 'home.treasury.gov' in u:
            score += 2
        return score

    items = sorted(items, key=lambda i: _score(i.get('link')), reverse=True)

    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; skipping summarise.")
        return
    openai.api_key = key

    # Chunking parameters (env-tunable)
    chunk_size_default = int(os.getenv('FORESIGHT_SUMMARISE_CHUNK_SIZE', '80'))
    max_merge_tokens = int(os.getenv('FORESIGHT_SUMMARISE_MAX_TOKENS', '900'))
    prompt_limit = int(os.getenv('FORESIGHT_SUMMARISE_PROMPT_LIMIT', '90000'))  # tokens

    def _summarise_block(block_items):
        text = "\n".join(f"- {i.get('title')} ({i.get('link')})" for i in block_items)
        prompt = (
            "Condense the following items into concise bullet points capturing only the most important "
            "financial, economic, scientific, or geopolitical insights. Return ONLY JSON per the schema.\n\n"
            "Items:\n" + text + "\n\n"
            "Schema: {\n"
            "  \"bullets\": [\n"
            "    {\n"
            "      \"text\": string,\n"
            "      \"tags\": array of strings from [\"macro\", \"markets\", \"rates\", \"energy\", \"tech\", \"geopolitics\", \"science\", \"other\"],\n"
            "      \"link\": string (optional)\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Constraints: 8-15 bullets; each bullet max 25 words; no commentary outside JSON.\n"
        )
        rf = _summary_json_schema() if _strict_for('summarise') else 'json'
        summary_json_text = _llm_respond(prompt, max_tokens=600, system=(
            "You are Foresight Forge. Output exactly one JSON object matching the schema. "
            "Think step-by-step internally but DO NOT expose chain-of-thought; only output the final JSON."
        ), response_format=rf)
        return json.loads(summary_json_text).get('bullets', [])

    try:
        # Estimate tokens for all items to decide single-pass or chunking
        all_text = "\n".join(f"- {i.get('title')} ({i.get('link')})" for i in items)
        total_tokens = _estimate_tokens(all_text)
        # Single pass if within prompt limit; otherwise dynamic chunking
        if total_tokens <= prompt_limit:
            click.echo(f"Summarising {len(items)} items in 1 chunk (~{total_tokens} toks)")
            bullets = _summarise_block(items)
        else:
            # Determine items per chunk based on token estimate with a safety margin
            avg_per_item = max(1, total_tokens // max(1, len(items)))
            items_per_chunk = max(40, int((prompt_limit * 0.8) / avg_per_item))
            chunk_size = max(40, min(items_per_chunk, chunk_size_default))
            click.echo(f"Summarising {len(items)} items in chunks of {chunk_size} (~{avg_per_item} toks/item)")
            # Map: summarise each chunk
            all_bullets = []
            for i in range(0, len(items), chunk_size):
                blk = items[i:i+chunk_size]
                try:
                    all_bullets.extend(_summarise_block(blk))
                except Exception as e:
                    click.echo(f"Chunk summarise failed at [{i}:{i+chunk_size}]: {e}")
            # Deduplicate by text
            seen = set()
            dedup = []
            for b in all_bullets:
                t = (b.get('text') or '').strip().lower()
                if t and t not in seen:
                    seen.add(t)
                    dedup.append(b)
            # Reduce: ask the model to merge and pick the most important 10-15 bullets
            merge_input = {
                'bullets': dedup
            }
            merge_prompt = (
                "You are merging multiple partial summaries into a final concise summary. "
                "From the provided bullets (with optional links), select and rewrite the most important items, "
                "avoid duplication, and produce ONLY JSON per the schema."
            )
            rf = _summary_json_schema() if _strict_for('summarise') else 'json'
            merged_text = _llm_respond(
                json.dumps(merge_input),
                max_tokens=max_merge_tokens,
                system=(
                    "You are Foresight Forge. Output exactly one JSON object matching the schema. "
                    "Target 10-15 bullets."
                ),
                response_format=rf,
            )
            bullets = json.loads(merged_text).get('bullets', [])

        # Persist structured JSON
        os.makedirs("summaries", exist_ok=True)
        with open(f"summaries/{date}.json", "w") as jf:
            json.dump({'bullets': bullets}, jf, indent=2)

        # Render to markdown
        lines = []
        for b in bullets:
            tags = b.get('tags') or []
            tag_prefix = f"[{'/'.join(tags)}] " if tags else ""
            link = b.get('link')
            t = b.get('text', '').strip()
            if link:
                lines.append(f"- {tag_prefix}{t} ({link})")
            else:
                lines.append(f"- {tag_prefix}{t}")
        summary_md = "\n".join(lines)
    except Exception as e:
        click.echo(f"Error during structured summarise; falling back to plain text: {e}")
        # Fallback to a simple text summary over ALL items
        text = "\n".join(f"- {i.get('title')} ({i.get('link')})" for i in items)
        try:
            summary_md = _llm_respond(
                "Condense the following items into concise bullets. Be succinct, avoid fluff:\n" + text,
                max_tokens=400,
                system="You are Foresight Forge. Return bullets only.")
        except Exception as e2:
            click.echo(f"Error during summarise fallback: {e2}")
            return

    os.makedirs("summaries", exist_ok=True)
    out = f"summaries/{date}.md"
    with open(out, "w") as f:
        f.write(summary_md)
    click.echo(f"Wrote summary to {out}")


@cli.command()
@click.option("--date", "date_opt", default=None, help="Date (YYYY-MM-DD) to predict for; defaults to today.")
def predict(date_opt=None):
    """Generate structured predictions (strict JSON) with historical context for a given date (default: today)."""
    date = date_opt or datetime.date.today().isoformat()
    infile = f"summaries/{date}.md"
    if not os.path.exists(infile):
        click.echo("No summary found for today; skipping predict.")
        return
    summary_md = open(infile).read()
    # If structured summary exists, prefer it
    summary_json_path = f"summaries/{date}.json"
    structured_summary = json.load(open(summary_json_path)) if os.path.exists(summary_json_path) else None

    # Get historical prediction context (last 7 days), include ids for superseding
    hist_entries = []
    today_dt = datetime.date.today()
    for i in range(1, 8):
        d = (today_dt - datetime.timedelta(days=i)).isoformat()
        pth = f"predictions/{d}.json"
        if os.path.exists(pth):
            try:
                data = json.load(open(pth))
                for p in data.get('predictions', []):
                    hist_entries.append({
                        'id': p.get('id'),
                        'text': p.get('text'),
                        'category': p.get('category'),
                        'deadline': p.get('deadline'),
                        'confidence_pct': p.get('confidence_pct') or p.get('confidence'),
                    })
            except Exception:
                continue
    hist_json = json.dumps(hist_entries) if hist_entries else '[]'

    # Build strict JSON prompt
    if structured_summary:
        summary_for_model = json.dumps(structured_summary)
        summary_prefix = "STRUCTURED_SUMMARY_JSON: "
    else:
        summary_for_model = summary_md
        summary_prefix = "SUMMARY_MD: "

    prompt = (
        "You are an expert forecasting analyst. Use recent history to avoid duplicates and calibrate confidence.\n\n"
        f"RECENT_PREDICTIONS_JSON: {hist_json}\n\n"
        f"{summary_prefix}{summary_for_model}\n\n"
        "Produce ONLY a JSON object with this schema:\n"
        "{\n"
        "  \"predictions\": [\n"
        "    {\n"
        "      \"text\": string,\n"
        "      \"category\": one of [\"macro\", \"markets\", \"rates\", \"energy\", \"tech\", \"geopolitics\", \"other\"],\n"
        "      \"horizon_days\": integer,\n"
        "      \"deadline\": string (YYYY-MM-DD),\n"
        "      \"confidence_pct\": integer 0-100,\n"
        "      \"log_odds\": number (optional),\n"
        "      \"verification_criteria\": string,\n"
        "      \"evidence\": array[string],\n"
        "      \"monitoring_signals\": array[string],\n"
        "      \"supersedes_id\": string (optional),\n"
        "      \"rationale\": string (optional, max 50 words)\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Constraints: 5-15 items; avoid duplicates vs RECENT_PREDICTIONS_JSON; if refining an existing idea, set supersedes_id to a prior id. Use clear numerical thresholds and concrete deadlines."
    )
    system = (
        "You are Foresight Forge. Output exactly one JSON object matching the schema. "
        "Do NOT include Markdown. Think step-by-step internally but do NOT reveal your chain-of-thought."
    )
    
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; skipping predict.")
        return
    openai.api_key = key
    try:
        rf = _predict_json_schema() if _strict_for('predict') else 'json'
        preds_json_text = _llm_respond(prompt, max_tokens=1200, system=system, response_format=rf)
    except Exception as e:
        click.echo(f"Error during predict: {e}")
        return

    # Parse and persist structured prediction log
    os.makedirs("predictions", exist_ok=True)
    pred_log_path = f"predictions/{date}.json"
    try:
        parsed = json.loads(preds_json_text)
        pred_entries = parsed.get('predictions', [])
    except Exception as e:
        click.echo(f"Failed to parse model JSON; aborting: {e}")
        return

    # Post-process: add ids and derived fields
    def _bin_conf(c):
        try:
            c = int(c)
        except Exception:
            return None
        c = max(0, min(100, c))
        low = (c // 10) * 10
        high = low + 10
        return f"{low}-{high}"

    for p in pred_entries:
        if not p.get('confidence_pct') and p.get('confidence') is not None:
            p['confidence_pct'] = p.pop('confidence')
        p['confidence_pct'] = max(0, min(100, int(p.get('confidence_pct', 0))))
        p['id'] = p.get('id') or str(uuid.uuid4())
        p['outcome'] = None
        p['confidence_bin'] = _bin_conf(p['confidence_pct'])

    with open(pred_log_path, "w") as fp:
        json.dump({"date": date, "predictions": pred_entries}, fp, indent=2)
    click.echo(f"Wrote structured predictions to {pred_log_path}")
    os.makedirs("newsletters", exist_ok=True)
    out = f"newsletters/{date}.md"
    with open(out, "w") as f:
        f.write(f"# Daily Newsletter {date}\n\n")
        f.write("## Summary\n\n")
        f.write(summary_md + "\n\n")
        f.write("## Predictions\n\n")
        for p in pred_entries:
            f.write(f"- {p.get('text')} — {p.get('confidence_pct')}% (deadline: {p.get('deadline')})\n")
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
    # Skip commits in Vercel or when explicitly requested. Allow GitHub Actions to commit.
    if os.getenv("VERCEL") or os.getenv("SKIP_GIT_COMMIT"):
        click.echo("CI environment detected; skipping git commit.")
        return
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
                            'prediction': pred.get('text'),
                            'confidence': pred.get('confidence_pct', pred.get('confidence')),
                            'outcome': pred.get('outcome')
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
        reply = _llm_respond(prompt, max_tokens=200)
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
    
    # Build review prompt with strict JSON assessment + prose analysis
    pred_for_model = [
        {
            'id': p.get('id'),
            'text': p.get('text'),
            'confidence_pct': p.get('confidence_pct', p.get('confidence')),
            'deadline': p.get('deadline'),
            'outcome': p.get('outcome'),
        }
        for p in predictions
    ]
    news_text = "\n".join([
        f"- {item['title']} ({item['link']})"
        for item in today_news[:50]  # Limit to first 50 items
    ])
    prompt = (
        "You are an expert forecasting analyst reviewing yesterday's predictions against today's news.\n\n"
        f"PREDICTIONS_JSON: {json.dumps(pred_for_model)}\n\n"
        f"TODAY_NEWS_BULLETS:\n{news_text}\n\n"
        "Return ONLY a JSON object with: {\n"
        "  \"assessments\": [ { \"id\": string, \"status\": one of [\"correct\", \"incorrect\", \"pending\", \"needs-clarification\"], \"status_rationale\": string } ],\n"
        "  \"analysis\": string (2-3 paragraphs of high-level review)\n"
        "}\n"
        "Update statuses only when evidence is clear."
    )
    system = (
        "You are Foresight Forge. Output exactly one JSON object as specified; no extra text. "
        "Think internally but do not reveal chain-of-thought."
    )
    
    # Get LLM review
    key = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        click.echo("OPENAI_API_KEY is not set; skipping review.")
        return
    
    openai.api_key = key
    
    try:
        rf = _review_json_schema() if _strict_for('review') else 'json'
        result_text = _llm_respond(prompt, max_tokens=1200, system=system, response_format=rf)
        result = json.loads(result_text)
        assessments = result.get('assessments', [])
        review_text = result.get('analysis', '').strip()

        # Build a human-readable list of predictions for the markdown
        def _conf(p):
            v = p.get('confidence_pct', p.get('confidence'))
            try:
                return int(v)
            except Exception:
                return 'N/A'
        pred_text = "\n".join([
            f"- {p.get('text')} (Confidence: {_conf(p)}%)"
            for p in predictions
        ])

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
        # Save structured assessments
        assess_file = f'reviews/{yesterday}-assess.json'
        with open(assess_file, 'w') as af:
            json.dump({"date": yesterday, "assessments": assessments}, af, indent=2)
        click.echo(f"Wrote structured assessments to {assess_file}")

        # Apply outcome updates where provided
        id_to_status = {a.get('id'): a.get('status') for a in assessments if a.get('id')}
        changed = False
        for p in predictions:
            pid = p.get('id')
            if pid and pid in id_to_status and id_to_status[pid] is not None:
                if p.get('outcome') != id_to_status[pid]:
                    p['outcome'] = id_to_status[pid]
                    changed = True
        if changed:
            with open(yesterday_pred_file, 'w') as pf:
                pred_data['predictions'] = predictions
                json.dump(pred_data, pf, indent=2)

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
@click.option('--since-days', type=int, default=7, help='Number of days to look back for discovery.')
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
        system = (
            "You are Foresight Forge. Output exactly a plain list of '- domain' or 'NONE'."
        )
        rf = _grammar_vet_list() if _strict_for('vet') else None
        approved_text = _llm_respond(prompt, max_tokens=500, system=system, response_format=rf)
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
    
    # RSS-only: filter additions to valid feed URLs
    filtered_add = _filter_feed_urls(to_add)
    skipped = [u for u in (to_add or []) if u not in filtered_add]
    if skipped:
        click.echo(f'Skipping non-feed URLs ({len(skipped)}): ' + ", ".join(skipped))

    # Apply additions (dedupe while preserving order)
    added = []
    for u in filtered_add:
        if u not in sources:
            sources.append(u)
            added.append(u)
    if added:
        click.echo(f'Added {len(added)} sources: ' + ", ".join(added))
    
    if not added and not to_remove:
        click.echo('No changes to apply.')
        return
    with open('sources.yaml', 'w') as f:
        yaml.safe_dump(sources, f, sort_keys=False)
    
    # Commit changes directly to main
    try:
        repo = Repo(os.getcwd(), search_parent_directories=True)
        repo.git.add('sources.yaml')
        date = datetime.date.today().isoformat()
        repo.index.commit(f'Brain auto-update sources (RSS-only): {date}')
        click.echo(f'Applied brain\'s source changes: {len(added)} added, {len(to_remove)} removed (RSS-only)')
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
            (
                f"<li>{p.get('text')}" + (
                    f" — {p.get('confidence_pct', p.get('confidence'))}%" if p.get('confidence_pct', p.get('confidence')) is not None else ''
                ) + '</li>'
            )
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
