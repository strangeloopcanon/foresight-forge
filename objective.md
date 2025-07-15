
# Foresight Forge — Objective

*Version: 2025-07-15*

## Purpose  

Build a self‑running forecasting service that, each day, digests incoming information, publishes probabilistic predictions, and records every artefact for future reference.

## Daily Cycle  

1. **Ingest** – Read every URL in the current `sources.yaml` list and pull only items that have not been seen before.  
2. **Summarise** – Condense the new items into ≤10 clear bullet points.  
3. **Predict** – From each day’s summary (plus recent history) generate at least three testable predictions with explicit confidence levels.  
4. **Record** – Append raw pulls, the summary, and the predictions to version control so the full history is visible.  
5. **Publish** – Produce a Markdown newsletter for the day; this will later be pasted into Substack.  

## Comment & Review Mode  

* Accept comments or questions.  
* Load the relevant day’s summary and predictions plus any prior context.  
* Draft a reply and save it to a `comments/` folder so it is also versioned.  

## Source List Maintenance  

* **Weekly discovery pass** proposes new high‑signal sources.  
* The agent decides what heuristics to use (volume, novelty, relevance, etc.).  
* Candidates are stored in `discover/` and can be merged automatically or by human review, updating `sources.yaml`.  

## Self‑Update Capability  

The agent can open a branch, apply its own changes (for example, adding a new source), and push a pull request or commit, keeping the repository current without manual intervention.

## Minimal Dashboard (Optional)  

A simple read‑only page can display the latest summary, today’s predictions, and a link to the full history. Rendering can be done directly from the Markdown files.

## Storage Philosophy  

Everything—raw data, summaries, predictions, comments, and source lists—is plain text and lives in the repository. This guarantees transparency, diffability, and easy rollback.

## Success Criteria  

* A new Markdown newsletter appears every calendar day with fresh predictions.  
* All historical data can be browsed and reused by the agent for future reasoning.  
* The source list evolves automatically when valuable new feeds emerge.  
* Operating the system never requires manual database work—only reviewing newsletters or occasional pull requests.

---
