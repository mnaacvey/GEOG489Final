# Geospatial Embedding Strategy Recommender

The tool takes a CSV geospatial dataset and a stated user intent, profiles the dataset deterministically, and asks Claude to apply a rubric. Returns a ranked list of five embedding strategy recommendations with a reason for each. Output is a single Markdown report at `reports/report.md`. This is the GEOG 489 final project deliverable and a working prototype of the central artifact of my MS thesis in Spatial Data Science.

## Quickstart

Requires Python 3.11+ and an Anthropic API key.

```bash
pip install ".[dev]"

embedding-recommender \
  --input samples/pois_nova_real.csv \
  --intent-category similarity_search \
  --intent-description "find points of interest semantically similar to a given example" \
  --output-dir reports/
```

If `ANTHROPIC_API_KEY` is not in the environment or in a `.env` file, the tool prompts for it on stdin and the input is hidden. Paste the key once and the run proceeds. To skip the prompt on future runs, save the key to `.env`: `cp .env.example .env`, then edit the `ANTHROPIC_API_KEY=` line.

The report lands at `reports/report.md`. The included sample (`samples/pois_nova_real.csv`) is real OSM amenity data, around 4,000 points across inner Northern Virginia. It is checked in. No need to rebuild it. The build pipeline (`samples/build_real_sample.sh`, `samples/convert_osm_to_csv.py`) is checked in for reproducibility but is not part of the run path.

`--intent-category` must be one of `similarity_search`, `classification`, `clustering`, `rag_qa`, `spatial_relational`. `--intent-description` is a short free-text sentence describing the actual task.

## What the tool actually does
The pipeline is two halves. A deterministic profiler measures the dataset. A rubric-bound LLM call applies the rubric to the profile and the user intent. The LLM does not invent rules. It applies the rubric in `rubric.py` to the numbers in the profile. That split was the most important design decision in the project.

**Profiler.** Classifies every column into text, categorical, numeric, or identifier using dtype checks and substring matching against curated lists. Detects geometry from columns named `lat`, `latitude`, `y`, `lon`, `lng`, `longitude`, or `x`. Computes a bounding box and point density. Collapses everything into five normalized signals: text richness, categorical density, numeric attribute presence, geometric complexity, and a discrete scale tier.

**Recommender.** Sends the profile, the intent, and the rubric prose to Claude via the Anthropic SDK. Asks for a ranked array of exactly five recommendations from this set: `text_attribute_embedding`, `categorical_attribute_embedding`, `geometric_embedding`, `hybrid_text_geometric`, `hybrid_categorical_geometric`. Validates the response against a JSON schema in `schema.py` and a separate business rule check (ranks must be a permutation of 1 through 5, each candidate must appear exactly once). On any validation failure the run aborts with a clear error and the operator reruns. LLM nondeterminism is real even at temperature zero. The retry path is the answer.

**Reporter** Renders the validated report as Markdown via plain f-strings. No templating engine.
Each recommendation references a starter Python skeleton in `src/embedding_recommender/scaffolds/`. The skeletons are intentionally minimal. They sketch the shape of an implementation, not a turnkey model.

## Architecture
One file per responsibility. Anthropic is the only LLM provider in scope and CSV is the only input format, so neither has been wrapped in an abstraction. Both are plain module functions. A second provider or a second input format is the point at which an abstraction would earn its place.

## File map
```
src/embedding_recommender/
├── cli.py             # argparse entry point
├── config.py          # env vars and module-level constants
├── data_source.py     # load_csv: reads a CSV into a DataFrame
├── profiler.py        # profile() and signal computation
├── intent.py          # validates the user intent input
├── rubric.py          # rubric prose and decision principles
├── schema.py          # JSON schemas + business-rule validator
├── llm_provider.py    # generate_recommendations: Anthropic call
├── recommender.py     # orchestrates profiler + rubric + LLM
├── reporter.py        # validates and writes the Markdown report
├── timing.py          # @timed decorator for profile() and the LLM call
└── scaffolds/         # starter skeletons for each candidate strategy
```

## Scope

This is an MVP. Explicitly out of scope:

- Input formats other than CSV. GeoPackage, Shapefile, GeoParquet, and direct PostGIS reads are on the thesis roadmap, not here.
- Local LLM providers. Anthropic only.
- Graph-structure embedding strategies (node2vec, GraphSAGE, knowledge-graph embeddings). Second axis on the rubric, not yet implemented.
- Multimodal strategies (raster, imagery).
- Any GUI.
- A templating engine for the report.

A `Dockerfile` and `docker-compose.yml` are checked in. The Docker setup is a stub. Local install is the supported run path.

## Tests

```bash
pytest tests/ -v
```

Unit tests cover the profiler (column classification, geometry detection, signal computation), the JSON schemas, the business rule validator, the reporter, and the intent validator. The LLM call is not in the test suite. Recording a real Anthropic response and replaying it under a fixture is on the to-do list.
