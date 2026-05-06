# Embedding Strategy Recommendation Report

## Report Metadata

- **Generated at:** 2026-05-02T21:46:30Z
- **Input file:** `samples/pois_nova_real.csv`
- **Tool version:** 0.1.0
- **Rubric version:** mvp-modality-text-v1
- **LLM provider:** anthropic
- **LLM model:** claude-sonnet-4-5-20250929

## Dataset Profile

- **Rows:** 3,963
- **Columns:** 10

### Geometry

- **Source:** lat_lon_columns
- **Implied type:** point
- **Bounding box:** [-77.3600, 38.6607, -77.0053, 38.9596]
- **Point density (per km^2):** 3.89

### Attribute Summary

- **Text columns:** `name`, `description`
- **Categorical columns:** `category`, `amenity_type`, `access`
- **Numeric columns:** `capacity`, `rating`
- **Identifier columns:** `osm_id`

### Signals

- **Text richness:** 0.2
- **Categorical density:** 0.3
- **Numeric attribute presence:** 0.2
- **Geometric complexity:** 0.05
- **Scale tier:** small

### Signal Rationale

- **text_richness:** 2 text column(s) detected with mean length 19 characters.
- **categorical_density:** 3 categorical column(s) out of 10 total.
- **numeric_attribute_presence:** 2 numeric column(s) out of 10 total (coordinate columns excluded).
- **geometric_complexity:** Geometry type point detected via lat_lon_columns.

## User Intent

- **Category:** similarity_search
- **Description:** find points of interest semantically similar to a given example

## Recommendations

### Rank 1: categorical_attribute_embedding (score: 0.62)

**Rationale:** With categorical_density=0.3 and three categorical columns (category, amenity_type, access), this provides the strongest signal in the dataset. For similarity_search over points of interest, categorical attributes like amenity_type capture semantic groupings (restaurants, parks, etc.) that align well with 'similar POI' queries. Text_richness=0.2 is too low to favor text strategies, and geometric_complexity=0.05 offers minimal shape information.

**Implementation notes:**

- Use entity embeddings or target encoding for category, amenity_type, and access columns
- Normalize numeric columns (capacity, rating) and concatenate with categorical embeddings
- Consider dimensionality between 16-32 for the embedding layer given small scale_tier
- FAISS or Annoy indexing will support efficient similarity search at this scale

**Validation suggestions:**

- Sample known similar POI pairs (e.g., two coffee shops) and verify high cosine similarity
- Check that different amenity_types cluster separately in embedding space
- Evaluate retrieval precision@k for held-out query POIs with known similar examples

**Code scaffolding:** `scaffolds/categorical_attribute_embedding.py`

### Rank 2: text_attribute_embedding (score: 0.48)

**Rationale:** Text_richness=0.2 is below the strong-fit threshold (0.5), with only 2 text columns averaging 19 characters. However, similarity_search intent favors semantic strategies, and name/description fields may still carry useful POI semantics despite brevity. This ranks second because categorical_density=0.3 provides a stronger signal, but text embeddings remain viable for capturing nuanced semantic similarity.

**Implementation notes:**

- Apply sentence-transformer (e.g., all-MiniLM-L6-v2) to concatenated name + description
- Handle missing or very short text with padding or default tokens
- Consider fine-tuning on POI domain data if available to improve short-text representations

**Validation suggestions:**

- Test retrieval quality on POIs with descriptive names vs. sparse names
- Compare embedding similarity for synonymous POI types (e.g., 'cafe' vs. 'coffee shop')
- Measure coverage: what fraction of POIs have sufficient text for meaningful embeddings

**Code scaffolding:** `scaffolds/text_attribute_embedding.py`

### Rank 3: hybrid_categorical_geometric (score: 0.42)

**Rationale:** Categorical_density=0.3 is moderate, but geometric_complexity=0.05 is very low (point-only geometry with no shape features). While combining categorical embeddings with lat/lon coordinates could capture 'similar POI types nearby', the weak geometric signal limits the hybrid benefit. This ranks third because the categorical component is useful, but the geometric fusion adds minimal value given the low complexity.

**Implementation notes:**

- Concatenate categorical embeddings with normalized [lat, lon] coordinates
- Apply spatial normalization (e.g., min-max scaling within bbox) to balance feature magnitudes
- Weight categorical features more heavily than coordinates given the signal imbalance

**Validation suggestions:**

- Test whether spatial proximity improves or degrades categorical similarity for POI retrieval
- Ablation study: compare hybrid vs. categorical-only to quantify geometric contribution

**Code scaffolding:** `scaffolds/hybrid_categorical_geometric.py`

### Rank 4: hybrid_text_geometric (score: 0.35)

**Rationale:** Text_richness=0.2 is low and geometric_complexity=0.05 is very low. Both constituent signals are weak (text below 0.3 threshold, geometry far below 0.5). Similarity_search intent favors semantic strategies, but fusing two weak signals does not produce a strong combined representation. This ranks fourth because neither modality provides sufficient signal for effective hybrid fusion.

**Implementation notes:**

- Concatenate sentence embeddings with normalized coordinates if spatial context is deemed important
- Expect limited improvement over text-only given weak geometric signal

**Validation suggestions:**

- Compare retrieval quality against text_attribute_embedding baseline to assess spatial contribution
- Check if spatial features introduce noise rather than signal for semantic similarity

**Code scaffolding:** `scaffolds/hybrid_text_geometric.py`

### Rank 5: geometric_embedding (score: 0.18)

**Rationale:** Geometric_complexity=0.05 is very low, reflecting point-only geometry with no shape information. Coordinates alone do not carry semantic meaning for POI similarity_search tasks—two nearby POIs may be entirely different types. This strategy is a poor fit because the dataset lacks the geometric richness (polygons, complex shapes) needed for meaningful geometric embeddings, and the intent requires semantic understanding, not spatial structure.

**Implementation notes:**

- Only viable if spatial clustering is the primary goal, not semantic similarity
- Could encode [lat, lon] directly, but this will group by location rather than POI type or meaning

**Validation suggestions:**

- Verify that this approach fails to distinguish semantically different POIs at the same location
- Use only as a baseline to demonstrate the need for attribute-based strategies

**Code scaffolding:** `scaffolds/geometric_embedding.py`

## Warnings

- Column 'description' is 97% null. Consider whether to include it in the embedding input.
- Column 'access' is 98% null. Consider whether to include it in the embedding input.
- Column 'capacity' is 92% null. Consider whether to include it in the embedding input.
- Column 'rating' is 100% null. Consider whether to include it in the embedding input.
- Spatial extent is small (sub-degree). For coordinate normalization, use the bounding box of this dataset rather than a global range.
