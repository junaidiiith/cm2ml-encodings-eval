# YAML Config Schema

Top-level keys:

- `seed`: integer random seed
- `dataset`:
  - `dataset_name`: one of available dataset names (`modelset`, `ecore_555`, `mar-ecore-github`, `eamodelset`, `ontouml`)
  - `params`: forwarded to `get_models_dataset(...)`
  - `max_graphs`: optional cap for quick experiments
- `masking`:
  - `enabled`: bool
  - `ratio`: test-node ratio per graph
  - `seed`: masking seed
  - `reset_existing`: bool
- `metadata`:
  - `cls`: target label field
  - `label`: text label field
  - `attributes`: node attributes field
- `split`: optional split config for relevant pipelines
- `text_build`: text/path construction options
- `text_experiment`: text encoder/classifier options
- `bert`: BERT classifier options
- `gnn`: GNN options
- `tree`: tree serialization options
- `runs`: list of runs, each with:
  - `name`: run label
  - `pipeline`: one of
    - `text_node`
    - `text_edge`
    - `gnn_node`
    - `gnn_edge`
    - `tree_text_node`
    - `tree_gnn_node`
  - optional per-run overrides for any config section above
  - optional `max_graphs` to limit graphs for that run only
  - optional `node_encoder_name`, `edge_encoder_name`
  - optional `graph_index` (used by single-graph pipelines like `gnn_edge`)

Encoder examples:

- `tfidf`
- `word2vec`
- `custom_w2v` (defaults to `out/skip_gram_modelling/skip_gram_vectors.kv`)
- `custom_w2v:out/skip_gram_modelling/skip_gram_vectors.kv` (explicit path)

## Run

```bash
python3 -m cm2ml_encodings_eval.yaml_runner --config configs/pipeline_text_gnn.yaml --output results/text_gnn.json
```
