# Node-classification re-run (results_v2)

This folder holds the corrected node-classification results for Section 5 (Quantitative Evaluation) of the CM2ML paper. They replace every number in Table 13 and Figures 11–14. This README covers what was wrong, what was run, the new results, and which paper claims hold or change, section by section.

- Code: branch `fix/node-cls-pipeline`. The runs used commits `57418ee` and `3776f83`, plus the ModelSet label fix that is committed together with these results.
- Runs: 960 configurations, 0 failures, one seed (42).
- Date: 18–19 November 2026.

## Contents

1. [What was wrong with the old results](#1-what-was-wrong-with-the-old-results)
2. [Experimental setup as run](#2-experimental-setup-as-run-text-for-section-51)
3. [Results: top configurations (Table 13)](#3-results-top-configurations-table-13)
4. [Factor analysis (Figures 12–14)](#4-factor-analysis-figures-1214)
5. [Interactive transition plot (Figure 11)](#5-interactive-transition-plot-figure-11)
6. [Tree vs graph, like-for-like](#6-tree-vs-graph-like-for-like)
7. [Paper claims: what holds and what changes](#7-paper-claims-what-holds-and-what-changes)
8. [Suggested edits by section](#8-suggested-edits-by-section)
9. [Caveats and threats to validity](#9-caveats-and-threats-to-validity)
10. [Reproducing](#10-reproducing)
11. [File index](#11-file-index)

---

## 1. What was wrong with the old results

The old GNN and tree rows of Table 13, the old ModelSet numbers, and every factor analysis built on them are invalid. Old text-classifier results on graphs at d = 0 were correct: the new pipeline reproduces them exactly, for example EA Layer BoW at d = 0 gives 0.6147 in both.

| # | Problem | Effect on the old paper |
|---|---|---|
| G1 | GNN node features were always built at d = 0. Path length, edge type and edge label were ignored. | GNN rows at d = 1–3 were the same model as d = 0. That is why the old EA Layer top 5 were five near-identical rows (0.7118–0.7135). |
| G2 | The `include_node_type` flag added the **target** (layer or abstract), not the node type. | A label leak if switched on. It was off in the old grid. |
| G3 | Test-node types inside paths were not hidden. | A potential leak through neighbours. |
| G4, G5 | Bag-of-Paths followed outgoing edges only. GNN messages flowed one way. | Half of each neighbourhood was invisible. |
| T1–T3 | Tree serializer: the object node's text was the literal token `OBJ`, its name and attributes sat in child nodes, and associations hung off the root. Tree GNNs only received messages from the root. | Tree GNNs predicted a constant class (F1 0.0025–0.084). This produced the "trees are insufficient" finding. |
| T4, T5 | For EA Type, tree roles (`object_name`, …) became classes, and the object's `type` field leaked the label. | Wrong label space and a leak. |
| L1 | **ModelSet label bug** in [lang2graph/ecore.py](../lang2graph/ecore.py): every class that another class inherits from was marked abstract, whatever the model file declared. | 2,474 of 29,468 labels (8.4%) were wrong, 36% of all "abstract" labels. On those labels, the rule "abstract iff it has a subclass" scored 0.985. |
| X1–X3 | The corpus cache never hit, an O(n²) masking lookup slowed everything, and run records lacked configs. | Speed and traceability only. |

All fixes are covered by `tests/test_bug_fixes.py` (unit tests), `tests/verify_real_data.py` (leak audit on real data, exact reproduction of the old d = 0 results) and `tests/check_grid_duplicates.py` (no two grid configs give identical inputs).

The superseded ModelSet runs on the buggy labels are kept in [superseded_modelset_labelbug/](superseded_modelset_labelbug/) for reference only.

## 2. Experimental setup as run (text for Section 5.1)

**Datasets and split.**
- Datasets are unchanged:
  - EA ModelSet: 456 graphs and 59,095 nodes. EA Type has 63 classes; EA Layer has 7.
  - ModelSet Abstract: 845 graphs, 29,468 nodes, 2 classes.
- 20% of nodes in every graph are test nodes (mask seed 42). Metrics are computed on test nodes only.
- ModelSet "abstract" is the declared `EClass@abstract` flag.

**Leakage controls.**
- The classified node never shows its target. On EA it also never shows its own type.
- Test-node neighbours show their name but not their type.
- Key/value meta-properties that reveal the target, such as `isAbstract`, are dropped from tree attribute nodes.

**Factors.**

| Factor | Levels |
|---|---|
| Structural encoding | graph, tree (this repo's serializer: MODEL → OBJ → ATTS/ASSOCS → REF) |
| Text encoder | BoW, TF-IDF (50k features), Word2Vec (128-d, trained on the corpus), frozen BERT (`bert-base-uncased`, mean-pooled). BERT fine-tuning was removed. |
| Path length d (Bag-of-Paths) | 0–3. Paths follow incoming and outgoing edges, with direction marked in the text (`->`, `<-`), capped at 64 paths per node. |
| Text configuration, EA | T = edge type, N = neighbour node type (training neighbours only). EA has no attributes or edge labels. |
| Text configuration, ModelSet | A = attributes, L = edge label, T = edge type. ModelSet has no node types. |
| Training setup | Text classifier (logistic regression, max_iter 3000), or GNN: GraphSAGE, GCN or GAT (2 layers, hidden size 128, dropout 0.3, learning rate 1e-3, weight decay 5e-4, 30 epochs). GNN features are the encoded node text at the configured d. |

**Edges in the GNN.**
- Graph GNNs use the original edge direction.
- Tree GNNs use bottom-up edges. Top-down edges let object nodes hear only from the root, which collapsed predictions to one class.

**Only non-vacuous configurations are run.**
- A flag that cannot change the input is not run and is shown as **n/a**:
  - edge and neighbour flags at d = 0;
  - those flags at d = 1 for tree + text classifier, because a tree object reaches its references only at d = 2 (OBJ → ASSOCS → REF).
- Tree + text classifier at d = 0 is identical to graph + text classifier at d = 0 by construction. It is not run, and the analysis reuses the graph run.

**Reduced design.** This replaces the full grid, which would have needed 1,648 runs.

| Part | Content | EA Type | EA Layer | ModelSet |
|---|---|---|---|---|
| Core | text classifier + GraphSAGE × every encoder × structure × d × text config | 192 | 192 | 384 |
| Slice | GCN and GAT, only with every available text flag on | 64 | 64 | 64 |
| **Total** | | **256** | **256** | **448** |

Factors are compared only where they were varied:
- encoder, path length and text flags on the core;
- training setup on the slice, where all four learners are present for the same inputs.

The best GCN or GAT run is 0.035–0.169 F1 below each dataset's 5th-best run, so the reduced design cannot have missed a top-5 configuration.

## 3. Results: top configurations (Table 13)

LaTeX, ready to paste: [report/top5_table.tex](report/top5_table.tex). It adds an **NType** column, because EA now has two text flags. The top 10 per dataset are in `report/top10_<dataset>.csv`.

| Dataset | Encoder | Cls. | Tree | d | Attr. | EType | ELabel | NType | GNN | Acc. | F1_M |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EA Type | BoW | Text | F | 1 | n/a | T | n/a | T | n/a | 0.6919 | **0.5962** |
| EA Type | BoW | GNN | F | 1 | n/a | T | n/a | F | SAGE | 0.6716 | 0.5853 |
| EA Type | BoW | GNN | F | 1 | n/a | T | n/a | T | SAGE | 0.6854 | 0.5838 |
| EA Type | BoW | GNN | F | 2 | n/a | T | n/a | T | SAGE | 0.6980 | 0.5808 |
| EA Type | BERT | Text | F | 1 | n/a | T | n/a | T | n/a | 0.6550 | 0.5801 |
| EA Layer | BoW | GNN | F | 1 | n/a | T | n/a | T | SAGE | 0.8780 | **0.8361** |
| EA Layer | BoW | GNN | F | 3 | n/a | T | n/a | T | SAGE | 0.8755 | 0.8270 |
| EA Layer | BoW | GNN | F | 2 | n/a | T | n/a | F | SAGE | 0.8741 | 0.8265 |
| EA Layer | BoW | GNN | F | 2 | n/a | T | n/a | T | SAGE | 0.8772 | 0.8250 |
| EA Layer | BoW | GNN | F | 3 | n/a | T | n/a | F | SAGE | 0.8711 | 0.8248 |
| ModelSet | BoW | GNN | F | 1 | F | T | T | n/a | SAGE | 0.9390 | **0.8807** |
| ModelSet | BoW | GNN | F | 1 | T | T | T | n/a | SAGE | 0.9377 | 0.8794 |
| ModelSet | BoW | GNN | F | 1 | F | T | F | n/a | SAGE | 0.9366 | 0.8792 |
| ModelSet | BoW | GNN | F | 1 | T | T | F | n/a | SAGE | 0.9353 | 0.8734 |
| ModelSet | W2V | GNN | F | 1 | F | T | F | n/a | SAGE | 0.9281 | 0.8709 |

Best macro-F1, old vs new:

| Dataset | Old | New | d = 0 baseline (graph, BoW, text classifier) |
|---|---|---|---|
| EA Type | 0.506 | **0.596** | 0.419 |
| EA Layer | 0.714 | **0.836** | 0.615 (identical to the old paper) |
| ModelSet Abstract | 0.861 (wrong labels) | **0.881** | 0.618 / 0.624 (attributes off / on) |

**ModelSet rule baseline.** On the corrected labels, the rule "abstract iff the class has a subclass" scores macro-F1 **0.857** (accuracy 0.913) on the same 5,887 test nodes; see `analysis/modelset_rule_baseline.py`. The best model adds 0.024 over this rule. We recommend reporting the rule as a baseline. With edge type on, the node text contains `<- supertype`, which exposes exactly this signal. That is legitimate input, but it explains most of the edge-type gain on ModelSet.

## 4. Factor analysis (Figures 12–14)

These figures replace the paper's figures and use the same style. Each comes as PNG (scale 2) and PDF in [report/figures/](report/figures/).

| Paper figure | Pearson panel (graph and tree) | eta² panel (graph and tree) |
|---|---|---|
| Fig. 12, EA Layer | `ea_layer_state_correlations` | `ea_layer_factor_association` |
| Fig. 13, EA Type | `ea_type_state_correlations` | `ea_type_factor_association` |
| Fig. 14, ModelSet | `modelset_abstract_state_correlations` | `modelset_abstract_factor_association` |

**How the figures are computed.**
- Pearson r is computed between each parameter state (one-hot) and F1. eta² is computed per factor with a one-way analysis.
- Training-setup states and factors use the slice. Encoder, path length and flag states use the core.
- A flag is counted only where it can change the input.
- "Bag of Paths" is the factor d ∈ {0, 1, 2, 3}.
- Raw values: `report/factor_stats_<dataset>.csv`.

**eta² per factor, new results.**

| | Training setup | Cls. type (text vs GNN) | Text encoder | Bag of Paths | Edge type | Node type / Attr. / ELabel |
|---|---|---|---|---|---|---|
| EA Type, graph | **0.42** | 0.27 | 0.35 | 0.13 | 0.03 | NType 0.07 |
| EA Type, tree | **0.37** | 0.30 | 0.24 | 0.07 | 0.04 | NType 0.01 |
| EA Layer, graph | 0.03 | 0.00 | **0.28** | 0.24 | 0.04 | NType **0.26** |
| EA Layer, tree | 0.01 | 0.00 | **0.51** | 0.09 | 0.02 | NType 0.06 |
| ModelSet, graph | **0.24** | 0.00 | 0.07 | 0.22 | 0.07 | Attr 0.00, ELabel 0.00 |
| ModelSet, tree | 0.06 | 0.05 | **0.49** | 0.04 | 0.01 | Attr 0.00, ELabel 0.00 |

**Matched-pair effects** (`report/matched_pairs_<dataset>.csv`). Each value is the mean ΔF1 when one factor changes and everything else stays fixed. Brackets give the share of pairs that improve.

| Change | EA Type | EA Layer | ModelSet |
|---|---|---|---|
| BoW → TF-IDF | −0.132 (0%) | −0.026 (18%) | −0.056 (9%) |
| BoW → W2V | −0.113 (12%) | −0.091 (12%) | −0.072 (18%) |
| BoW → BERT (frozen) | −0.128 (8%) | −0.147 (2%) | −0.103 (19%) |
| Edge type off → on | +0.038 (99%) | +0.030 (99%) | +0.028 (86%) |
| Neighbour node type off → on | +0.037 (82%) | +0.064 (88%) | — |
| Attributes off → on | — | — | −0.008 (40%) |
| Edge label off → on | — | — | +0.002 (51%) |
| d 0 → 1 | **+0.129** (97%) | **+0.159** (94%) | **+0.109** (88%) |
| d 1 → 2 | +0.013 (63%) | +0.017 (50%) | −0.018 (38%) |
| d 2 → 3 | −0.014 (13%) | −0.006 (34%) | −0.019 (38%) |
| Text classifier → SAGE | −0.084 (16%) | +0.013 (56%) | **+0.077** (78%) |
| Text classifier → GCN | −0.186 (3%) | −0.025 (34%) | +0.003 (56%) |
| Text classifier → GAT | −0.175 (3%) | −0.013 (41%) | +0.011 (56%) |

The same CSVs split every effect by structure, graph vs tree. Section 6 gives the full tree vs graph comparison.

## 5. Interactive transition plot (Figure 11)

- **Link:** [explorer artifact](https://claude.ai/artifact/9ijVrY6hBcoFn2VMrvPjXp) (private until you share it).
- **Local copy:** open `report/explorer.html` in a browser. It works offline except for loading Plotly from a CDN.
- **Default path:** graph → BoW → A0-L0-T0 (EA: T0-N0) → d = 0 → text classifier.
- **How it works:** pick any values for structure, encoder, text config, path length and training. Each stage keeps the earlier choices and the later defaults, as described in Section 5.2.2.
- **What it shows:** each dataset tab also shows the top-5 table and the four factor charts.
- **Missing configs:** configurations outside the reduced design, such as GCN/GAT with flags off, are shown as missing and the page says why.
- **For the paper:** take a screenshot of the EA Layer tab. The page is built from `report/explorer_data.json` by `analysis/build_explorer.py`.

The numbers quoted in Section 5.2.2 change. The EA Layer, graph, BoW, text-classifier path with edge type off is:

| d | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Old paper | 0.615 | 0.667 | 0.674 | similar |
| New | 0.615 | 0.726 | 0.753 | 0.725 |

The best GNN on EA Layer is now 0.836 (SAGE, d = 1, T1-N1), not 0.714.

## 6. Tree vs graph, like-for-like

Averaging over all configurations would mix up structure with the other factors. Each tree run is therefore paired with the graph run that has the same encoder, training setup, d and text configuration. Pairs at d = 0 with the text classifier are excluded, because their inputs are identical by construction. Outputs are in [report/structure/](report/structure/) and come from `analysis/structure_comparison.py`.

**Overall.** Tree − graph ΔF1. All overall gaps are significant (Wilcoxon signed-rank, p < 0.001).

| | Pairs | Mean Δ | Tree better in | Best tree | Best graph | Best-vs-best Δ |
|---|---|---|---|---|---|---|
| EA Type | 132 | −0.065 | 10% | 0.519 | 0.596 | −0.078 |
| EA Layer | 132 | −0.109 | 2% | 0.737 | 0.836 | −0.099 |
| ModelSet | 232 | −0.066 | 23% | 0.795 | 0.881 | −0.086 |

**By path length.** At d = 0 the two encodings see the same node text and perform alike. The gap opens once neighbourhood context is added.

| | d = 0 | d = 1 | d = 2 | d = 3 |
|---|---|---|---|---|
| EA Type | −0.007 (n.s.) | −0.099 | −0.070 | −0.045 |
| EA Layer | −0.031 (n.s.) | −0.144 | −0.110 | −0.098 |
| ModelSet | −0.020 (n.s.) | −0.113 | −0.059 | −0.036 |

**By training setup.**

| | Text cls | SAGE | GCN | GAT |
|---|---|---|---|---|
| EA Type | −0.075 | −0.083 | −0.020 | −0.025 |
| EA Layer | −0.118 | −0.113 | −0.093 | −0.086 |
| ModelSet | −0.074 | −0.084 | +0.001 | **+0.035** |

- ModelSet with GCN or GAT is the only place where tree keeps up: tree is better in 62% of those pairs.
- The best tree GCN (0.784) beats the best graph GCN (0.761).
- These cells have only 16 pairs each, all-flags-on only. The large GAT d = 3 value comes mostly from graph GAT getting worse with depth.

**By encoder.**

| | BoW | TF-IDF | W2V | BERT |
|---|---|---|---|---|
| EA Type | −0.090 | −0.044 | −0.050 | −0.076 |
| EA Layer | −0.102 | −0.102 | −0.084 | **−0.147** |
| ModelSet | −0.020 | −0.010 (n.s.) | −0.089 | **−0.145** |

On ModelSet, BoW and TF-IDF bring tree close to graph, while frozen BERT shows the largest gap everywhere.

**What drives the size of the gap.** This is eta² of each factor on the paired Δ, using the `moderators_<dataset>.csv` files.

| | Path length | Training setup | Encoder | Node type | Edge type | Attr. / ELabel |
|---|---|---|---|---|---|---|
| EA Type | 0.17 | 0.15 | 0.10 | 0.07 | 0.01 | — |
| EA Layer | 0.17 | 0.03 | 0.09 | 0.14 | 0.00 | — |
| ModelSet | 0.18 | 0.17 | **0.55** | — | 0.02 | 0.00 / 0.00 |

**Equal reach.**
- A tree object needs two steps (OBJ → ASSOCS → REF) to reach what a graph node reaches in one.
- So `reach_<dataset>.csv` compares tree text at d + 1 with graph text at d.
- Tree d = 1 vs graph d = 0: about equal (+0.00 to +0.04).
- Tree d = 2 and d = 3 vs graph d = 1 and d = 2: still −0.06 to −0.10.
- Extra distance alone does not explain the gap.

**Mechanism.**
- The tree serializer lists only each object's *outgoing* references, as leaf nodes (`cm2ml_encodings_eval/tree_transform.py`).
- No incoming references: in a tree, a class cannot see who references or inherits from it. On ModelSet that hides subclasses, the strongest cue for "abstract".
- No second hop: nothing beyond a referenced object is reachable, so EA tree results are identical at d = 2 and d = 3.
- In this implementation, "tree vs graph" also means "outgoing references, one hop" vs "both directions, several hops". This should be stated in the paper.

**Heatmap.** [report/structure/figures/structure_gap.png](report/structure/figures/structure_gap.png) shows tree − graph by training setup × d for the three datasets.

## 7. Paper claims: what holds and what changes

### Holds

| Claim | Evidence |
|---|---|
| Encoding design is a primary factor in ML performance | F1 ranges from 0.06 to 0.60 (EA Type), 0.32 to 0.84 (EA Layer) and 0.46 to 0.88 (ModelSet) across configurations. |
| No single configuration is best for every task | EA Type is best with the text classifier; EA Layer and ModelSet are best with SAGE. |
| Graph beats tree, and no tree appears in any top 5 | No tree appears in any top 10. Section 6 gives the like-for-like gap. |
| BoW is the strongest text encoder, and simple beats complex | Switching away from BoW costs 0.03–0.15 F1, and BoW wins in 80–100% of matched pairs. This is stronger than before. |
| GraphSAGE is the best GNN, GCN and GAT are worse | Holds on all three datasets. |
| EA Layer top 5: graph + BoW + SAGE | All five rows. |
| EA Type is best with graph + text classifier | Holds, but SAGE is close (0.585 vs 0.596). |
| ModelSet top 5: graph + SAGE + BoW | 4 of 5 are BoW; the 5th is W2V. |
| Short local context is the critical bridge (Discussion) | d = 0 → 1 is the largest single effect (+0.11 to +0.16). Going deeper adds about 0 or hurts. |

### Changes

| Old claim | New result |
|---|---|
| Trees are insufficient: the best tree at d = 0 reaches 0.084, and Bag-of-Paths "rescues" trees | **Bug artifact.** Tree + SAGE at d = 0 scores 0.612 on EA Layer, the same as graph. Bag-of-Paths helps both encodings, and helps graphs more (d = 0 → 1: +0.23 graph vs +0.09 tree on EA Layer). |
| Fine options (edge type, edge label, attributes) have small effects | **Edge type** gives +0.03 to +0.04 in 86–99% of pairs. **Neighbour node type** gives +0.04 to +0.06; on the EA Layer graph its eta² (0.26) is about equal to the encoder's (0.28). Attributes and edge labels have no effect on ModelSet. |
| Text → GNN costs −0.158 on EA Layer; GNNs help only on strong representations | On EA Layer, text → SAGE is +0.013. The benefit depends on task and encoder: SAGE gives −0.08 on EA Type, +0.01 on EA Layer and +0.08 on ModelSet. |
| Hierarchy: graph → the encoder decides; tree → the learner and Bag-of-Paths decide | **No longer holds.** Tree is encoder-driven on EA Layer (0.51) and ModelSet (0.49). EA Type is learner-driven in both structures. ModelSet graph is driven by the learner (0.24) and Bag-of-Paths (0.22), with the encoder at only 0.07. |
| EA Layer graph: encoder eta² 0.89; Bag-of-Paths negligible | Encoder 0.28, node type 0.26 and Bag-of-Paths 0.24 are similar. |
| Frozen BERT is mildly negative or weakly positive | Weakest or near-weakest encoder everywhere (−0.10 to −0.15 vs BoW). |
| ModelSet: attributes help | No effect (−0.008; helps in 40% of pairs). |
| ModelSet: BoW, TF-IDF and BERT positive; W2V strongly negative | Only BoW is positive. |
| ModelSet tree: text-only best, GNN negative | **Reversed.** On trees, SAGE beats the text classifier (+0.066) and GAT does too (+0.062). |
| Synthesis: ModelSet best was graph-based text classification | Wrong in the old paper too. The best rows are SAGE GNNs. |
| Sec. 5.2.2: EA has two text configs (A0-L0-T0, A0-L0-T1) | EA now has four: edge type × neighbour node type (T, N). |

### Effect on the overall message

- **The headline gets stronger.** "Encoding design is a primary, task-dependent factor, and simple encoders are competitive."
- **Three supporting arguments need replacing:**
  1. "Trees fail and Bag-of-Paths rescues them" becomes: **one step of local context (d = 1) is the most valuable choice in every setting.** Trees are a consistent step down from graphs, because the tree exposes only outgoing references.
  2. "High-level choices matter more than fine options" becomes: **cheap contextual options (edge type, neighbour types) are as influential as the encoder choice.** Attributes and edge labels are not.
  3. "GNNs only help on strong graph representations" becomes: **GNN benefit depends on the task.** SAGE helps ModelSet, is neutral on EA Layer and hurts EA Type. GCN and GAT trail SAGE everywhere.
- **Simpler overall story:** a simple lexical encoder (BoW), one hop of context and edge types reach most of the achievable performance. The choice of learner is what depends on the task.

## 8. Suggested edits by section

| Section | Edit |
|---|---|
| 5.1 Experimental Setup | Add the leakage controls, bidirectional paths with direction markers, GNN features at the configured d, tree GNNs classifying only object nodes, the reduced design (GCN/GAT with all flags on), frozen BERT only, the declared ModelSet `abstract`, the single seed and mask seed 42. Text is in Section 2 above. |
| Table 12 | Unchanged. |
| Table 13 | Replace with `report/top5_table.tex` (adds an NType column; the caption explains n/a). |
| 5.2.1 | New best scores, and the ModelSet rule baseline (0.857). EA Type: the text classifier wins narrowly over SAGE. ModelSet: best rows are graph + SAGE + d = 1 + edge type. |
| 5.2.2 / Fig. 11 | New screenshot from the explorer. Fix the text-config description for EA. Remove the −0.158 paragraph and the 0.084 / 0.614 tree paragraph. Use the new numbers: best EA Layer 0.836, text → SAGE +0.013, d = 0 → 1 +0.159. |
| 5.2.3 / Figs. 12–14 | Replace the figures and rewrite each dataset paragraph from the eta² table in Section 4. |
| New subsection (optional) | Tree vs graph like-for-like, from Section 6. Suggested text is below. |
| 5.3 Synthesis | Rewrite findings 2 and 3 as in Section 7. Fix the ModelSet sentence (the best is a GNN). |
| 5.4 RQ2 | Replace "tree-based encodings require additional local path context to become competitive" and "high-level representational decisions have a larger impact than fine-grained options". Keep "BoW often outperforms" and "encoding design is a primary factor". |
| 6.1 Insights | Recommend d = 1 with edge types as the default. Trees lose context in this serializer because they omit incoming references. Fine-grained flags are worth exposing. |
| 6.3 Threats | Add the single seed, BLAS noise, the reduced design and the ModelSet rule baseline (Section 9). |
| 7 Conclusion | Align with the updated RQ2 answer. |

Suggested text for the tree vs graph comparison:

> With identical node text (d = 0), tree and graph encodings perform alike. Once neighbourhood context is added, the graph encoding is better by 0.10–0.14 macro-F1 at d = 1, and remains better when each encoding is given its best settings (−0.08 to −0.10). The gap is smallest with frequency-based encoders and with GCN/GAT, where tree can match graph on ModelSet, and largest with frozen BERT. The tree exposes only an element's outgoing references, which accounts for much of the difference.

## 9. Caveats and threats to validity

- **Single seed.** All results use seed 42.
  - GNN noise is about ±0.001–0.006 F1.
  - The EA Type top 5 spans 0.016 and the ModelSet top 5 spans 0.010, so the order within each top 5 is not reliable.
  - The planned extra-seed runs for the top 10 per dataset have not been run yet.
- **Thread-count noise.**
  - Logistic regression (lbfgs, 63 classes) shifts by about 0.004 F1 with the BLAS thread count. EA Type at d = 0 gives 0.4190 with 8 threads, as in the grid, and 0.4227 unrestricted.
  - Differences below about 0.005 between text-classifier runs are noise.
- **Reduced design.**
  - GCN and GAT were run only with every text flag on.
  - So their effects are measured on 16 matched pairs per dataset, and tree vs graph comparisons for GCN/GAT rest on few runs.
- **ModelSet rule baseline.** The has-subclass rule reaches 0.857, so models add about 0.02 beyond structure that is visible in one hop.
- **Tree serializer.** The tree shows outgoing references only, so conclusions about trees apply to this serializer, not to every tree encoding.
- **Other pickles.** `datasets/pickles/ecore_555.pkl` and `mar-ecore-github.pkl` were built with the old `ecore.py` and still carry the old abstract labels. Rebuild them before any use.

## 10. Reproducing

Environment: the repository's Python environment. GPUs are chosen by `gpu_select.py`, which requires at least twice the needed memory free and at most 20 GB in use.

```bash
# 1. ModelSet pickle with declared abstract labels (only needed once)
python -c "from data_loading.models_dataset import get_models_dataset as g; \
           g('modelset', min_enr=1.2, min_edges=10, reload=True, remove_duplicates=True)"

# 2. The grid (reduced design). EA jobs as run:
python run_all_node_cls_combinations.py --dataset eamodelset --target type  --gpu-mem-mb 4000 --torch-threads 8
python run_all_node_cls_combinations.py --dataset eamodelset --target layer --gpu-mem-mb 4000 --torch-threads 8
#    ModelSet was run as 4 shards, i = 0..3:
python run_all_node_cls_combinations.py --dataset modelset --target abstract --shard $i/4 \
       --gpu-mem-mb 4000 --torch-threads 8
#    Runs resume automatically from the jsonl files.

# 3. Reports, figures, explorer
python analysis/node_cls_report.py          # report/: top5/top10, factor stats, matched pairs, figures
python analysis/structure_comparison.py     # report/structure/
python analysis/build_explorer.py           # report/explorer.html
python analysis/modelset_rule_baseline.py   # rule baseline (CPU)

# 4. Checks
python -m pytest tests/test_bug_fixes.py
python tests/verify_real_data.py
python tests/check_grid_duplicates.py
```

## 11. File index

| Path | Content |
|---|---|
| `node_cls_grid_eamodelset_type.jsonl` | 256 EA Type runs: one JSON record per run with full config, metrics, git commit, timings and GPU memory. |
| `node_cls_grid_eamodelset_layer.jsonl` | 256 EA Layer runs. |
| `node_cls_grid_modelset_abstract_shard{0..3}of4.jsonl` | 448 ModelSet runs on the corrected labels. |
| `*_summary.json` | Per-job counts. The EA Type summary shows 16 because that job was resumed; the jsonl has all 256. |
| `logs/*.log.gz` | Run logs, gzipped. Read them with `zcat`. |
| `superseded_modelset_labelbug/` | Old ModelSet runs on the buggy labels. Do not use. |
| `report/top5_table.tex` | Table 13 replacement. |
| `report/top5_*.csv`, `report/top10_*.csv` | Top configurations per dataset. |
| `report/factor_stats_*.csv` | Pearson and eta² values behind Figures 12–14. |
| `report/matched_pairs_*.csv` | One-factor-at-a-time ΔF1, overall and by structure. |
| `report/figures/*_state_correlations.{png,pdf}` | Pearson panels. |
| `report/figures/*_factor_association.{png,pdf}` | eta² panels. |
| `report/explorer.html`, `report/explorer_data.json` | Interactive transition plot (Figure 11). |
| `report/structure/pairs_*.csv` | Every matched tree/graph pair. |
| `report/structure/conditional_*.csv` | Gap per factor level: mean, median, share tree better, Wilcoxon p. |
| `report/structure/cells_*.csv` | Gap per two-factor cell (setup × d, encoder × setup, encoder × d). |
| `report/structure/moderators_*.csv` | eta² of each factor on the gap. |
| `report/structure/best_*.csv` | Best tree vs best graph, overall and per level. |
| `report/structure/reach_*.csv` | Equal-reach comparison (tree d + 1 vs graph d). |
| `report/structure/figures/structure_gap.{png,pdf}` | Heatmap of tree − graph. |

Scripts are in `../analysis/`: `node_cls_report.py`, `structure_comparison.py`, `build_explorer.py` with `explorer_template.html`, and `modelset_rule_baseline.py`.
