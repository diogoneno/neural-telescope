# The Neural Telescope: Interferometric Control of Autonomous Agents

> **Project Martian Candidate | 2026**  
> Phase 1 (public): measurement + statistical validation • Phase 2 (under review): interlock control + low-latency bridge

## Overview

The Neural Telescope is a program for interpretability and control of autonomous LLM agents.  
It is organized as an interlock: (i) instrument internal state, (ii) transport telemetry, and (iii) enable intervention policies.

This repository is a **grant-review artifact** containing a validated Phase-1 evaluation pipeline on a factual-statement benchmark and the related result records. Full documentation on critical components for Phase-2 technical due diligence is under controlled access.

### Terminology (Phase-1 operational meaning)

- Topological Phase Attribution (TPA) includes layer-resolved probes that quantify the separability of internal representations for distinguishing true from constructed-false statements, ensuring stability across seeds and holdouts.
- Phase Drift: reproducible changes in probe separability across layers, estimated through layerwise selection and stability.
- Phase-1 does not claim a token-level "early warning window"; it reports validation-selected layers under controlled evaluation.

## The Architecture: "Neuron-X" Interlock

This repository offers a detailed overview of interlock architecture designed for grant evaluation standards.

- Extract hidden-state summaries and evaluate candidate discriminators by layer.
- Phase-1 evidence shows that validation-selected layers concentrate in **8–11** (median **10**) using last-token pooling.

- * **Bridge (Shared-Memory Steering Bridge):** a low-latency telemetry transport designed for online policies.  
  Phase-2 focus: deterministic schemas, bounded-latency telemetry, reproducible stream semantics.  
  Implementation and benchmarks aren't part of Phase-1 public artifact.

- Control policies for gating, damping, or rerouting based on telemetry.  
  Phase-2 focus: policy evaluation with safety constraints and measurable criteria.

## Benchmarks

### Dataset provenance

Phase-1 uses a Wikidata-derived country–capital dataset:

- `capitals_clean.json` — 187 `{city, country}` pairs
- `capitals_clean.meta.json` — WDQS endpoint, retrieval time, SPARQL query, and filtering notes
  - Source: **Wikidata Query Service (WDQS)**
  - Property: **P36 (capital)**
  - Entity class: **sovereign state (Q3624078)**
  - Filter: countries with exactly one distinct capital

### Task construction (controlled factual-statement discrimination)

We generate labeled items using **four paraphrase templates**:

1) "{city} is the capital of {country}."  
2) "The capital of {country} is {city}."  
3) "{city} serves as the capital of {country}."  
4) "In {country}, the capital is {city}."

Binary labels are generated as true vs constructed-false statements through negative sampling.  
(e.g., incorrect city or country) with safeguards against accidental truths.

### Holdout design (generalization constraints)

Cities are divided into train, validation, and test sets (entity OOD).  
Leave-one-template-out: train+val with 3 templates; test on held-out template.

## Statistical Validation

We use nonparametric tests for complex statistics without parametric assumptions.

- Monte Carlo permutation test (AUROC): p-values computed with standard +1 correction to prevent zero p-values and maintain calibration.
- Sign-flip randomization test (Δ): paired deltas per city tested with random sign flips.

## Results Snapshot (Phase-1 artifact)

Model: **EleutherAI/gpt-j-6B**  
Runs: **20** (seeds 0–4 × template holdouts 0–3)  
Metric: test **AUROC** (ROC AUC), plus paired entity metrics and randomization tests.

Aggregated across the 20 recorded runs (`neuronx_paper_grade_multiseed_holdout_v2.json`):

- **Test AUROC:** **0.9777 ± 0.0220** (mean ± SD)
- **Permutation p(AUC):** median **0.0073** (min **0.0020**, max **0.0624**)
- **Entity-paired accuracy:** **0.9800 ± 0.0225** (mean ± SD)
- **Mean per-city delta (true − false):** **172.40 ± 66.26** (mean ± SD)
- **Sign-flip p(Δ):** **0.0002** (with 5000 sign permutations)

Layer selection behavior (same 20 runs):

- **Selected layer range:** **8–11** (median **10**)
- **Selected pooling:** **last-token** (all runs)
- Layer selection counts: L8=1, L9=7, L10=8, L11=4

## Compute / Lab Environment (provenance)

Primary local environment used during Phase-1 development:

- **Host:** Lenovo ThinkStation P700
- **CPU:** Dual Xeon E5-2699 v3
- **RAM:** 128GB
- **Storage:** 4TB NVMe (Flex Connector), 1TB SSD, 2TB HDD

### Telemetry Matrix

| Name | Hardware | Role | Spec | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Turing** | NVIDIA P100 | Telemetry Core | 12 vCPUs, 32GB RAM | Active (Debian 12.11) |
| **Bengio** | NVIDIA K2200 | Experimental | 24 vCPUs, 48GB RAM | Active (Debian 12.11) |

## Scope boundaries (important for reviewers)

Phase-1 claims focus on the **controlled benchmark** of true vs constructed-false capital statements under entity/template holdouts. It does **not** claim generalized “hallucination detection," “confidence estimation,” or a token-level lead-time window.

Phase-2 work (under review) extends the interlock to: streaming telemetry (bridge) with bounded latency, online policy evaluation (control) under safety constraints, and scaling evaluation to larger agents and benchmarks.

## Artifact inventory

Public artifacts:
- `capitals_clean.json`
- `capitals_clean.meta.json`
- `neuronx_paper_grade_multiseed_holdout_v2.json`
- `README.md`

Restricted artifacts (Phase-2 due diligence):
- performance instrumentation details,
- bridge/control implementations and latency benchmarks,
- extended evaluation suites and scaling runs.

## References (external)

- WDQS documentation and SPARQL endpoint: [Link]
- Wikidata Property P36 (capital): [Link]
- Phipson & Smyth (Permutation p-values should never be zero): [Link]
- GPT-J-6B model card: [Link]
- scikit-learn ROC AUC: [Link]
