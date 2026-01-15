# The Neural Telescope: Interferometric Control of Autonomous Agents

> Phase 1 (public): measurement + statistical validation • Phase 2 (under review): interlock control + low-latency bridge

## Overview

The Neural Telescope enables interpretability and control of LLM agents.
It is organized as an interlock: (i) instrument internal state, (ii) transport telemetry, and (iii) enable intervention policies.

This repository is a **grant-review artifact** containing a validated Phase-1 evaluation pipeline on a factual-statement benchmark and the related methodology. Detailed Phase-1 result tables and Phase-2 technical materials are provided under **controlled access**.

### Terminology (Phase-1 operational meaning)

- **Topological Phase Attribution (TPA):**Layer-resolved probes quantify the separability of internal representations to distinguish true from false statements, assessing stability across seeds and holdouts.
- **Phase Drift:** Reproducible changes in probe separability estimated through layerwise selection.
- Phase-1 reports validation-selected layers under controlled evaluation, not claiming a token-level “early warning window.”

## The Architecture: "Neuron-X" Interlock

This repository describes an interlock architecture designed for grant evaluation standards:

- Extract hidden-state summaries and evaluate discriminators by layer.
- Utilize validation-selected layers within entity/template generalization constraints.

**Bridge (Shared-Memory Steering Bridge):** A low-latency transport for online policies.
Phase-2 focus: deterministic schemas, bounded-latency telemetry, reproducible semantics.
Implementation and benchmarks are excluded from Phase-1.

**Control:** policies for gating, damping, or rerouting based on telemetry.  
Phase-2 focus: policy evaluation with safety constraints and measurable criteria.

## Benchmarks

### Dataset provenance

Phase-1 uses a Wikidata-derived country–capital dataset:

- `capitals_clean.json` — `{city, country}` pairs
- `capitals_clean.meta.json` — WDQS endpoint, retrieval time, SPARQL query, and filtering notes
  - Source: **Wikidata Query Service (WDQS)**
  - Property: **P36 (capital)**
  - Entity class: **sovereign state (Q3624078)**
  - Filter: countries with exactly one distinct capital

### Task construction (controlled factual-statement discrimination)

We generate labeled items using four paraphrase templates:

1) "{city} is the capital of {country}."
2) "The capital of {country} is {city}."
3) "{city} serves as the capital of {country}."
4) "In {country}, the capital is {city}."

Binary labels are created as true versus constructed-false using negative sampling.
(e.g., incorrect city or country) with safeguards against accidental truths.

### Holdout design (generalization constraints)

-Entity OOD: cities split into train, validation, test sets.
- Leave-one-template-out: train+val with 3 templates; test on held-out template.

## Statistical Validation

We use nonparametric tests for complex statistics.

- Monte Carlo permutation test (AUROC): p-values calculated using standard +1 correction to avoid zero p-values and ensure calibration.
- Sign-flip randomization test (Δ): paired deltas per city with random flips.

## Phase-1 Results

Phase-1 numeric result tables, run logs, and per-holdout breakdowns are available under **controlled access** for reviewers.
The public artifact encompasses the complete pipeline, dataset provenance, and methodology needed to reproduce the evaluation.

## Compute / Lab Environment

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

## Scope boundaries

Phase-1 claims emphasize comparing true and constructed-false capital statements using entity/template holdouts.
It does not claim generalized “hallucination detection” or token-level guarantees.

Phase-2 work (under review) extends interlock to streaming telemetry with bounded latency, online policy evaluation under safety constraints, and scaling for larger agents.

## Artifact inventory

Public artifacts:
- `capitals_clean.json`
- `capitals_clean.meta.json`
- Phase-1 runner / evaluation code (this repository)
- `README.md`

Controlled-access artifacts:
- Phase-1 result tables + run logs (per-model/per-seed/per-holdout)
- performance instrumentation details
- bridge/control implementations and latency benchmarks
- extended evaluation suites and scaling runs

## References (external)

- WDQS documentation and SPARQL endpoint: [Link]
- Wikidata Property P36 (capital): [Link]
- Phipson & Smyth (Permutation p-values should never be zero): [Link]
- GPT-J-6B model card: [Link]
- scikit-learn ROC AUC (`roc_auc_score`): [Link]
