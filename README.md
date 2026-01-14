
# The Neural Telescope: Interferometric Control of Autonomous Agents

> **Project Martian Candidate | 2026**

## Overview

The Neural Telescope is a mechanistic interpretability suite that identifies and stops "Confident Hallucinations" in autonomous agents by monitoring **Topological Phase Attribution (TPA)** of neural activations in real-time.

We identify "Phase Drift" in the transformer's residual stream as a geometric misalignment occurring **8 layers before** the error becomes a token.

## The Architecture: "Neuron-X" Interlock

This repository provides the reference implementation of the **Shared Memory Steering Bridge** for **sub-50ms ITI**.

* **Telemetry (Layers 14-19):** Detects "Decoherence Signals" (Deviation).
* **Bridge (Arrow + shm):** Bypasses IPC overhead to steer Llama-3 70B real-time.
* **Control (PLL):** Phase-Locked Loop: Uses inverse steering vectors to adjust trajectories seamlessly.

## Infrastructure & Benchmarks

This system is validated on a tiered local inference cluster designed for high-availability interpretation.

### Node: Proxmox01 (Lenovo P700)
* **Compute:** 2x Xeon 2699v3 (72 vCPUs), 128GB RAM
* **Storage:** 4TB NVMe (Flex Connector), 1TB SSD, 2TB HDD

### Telemetry Matrix

| Name | Hardware | Role | Spec | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Turing** | NVIDIA P100 | Telemetry Core | 12 vCPUs, 32GB RAM | Active (Debian 12.11) |
| **Bengio** | NVIDIA K2200 | Experimental | 24 vCPUs, 48GB RAM | Active (Debian 12.11) |

## Key Findings (P100 Telemetry)

Research conducted on the **NVIDIA Tesla P100 (Pascal)** hardware indicates:

1.  **The 8-Layer Window:** Hallucinations exhibit a unique vector signature in **Layer 14** before appearing in Layer 19.
2.  **Stability-Scale Paradox:** Larger models (70B) require higher precision (TF32) for signal tracking due to lower circuit stability (Jaccard Index ~0.6).