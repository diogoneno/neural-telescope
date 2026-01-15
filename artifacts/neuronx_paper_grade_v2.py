#!/usr/bin/env python3

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = os.environ.get("NEURONX_DEVICE", "cuda:0")
DTYPE = torch.float16

TEMPLATES = [
    "{city} is the capital of {country}.",
    "The capital of {country} is {city}.",
    "{city} serves as the capital of {country}.",
    "In {country}, the capital is {city}.",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(s: str) -> List[int]:
                                
    parts = []
    for chunk in s.replace(",", " ").split():
        if chunk.strip():
            parts.append(int(chunk.strip()))
    if not parts:
        raise ValueError("Empty list")
    return parts


def perm_pvalue(t_obs: float, t_perm: np.ndarray, two_sided: bool = True) -> float:
    m = int(t_perm.shape[0])
    if two_sided:
        t_obs = abs(t_obs)
        t_perm = np.abs(t_perm)
    b = int(np.sum(t_perm >= t_obs))
    return float((b + 1) / (m + 1))


def split_entities(cities: List[str], seed: int, n_test: int, n_val: int) -> Tuple[List[str], List[str], List[str]]:
    rng = np.random.default_rng(seed)
    uniq = sorted(set(cities))
    rng.shuffle(uniq)
    test = uniq[:n_test]
    val = uniq[n_test : n_test + n_val]
    train = uniq[n_test + n_val :]
    if len(train) < 10:
        raise ValueError(f"Train too small ({len(train)} cities). Reduce test/val or add data.")
    return train, val, test


def build_items(
    pairs: List[Dict[str, str]],
    templates: List[str],
    rng: np.random.Generator,
) -> List[Tuple[str, int, str, int]]:
                                                                      
    country_to_city = {p["country"]: p["city"] for p in pairs}
    countries = list(country_to_city.keys())

    items: List[Tuple[str, int, str, int]] = []

    for p in pairs:
        city = p["city"]
        true_country = p["country"]

                                                                                           
        eligible = [c for c in countries if c != true_country and country_to_city[c] != city]
        if not eligible:
                                                            
            raise RuntimeError(f"No eligible wrong_country for city={city!r}, country={true_country!r}")
        wrong_country = rng.choice(eligible)

        for tpl_id, tpl in enumerate(templates):
            t = tpl.format(city=city, country=true_country)
            f = tpl.format(city=city, country=wrong_country)
            items.append((city, tpl_id, t, 1))
            items.append((city, tpl_id, f, 0))

    return items


class BatchedExtractor:

    def __init__(self, model_name: str, device: str, dtype: torch.dtype):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

                                                                               
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(device)

        self.model.eval()
                                                             
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

                                                                        
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    @torch.inference_mode()
    def extract_layers(
        self,
        texts: List[str],
        which_token: str,
        batch_size: int,
        drop_embeddings: bool = True,
    ) -> List[np.ndarray]:
        if which_token not in ("last", "mean"):
            raise ValueError("which_token must be 'last' or 'mean'")

        X_layers: List[List[np.ndarray]] | None = None

        for i in tqdm(range(0, len(texts), batch_size), desc=f"extract:{which_token}", leave=False):
            chunk = texts[i : i + batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}

            attn = enc.get("attention_mask")
            if attn is None:
                raise RuntimeError("attention_mask missing; cannot do padding-safe pooling")

            out = self.model(**enc, output_hidden_states=True, use_cache=False)
            hs = list(out.hidden_states)

                                                
            if drop_embeddings and len(hs) >= 2:
                hs = hs[1:]

                                                         
            last_idx = (attn.sum(dim=1) - 1).clamp(min=0)

            if X_layers is None:
                X_layers = [[] for _ in range(len(hs))]

                                            
            mask = attn.to(dtype=torch.float32).unsqueeze(-1)           
            denom = mask.sum(dim=1).clamp(min=1.0)         

            for L, h in enumerate(hs):              
                if which_token == "mean":
                    v = (h.to(torch.float32) * mask).sum(dim=1) / denom         
                else:
                    idx = last_idx.view(-1, 1, 1).expand(-1, 1, h.size(-1))
                    v = h.gather(dim=1, index=idx).squeeze(1).to(torch.float32)         

                X_layers[L].append(v.cpu().numpy())

        assert X_layers is not None
        return [np.concatenate(chunks, axis=0).astype(np.float32) for chunks in X_layers]

    def close(self):
        del self.model
        torch.cuda.empty_cache()


def learn_theta(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = X[y == 1]
    X0 = X[y == 0]
    if len(X1) == 0 or len(X0) == 0:
        raise RuntimeError("Train split missing a class; cannot learn theta")
    return X1.mean(axis=0) - X0.mean(axis=0)


def entity_grouped_scores(items, scores):
    by = {}
    for (city, _tpl, _text, y), s in zip(items, scores):
        by.setdefault(city, {0: [], 1: []})[y].append(float(s))
    return by


def paired_acc(items, scores) -> float:
    by = entity_grouped_scores(items, scores)
    ok = 0
    tot = 0
    for _city, d in by.items():
        if len(d[0]) == 0 or len(d[1]) == 0:
            continue
        tot += 1
        ok += int(np.mean(d[1]) > np.mean(d[0]))
    return ok / tot if tot else float("nan")


def deltas_by_entity(items, scores) -> np.ndarray:
    by = entity_grouped_scores(items, scores)
    deltas = []
    for _city, d in by.items():
        if len(d[0]) == 0 or len(d[1]) == 0:
            continue
        deltas.append(np.mean(d[1]) - np.mean(d[0]))
    return np.array(deltas, dtype=np.float64)


def entity_bootstrap_ci_from_entity_values(vals: np.ndarray, n_boot: int, alpha: float, rng: np.random.Generator) -> Tuple[float, float]:
    if vals.size == 0:
        return (float("nan"), float("nan"))
    boots = []
    n = vals.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n, endpoint=False)
        boots.append(float(np.mean(vals[idx])))
    boots = np.array(boots, dtype=np.float64)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


@dataclass
class RunResult:
    model: str
    seed: int
    holdout_tpl: int
    selected_token: str
    selected_layer: int
    val_auc: float
    test_auc: float
    test_auc_perm_p: float
    test_auc_perm_mean: float
    test_auc_perm_std: float
    paired_acc: float
    paired_ci_lo: float
    paired_ci_hi: float
    mean_delta: float
    delta_ci_lo: float
    delta_ci_hi: float
    signflip_p: float

    def as_dict(self) -> Dict:
        return self.__dict__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="JSON list of {city,country} pairs.")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", required=True, help="Comma/space separated seeds, e.g. 0,1,2,3,4")
    ap.add_argument("--holdouts", required=True, help="Comma/space separated holdout template ids, e.g. 0,1,2,3")
    ap.add_argument("--test_k", type=int, default=50)
    ap.add_argument("--val_k", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--n_signperm", type=int, default=5000)
    ap.add_argument("--two_sided", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out_json", default="neuronx_paper_grade_multiseed_holdout.json")
    args = ap.parse_args()

    seeds = parse_int_list(args.seeds)
    holdouts = parse_int_list(args.holdouts)

    print(f"SEEDS={seeds} HOLDOUTS={holdouts} test_k={args.test_k} val_k={args.val_k}")
    print(f"n_perm={args.n_perm} n_signperm={args.n_signperm} two_sided={args.two_sided}\n")

    pairs = json.load(open(args.dataset, "r"))
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Dataset must be a non-empty list of {city,country}")

    cities = [p["city"] for p in pairs]

    all_results: List[Dict] = []

    for model_name in args.models:
        print("\n" + "=" * 30)
        print(f"MODEL: {model_name}")
        print("=" * 30)

        per_model_results: List[RunResult] = []

        for seed in seeds:
            print(f"\n--- seed={seed} ---")
            seed_all(seed)

            train_ents, val_ents, test_ents = split_entities(cities, seed, args.test_k, args.val_k)
            print(f"Entities: train={len(train_ents)} val={len(val_ents)} test={len(test_ents)} | total={len(set(cities))}")

            rng_data = np.random.default_rng(seed)
            all_items = build_items(pairs, TEMPLATES, rng_data)

            def filt(items, ents, holdout_tpl=None, exclude_tpl=None):
                ents = set(ents)
                out = [x for x in items if x[0] in ents]
                if holdout_tpl is not None:
                    out = [x for x in out if x[1] == holdout_tpl]
                if exclude_tpl is not None:
                    out = [x for x in out if x[1] != exclude_tpl]
                return out

            ext = BatchedExtractor(model_name, DEVICE, DTYPE)

            for holdout_tpl in holdouts:
                train_items = filt(all_items, train_ents, exclude_tpl=holdout_tpl)
                val_items = filt(all_items, val_ents, exclude_tpl=holdout_tpl)
                test_items = filt(all_items, test_ents, holdout_tpl=holdout_tpl)

                Xtr_last = ext.extract_layers([x[2] for x in train_items], "last", args.batch_size)
                Xva_last = ext.extract_layers([x[2] for x in val_items], "last", args.batch_size)
                Xte_last = ext.extract_layers([x[2] for x in test_items], "last", args.batch_size)

                Xtr_mean = ext.extract_layers([x[2] for x in train_items], "mean", args.batch_size)
                Xva_mean = ext.extract_layers([x[2] for x in val_items], "mean", args.batch_size)
                Xte_mean = ext.extract_layers([x[2] for x in test_items], "mean", args.batch_size)

                ytr = np.array([x[3] for x in train_items], dtype=np.int64)
                yva = np.array([x[3] for x in val_items], dtype=np.int64)
                yte = np.array([x[3] for x in test_items], dtype=np.int64)

                n_layers = len(Xtr_last)

                                                              
                candidates = []
                for which, XtrL, XvaL in [("last", Xtr_last, Xva_last), ("mean", Xtr_mean, Xva_mean)]:
                    for L in range(n_layers):
                        theta = learn_theta(XtrL[L], ytr)
                        auc_val = float(roc_auc_score(yva, XvaL[L] @ theta))
                        candidates.append((auc_val, which, L))

                candidates.sort(key=lambda x: x[0], reverse=True)
                best_auc_val, best_which, best_L = candidates[0]

                Xtr_sel = Xtr_last if best_which == "last" else Xtr_mean
                Xte_sel = Xte_last if best_which == "last" else Xte_mean

                theta = learn_theta(Xtr_sel[best_L], ytr)
                test_scores = Xte_sel[best_L] @ theta
                auc_test = float(roc_auc_score(yte, test_scores))
                pa = float(paired_acc(test_items, test_scores))

                                            
                deltas = deltas_by_entity(test_items, test_scores)
                mean_delta = float(np.mean(deltas))

                                             
                rng_ci = np.random.default_rng(seed + 1337 + holdout_tpl)
                                                                                               
                pa_city = (deltas > 0).astype(np.float64)
                pa_lo, pa_hi = entity_bootstrap_ci_from_entity_values(pa_city, args.n_boot, args.alpha, rng_ci)
                d_lo, d_hi = entity_bootstrap_ci_from_entity_values(deltas, args.n_boot, args.alpha, rng_ci)

                                                                                         
                rng_perm = np.random.default_rng(seed + 999 + holdout_tpl)
                auc_perm = []
                for _ in range(args.n_perm):
                    ysh = ytr.copy()
                    rng_perm.shuffle(ysh)
                    theta_p = learn_theta(Xtr_sel[best_L], ysh)
                    auc_perm.append(float(roc_auc_score(yte, Xte_sel[best_L] @ theta_p)))
                auc_perm = np.array(auc_perm, dtype=np.float64)

                                                 
                t_obs = abs(auc_test - 0.5) if args.two_sided else auc_test
                t_perm = np.abs(auc_perm - 0.5) if args.two_sided else auc_perm
                p_auc = perm_pvalue(t_obs, t_perm, two_sided=False)

                                                            
                rng_sf = np.random.default_rng(seed + 2024 + holdout_tpl)
                sf_stats = []
                for _ in range(args.n_signperm):
                    signs = rng_sf.choice([-1.0, 1.0], size=deltas.shape[0], replace=True)
                    sf_stats.append(float(np.mean(deltas * signs)))
                sf_stats = np.array(sf_stats, dtype=np.float64)
                p_sf = perm_pvalue(mean_delta, sf_stats, two_sided=args.two_sided)

                rr = RunResult(
                    model=model_name,
                    seed=int(seed),
                    holdout_tpl=int(holdout_tpl),
                    selected_token=str(best_which),
                    selected_layer=int(best_L),
                    val_auc=float(best_auc_val),
                    test_auc=float(auc_test),
                    test_auc_perm_p=float(p_auc),
                    test_auc_perm_mean=float(auc_perm.mean()),
                    test_auc_perm_std=float(auc_perm.std()),
                    paired_acc=float(pa),
                    paired_ci_lo=float(pa_lo),
                    paired_ci_hi=float(pa_hi),
                    mean_delta=float(mean_delta),
                    delta_ci_lo=float(d_lo),
                    delta_ci_hi=float(d_hi),
                    signflip_p=float(p_sf),
                )

                per_model_results.append(rr)
                all_results.append(rr.as_dict())

                print(
                    f"holdout_tpl={holdout_tpl} | sel={best_which}:L{best_L} valAUC={best_auc_val:.4f} | "
                    f"testAUC={auc_test:.4f} p_auc={p_auc:.6f} | paired={pa:.4f} (CI {pa_lo:.4f}-{pa_hi:.4f}) | "
                    f"meanΔ={mean_delta:.4f} (CI {d_lo:.4f}-{d_hi:.4f}) p_sf={p_sf:.6f}"
                )

            ext.close()

                           
        m = [r for r in per_model_results if r.model == model_name]
        aucs = np.array([r.test_auc for r in m], dtype=np.float64)
        pas = np.array([r.paired_acc for r in m], dtype=np.float64)
        dels = np.array([r.mean_delta for r in m], dtype=np.float64)
        pA = np.array([r.test_auc_perm_p for r in m], dtype=np.float64)
        pS = np.array([r.signflip_p for r in m], dtype=np.float64)

        print(f"\n=== SUMMARY (seeds×holdouts={len(m)}) ===")
        print(f"AUROC      = {aucs.mean():.4f} ± {aucs.std(ddof=1):.4f}")
        print(f"paired_acc = {pas.mean():.4f} ± {pas.std(ddof=1):.4f}")
        print(f"meanΔ      = {dels.mean():.4f} ± {dels.std(ddof=1):.4f}")
        print(f"perm p(AUC)   median={np.median(pA):.6f} min={pA.min():.6f} max={pA.max():.6f}")
        print(f"signflip p(Δ) median={np.median(pS):.6f} min={pS.min():.6f} max={pS.max():.6f}")

    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] wrote {args.out_json}")


if __name__ == "__main__":
    main()
