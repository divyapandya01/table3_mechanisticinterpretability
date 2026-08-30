"""
Table 3 extraction, corrected. Fixes two problems from the first run:

  1. rows had no ground-truth label -> probe target had to be concept_idx,
     which is recoverable from token identity alone (flat 100% from layer 1).
     Now stores the binary rule label per query.
  2. behavioral accuracy was never captured, so there was no way to tell
     what a probe result meant. Now read off the same forward pass, free.

Also stores rule `family` so the complexity breakdown and cross-concept
transfer probes can run without re-extraction.

Validate on a small model first:
    python table3_extract_v2.py --model Qwen/Qwen2.5-1.5B --smoke
    python table3_extract_v2.py --model Qwen/Qwen2.5-1.5B
Then scale:
    python table3_extract_v2.py --model meta-llama/Meta-Llama-3-8B
"""

import argparse, json, random, string
from pathlib import Path
import numpy as np

SEED = 0
CLASS_NAMES = ("blicket", "daxen")   # nonce class labels
SHOTS = (1, 2, 4, 8, 16)
N_CONCEPTS_PER_FAMILY = 5
N_QUERIES = 24                        # held-out queries per concept
FAMILIES = ("single", "conjunctive", "disjunctive", "negation")


# ---------------------------------------------------------------- concepts
def _nonce(rng, n=6):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def make_concept(rng, family, idx):
    """Binary rule over nonce tokens. Rule form varies by family so the
    complexity breakdown can be reported per family."""
    m1, m2 = _nonce(rng, 2), _nonce(rng, 2)
    while m2 == m1:
        m2 = _nonce(rng, 2)

    if family == "single":
        rule = lambda t: m1 in t
    elif family == "conjunctive":
        rule = lambda t: (m1 in t) and (m2 in t)
    elif family == "disjunctive":
        rule = lambda t: (m1 in t) or (m2 in t)
    elif family == "negation":
        rule = lambda t: (m1 in t) and (m2 not in t)
    else:
        raise ValueError(family)

    # generate a balanced item pool by rejection sampling, with injection
    # so positives are reachable for the rarer rule forms
    pos, neg, guard = [], [], 0
    target = (N_QUERIES + 16)          # queries + demonstration pool
    while (len(pos) < target or len(neg) < target) and guard < 200_000:
        guard += 1
        t = _nonce(rng, 7)
        if rng.random() < 0.5:         # inject markers half the time
            p = rng.randrange(0, 5)
            t = t[:p] + (m1 if rng.random() < 0.6 else m2) + t[p + 2:]
            if rng.random() < 0.35:
                q = rng.randrange(0, 5)
                t = t[:q] + m2 + t[q + 2:]
        if rule(t):
            if len(pos) < target:
                pos.append(t)
        else:
            if len(neg) < target:
                neg.append(t)

    return {
        "concept_idx": idx,
        "concept_id": f"{family}_{m1}_{m2}",
        "family": family,
        "m1": m1, "m2": m2,
        "positives": pos, "negatives": neg,
    }


def build_concepts():
    rng = random.Random(SEED)
    out, idx = [], 0
    for fam in FAMILIES:
        for _ in range(N_CONCEPTS_PER_FAMILY):
            c = make_concept(rng, fam, idx)
            if len(c["positives"]) < N_QUERIES + 8 or len(c["negatives"]) < N_QUERIES + 8:
                print(f"  [warn] {c['concept_id']}: thin pool "
                      f"({len(c['positives'])}/{len(c['negatives'])})")
            out.append(c)
            idx += 1
    return out


# ---------------------------------------------------------------- prompts
def build_prompt(c, query, k, rng):
    """k-shot, class-balanced, order-shuffled. No rule statement and no CoT --
    either would tell the model the rule and confound sample efficiency."""
    n_each = max(1, k // 2)
    pool_p = [t for t in c["positives"] if t != query]
    pool_n = [t for t in c["negatives"] if t != query]
    demos = ([(t, CLASS_NAMES[1]) for t in rng.sample(pool_p, n_each)] +
             [(t, CLASS_NAMES[0]) for t in rng.sample(pool_n, n_each)])
    rng.shuffle(demos)
    demos = demos[:k]
    lines = [f"{t} -> {lab}" for t, lab in demos]
    lines.append(f"{query} ->")
    return "\n".join(lines)


def build_dataset():
    rng = random.Random(SEED + 1)
    concepts = build_concepts()
    rows, prompts = [], []
    for c in concepts:
        items = ([(t, 1) for t in c["positives"][:N_QUERIES // 2]] +
                 [(t, 0) for t in c["negatives"][:N_QUERIES // 2]])
        rng.shuffle(items)
        for test_idx, (query, label) in enumerate(items):
            for k in SHOTS:
                rows.append({
                    "concept_idx": c["concept_idx"],
                    "concept_id":  c["concept_id"],
                    "family":      c["family"],
                    "k":           k,
                    "test_idx":    test_idx,
                    "query":       query,
                    "label":       label,      # <-- THE FIX
                })
                prompts.append(build_prompt(c, query, k, rng))
    return concepts, rows, prompts


# ---------------------------------------------------------------- extract
def extract(model, tokenizer, prompts, device, batch_size=16):
    """Returns (activations, predictions).

    Activations: (n, n_layers+1, hidden) float16, final token position.
    Predictions: argmax over the two class-name first tokens -> behavioral
    accuracy for free, from the same forward pass.
    """
    import torch
    from tqdm import tqdm

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tok_ids = [tokenizer.encode(" " + n, add_special_tokens=False)[0]
               for n in CLASS_NAMES]
    print(f"class label token ids: {tok_ids} "
          f"({[tokenizer.decode([t]) for t in tok_ids]})")
    if tok_ids[0] == tok_ids[1]:
        raise ValueError("class names share a first token -- pick different ones")

    A, P = [], []
    for i in tqdm(range(0, len(prompts), batch_size), desc="extracting"):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True, use_cache=False)
        hs = torch.stack(res.hidden_states, 0)[:, :, -1, :]       # (L,B,H)
        A.append(hs.permute(1, 0, 2).to(torch.float16).cpu().numpy())
        two = res.logits[:, -1, tok_ids]                          # (B,2)
        P.append(two.argmax(-1).cpu().numpy())
        del res, hs
        torch.cuda.empty_cache()

    return np.concatenate(A, 0), np.concatenate(P, 0)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    concepts, rows, prompts = build_dataset()
    print(f"{len(concepts)} concepts, {len(rows)} prompts "
          f"({len(FAMILIES)} families x {N_CONCEPTS_PER_FAMILY} x "
          f"{N_QUERIES} queries x {len(SHOTS)} shot counts)")

    if args.smoke:
        print("\n--- sample prompt ---")
        print(prompts[0])
        print(f"\nrow: {rows[0]}")
        lab = np.array([r['label'] for r in rows])
        fam = np.array([r['family'] for r in rows])
        print(f"\nlabel balance: {np.bincount(lab)}")
        print(f"families: {dict(zip(*np.unique(fam, return_counts=True)))}")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    # catch CPU offload before burning an hour on it
    dmap = getattr(model, "hf_device_map", {})
    offloaded = [k for k, v in dmap.items() if v in ("cpu", "disk")]
    if offloaded:
        print(f"  [warn] {len(offloaded)} modules offloaded to CPU/disk. "
              f"This is the 5 s/it problem. Use a smaller model or more VRAM.")

    acts, preds = extract(model, tokenizer, prompts, device, args.batch_size)

    labels = np.array([r["label"] for r in rows])
    correct = (preds == labels).astype(int)
    for i, r in enumerate(rows):
        r["pred"] = int(preds[i])
        r["correct"] = int(correct[i])

    out = Path(args.out_dir)
    np.save(out / "activations.npy", acts)
    (out / "rows.json").write_text(json.dumps(rows))
    (out / "concepts.json").write_text(json.dumps(concepts))
    print(f"\nsaved -> activations{acts.shape}, rows.json, concepts.json")

    # ---- behavioral accuracy = Table 1 material, free from this run
    print(f"\nBEHAVIORAL ACCURACY (chance 50%)")
    ks = np.array([r["k"] for r in rows])
    for k in SHOTS:
        print(f"  {k:2d}-shot: {100*correct[ks == k].mean():5.1f}%")
    fam = np.array([r["family"] for r in rows])
    print("  by family:")
    for f in FAMILIES:
        print(f"    {f:12s}: {100*correct[fam == f].mean():5.1f}%")
    print(f"  overall: {100*correct.mean():.1f}%")
    if correct.mean() < 0.55:
        print("\n  [!] At or near chance. If the model cannot learn the rule "
              "behaviorally,\n      there is no acquired concept for Table 3 to "
              "localise. That is a\n      finding, but it changes what the probe "
              "result means -- discuss\n      with Gautam before building on it.")


if __name__ == "__main__":
    main()
