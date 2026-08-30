"""
Table 3 analysis. Run after table3_extract_v2.py.

Probes the QUERY LABEL, not concept identity. Two splits:

  transfer : train on 15 concepts, test on 5 held out. Token identity cannot
             transfer to concepts the probe never saw, so above-chance here is
             evidence of abstract rule representation. This is the headline.
  item     : train/test split within concepts by test_idx. Weaker -- a probe
             can exploit per-concept features -- but comparable to the item
             split used in the first run.

Chance is 50% for both. The permutation null is the real floor.

    python table3_analyze.py --model-name Qwen2.5-1.5B
"""

import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

SEED = 0
N_PERM = 20
L_E_MARGIN = 0.10          # probe must clear null by this much
L_S_FRACTION = 0.90        # L_S = first layer holding >=90% of peak thereafter


def probe(X, y, tr, te, C=0.1):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, C=C).fit(sc.transform(X[tr]), y[tr])
    return clf.score(sc.transform(X[te]), y[te])


def sweep(acts, y, tr, te, desc=""):
    return np.array([probe(acts[:, L, :].astype(np.float32), y, tr, te)
                     for L in tqdm(range(acts.shape[1]), desc=desc)])


def permutation_null(acts, y, tr, te, n_perm=N_PERM, stride=4):
    """Shuffle labels, same pipeline. With hidden_dim >> n_train a probe can
    beat nominal chance by fitting noise, so 50% is not the honest floor."""
    rng = np.random.RandomState(SEED)
    runs = []
    for _ in tqdm(range(n_perm), desc="null"):
        ys = rng.permutation(y)
        runs.append(np.mean([probe(acts[:, L, :].astype(np.float32), ys, tr, te)
                             for L in range(0, acts.shape[1], stride)]))
    return float(np.mean(runs)), float(np.std(runs))


def derive(acc, null, null_sd):
    thresh = max(null + 2 * null_sd, null + L_E_MARGIN)
    above = np.where(acc >= thresh)[0]
    L_E = int(above[0]) if len(above) else None
    peak = float(acc.max())
    stable = [L for L in range(len(acc))
              if all(a >= L_S_FRACTION * peak for a in acc[L:])]
    L_S = int(stable[0]) if stable else None
    return L_E, L_S, peak, thresh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="activations.npy")
    ap.add_argument("--rows", default="rows.json")
    ap.add_argument("--model-name", default="model")
    ap.add_argument("--n-train-concepts", type=int, default=15)
    ap.add_argument("--out", default="table3_row.json")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    acts = np.load(args.acts)
    rows = json.loads(Path(args.rows).read_text())
    assert len(rows) == acts.shape[0], "rows/activations length mismatch"

    y    = np.array([r["label"] for r in rows])
    cidx = np.array([r["concept_idx"] for r in rows])
    ti   = np.array([r["test_idx"] for r in rows])
    ks   = np.array([r["k"] for r in rows])
    fam  = np.array([r["family"] for r in rows])
    n, n_layers, hidden = acts.shape
    print(f"{n} rows, {n_layers} layers, hidden {hidden}, "
          f"{len(np.unique(cidx))} concepts, label balance {np.bincount(y)}")

    if "correct" in rows[0]:
        corr = np.array([r["correct"] for r in rows])
        print(f"\nbehavioral accuracy: {100*corr.mean():.1f}% (chance 50%)")
        for k in sorted(set(ks)):
            print(f"  {k:2d}-shot: {100*corr[ks == k].mean():5.1f}%")
        for f in sorted(set(fam)):
            print(f"  {f:12s}: {100*corr[fam == f].mean():5.1f}%")
        if corr.mean() < 0.55:
            print("\n  [!] Behavior at/near chance. A strong probe here would mean"
                  "\n      the rule is encoded but NOT USED -- a different claim"
                  "\n      than H2 makes. Flag before writing this up.")

    results = {}

    # ---- transfer split: held-out concepts -------------------------------
    concepts = np.unique(cidx)
    held = concepts[args.n_train_concepts:]
    tr_t, te_t = ~np.isin(cidx, held), np.isin(cidx, held)
    print(f"\n[transfer] train {tr_t.sum()} / test {te_t.sum()} "
          f"({len(held)} held-out concepts)")
    acc_t = sweep(acts, y, tr_t, te_t, "transfer")
    null_t, sd_t = permutation_null(acts, y, tr_t, te_t)
    L_E, L_S, peak, thr = derive(acc_t, null_t, sd_t)
    results["transfer"] = dict(acc=acc_t.tolist(), null=null_t, null_sd=sd_t,
                               L_E=L_E, L_S=L_S, peak=peak, thresh=thr)
    print(f"  null {100*null_t:.1f}% +/- {100*sd_t:.1f}  thresh {100*thr:.1f}%")
    print(f"  peak {100*peak:.1f}% @ L{int(acc_t.argmax())}  L_E={L_E}  L_S={L_S}")

    # ---- item split: within concepts -------------------------------------
    cut = int(np.median(ti))
    tr_i, te_i = ti < cut, ti >= cut
    print(f"\n[item] train {tr_i.sum()} / test {te_i.sum()}")
    acc_i = sweep(acts, y, tr_i, te_i, "item")
    null_i, sd_i = permutation_null(acts, y, tr_i, te_i)
    L_E_i, L_S_i, peak_i, thr_i = derive(acc_i, null_i, sd_i)
    results["item"] = dict(acc=acc_i.tolist(), null=null_i, null_sd=sd_i,
                           L_E=L_E_i, L_S=L_S_i, peak=peak_i, thresh=thr_i)
    print(f"  null {100*null_i:.1f}% +/- {100*sd_i:.1f}")
    print(f"  peak {100*peak_i:.1f}% @ L{int(acc_i.argmax())}  "
          f"L_E={L_E_i}  L_S={L_S_i}")

    # ---- per shot count, transfer split ----------------------------------
    print("\n[by shot count, transfer split]")
    by_k = {}
    for k in sorted(set(ks)):
        m = ks == k
        tr, te = tr_t & m, te_t & m
        if min(np.bincount(y[tr], minlength=2)) < 10:
            print(f"  k={k}: too few training examples, skipped")
            continue
        a = sweep(acts, y, tr, te, f"k={k}")
        by_k[int(k)] = a.tolist()
        print(f"  k={k:2d}: peak {100*a.max():5.1f}% @ L{int(a.argmax())}")
    results["by_k"] = by_k

    results["model"] = args.model_name
    Path(args.out).write_text(json.dumps(results, indent=2))

    # ---- sanity checks ---------------------------------------------------
    print("\n" + "=" * 58)
    print(f"TABLE 3 ROW -- {args.model_name}  (transfer split)")
    print(f"  Encoding Layer (L_E)          {L_E}")
    print(f"  Stable Representation (L_S)   {L_S}")
    print(f"  Probe Accuracy                {100*peak:.1f}%")
    print(f"  Permutation null              {100*null_t:.1f}%")
    print("=" * 58)

    if L_E is not None and L_E <= 1:
        print("\n  [!] L_E <= 1: decodable before the model computes anything."
              "\n      Surface-feature leak. Check the probe target.")
    if peak > 0.98:
        print("\n  [!] Ceiling accuracy. Genuine concept probes rarely saturate;"
              "\n      suspect leakage.")
    if peak < null_t + L_E_MARGIN:
        print("\n  [!] Probe never clears the null. No transferable "
              "representation\n      detected -- report as a negative result, "
              "do not force a number.")
    if L_S is not None and L_E is not None and L_S < L_E:
        print("\n  [!] L_S < L_E: thresholds are inconsistent, re-check.")

    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(9, 4.5))
            plt.plot(100*acc_t, marker="o", ms=3, label="transfer (held-out concepts)")
            plt.plot(100*acc_i, marker="s", ms=3, alpha=.6, label="item split")
            plt.axhline(100*null_t, ls="--", c="gray", label=f"null ({100*null_t:.0f}%)")
            plt.axhline(50, ls=":", c="lightgray", label="chance (50%)")
            if L_E is not None:
                plt.axvline(L_E, c="tab:orange", label=f"$L_E$={L_E}")
            if L_S is not None:
                plt.axvline(L_S, c="tab:green", label=f"$L_S$={L_S}")
            plt.xlabel("layer (0 = embeddings)"); plt.ylabel("probe accuracy (%)")
            plt.title(f"Query-label decodability by layer — {args.model_name}")
            plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
            plt.savefig("table3_curve.png", dpi=150)
            print("\nplot -> table3_curve.png")
        except ImportError:
            pass


if __name__ == "__main__":
    main()
