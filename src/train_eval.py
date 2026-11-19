"""
Modeliu treniravimas ir eksperimentai.

Paleidimas (po generate_data.py):  python src/train_eval.py
Rezultatai: results/results.json, results/metrics_*.csv, results/test_predictions.csv, models/

Eksperimentu eiga:
  1. Duomenu valymas.
  2. Padalijimas pagal vairuotojus: 60 treniravimui, 20 testui (testo vairuotoju modelis nemato).
  3. A eksperimentas - bendras modelis (be pageidavimu): ar ML pagerina klasikine geometrija?
     Hiperparametrai parenkami GroupKFold kryžmine patikra tik treniravimo aibeje.
  4. B eksperimentas - pagrindinis: vairuotojo pageidaujami kampai.
     Baseline (LinearRegression) vs Proposed (hibridinis modelis + asmeninis profilis + 10 kadru mediana).
  5. Papildomai: duomenu nutekejimo patikra, profilio pataisymu skaicius, sprendimo
     "reguliuoti / nereguliuoti" konfuzijos matrica, inferencijos laikas, XAI.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, precision_score, recall_score, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, GEO_PHYS, build_features, clean
from geometry import TARGETS
from models import DriverProfile, GeometricModel, HybridMirrorModel

SEED = 42
TOL = 0.5            # leistina paklaida kiekvienam kampui, laipsniai
K_PROFILE = 3        # kiek rankiniu pataisymu turi profilis
EVAL_FROM = 5        # vertinami kadrai nuo 6-o (pirmi 5 rezervuoti profiliui)
WINDOW = 10          # kadru langas glodinimui
ROOT = Path(__file__).resolve().parent.parent
PREF = [t + "_pref" for t in TARGETS]


def metrics(y, p):
    err = np.abs(np.asarray(y) - np.asarray(p))
    return {"MAE": float(err.mean()), "RMSE": float(np.sqrt((err ** 2).mean())), "R2": float(r2_score(y, p)),
            "Tikslumas": float((err <= TOL).all(axis=1).mean()),
            **{f"MAE_{t}": float(err[:, i].mean()) for i, t in enumerate(TARGETS)}}


def cv_select(name, make, grid, tr, y, feats):
    """GroupKFold kryžminė patikra treniravimo aibėje."""
    res = []
    for g in grid:
        maes = []
        for a, b in GroupKFold(n_splits=5).split(tr, groups=tr["driver_id"]):
            m = make(g).fit(tr.iloc[a][feats], y[a])
            maes.append(mean_absolute_error(y[b], m.predict(tr.iloc[b][feats])))
        res.append({"modelis": name, **g, "cv_MAE": float(np.mean(maes))})
    return min(res, key=lambda r: r["cv_MAE"]), res


def windows(te):
    """Kiekvienam testo vairuotojui - nepersidengiantys WINDOW kadru langai vertinimo dalyje."""
    out = []
    for _, g in te.assign(_i=np.arange(len(te))).groupby("driver_id", sort=False):
        idx = g["_i"].to_numpy()[EVAL_FROM:]
        out += [idx[s:s + WINDOW] for s in range(0, len(idx) - WINDOW + 1, WINDOW)]
    return out


def main():
    rng = np.random.default_rng(SEED)
    for d in ("results", "models"):
        (ROOT / d).mkdir(exist_ok=True)
    log = {}
    raw = pd.read_csv(ROOT / "data" / "dataset.csv")
    log["kadru_is_viso"] = len(raw)
    df = build_features(clean(raw, log))

    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
                          .split(df, groups=df["driver_id"]))
    tr, te = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    log.update(train_vairuotojai=int(tr.driver_id.nunique()), test_vairuotojai=int(te.driver_id.nunique()),
               train_kadrai=len(tr), test_kadrai=len(te))
    eval_mask = te.groupby("driver_id").cumcount().to_numpy() >= EVAL_FROM

    # ================= hiperparametrai (tik treniravimo aibėje) =================
    hgb_best, hgb_cv = cv_select(
        "HGB", lambda g: MultiOutputRegressor(HistGradientBoostingRegressor(
            max_iter=600, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            random_state=SEED, **g)),
        [dict(learning_rate=lr, max_leaf_nodes=ln, l2_regularization=l2)
         for lr in (0.05, 0.1) for ln in (7, 15, 31) for l2 in (0.0, 1.0)],
        tr, tr[TARGETS].to_numpy(), FEATURES + GEO_PHYS)
    mlp_best, mlp_cv = cv_select(
        "MLP", lambda g: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=g["sluoksniai"], alpha=g["alpha"], max_iter=2000, early_stopping=True,
            n_iter_no_change=50, random_state=SEED)),
        [dict(sluoksniai=h, alpha=a) for h in ((32,), (64, 32)) for a in (1e-3, 1e-2)],
        tr, tr[TARGETS].to_numpy(), FEATURES + GEO_PHYS)
    hyb_best, hyb_cv = cv_select("Hibridinis", lambda g: HybridMirrorModel(**g),
                                 [dict(alpha=a) for a in (0.01, 0.1, 1.0, 10.0)],
                                 tr, tr[TARGETS].to_numpy(), list(df.columns))
    log["hiperparametru_paieska"] = hgb_cv + mlp_cv + hyb_cv
    hgb_params = {k: hgb_best[k] for k in ("learning_rate", "max_leaf_nodes", "l2_regularization")}
    log["geriausi"] = {"HGB": hgb_params, "MLP": {k: mlp_best[k] for k in ("sluoksniai", "alpha")},
                       "Hibridinis": {"alpha": hyb_best["alpha"]}}

    def make_models():
        return {
            "Vidurkio prognozė": (DummyRegressor(), FEATURES),
            "Geometrinis (be ML)": (GeometricModel("naive"), list(df.columns)),
            "Baseline: LinearRegression": (LinearRegression(), FEATURES),
            "HGB (LightGBM tipo)": (MultiOutputRegressor(HistGradientBoostingRegressor(
                max_iter=600, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
                random_state=SEED, **hgb_params)), FEATURES + GEO_PHYS),
            "MLP neuroninis tinklas": (make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=mlp_best["sluoksniai"], alpha=mlp_best["alpha"], max_iter=2000,
                early_stopping=True, n_iter_no_change=50, random_state=SEED)), FEATURES + GEO_PHYS),
            "Fizikinis (be ML)": (GeometricModel("phys"), list(df.columns)),
            "Hibridinis (fizika + Ridge)": (HybridMirrorModel(alpha=hyb_best["alpha"]), list(df.columns)),
        }

    # ================= A eksperimentas: bendri kampai (BGE) =================
    A, A_pred, fitted = {}, {}, {}
    for name, (m, feats) in make_models().items():
        m.fit(tr[feats], tr[TARGETS].to_numpy())
        p = m.predict(te[feats])
        A[name] = metrics(te[TARGETS].to_numpy()[eval_mask], p[eval_mask])
        A_pred[name] = p
        fitted[name] = (m, feats)
        print(f"A  {name:32s} MAE={A[name]['MAE']:.3f}  Tikslumas={A[name]['Tikslumas']:.3f}")

    # Vidinė paklaidos analizė: kiek paklaidos lemia vairuotojas (nuolatinis poslinkis), o kiek - kadras
    r = te[TARGETS].to_numpy() - A_pred["Hibridinis (fizika + Ridge)"]
    per_driver = pd.DataFrame(r, columns=TARGETS).assign(d=te.driver_id).groupby("d").mean()
    log["paklaidos_dalis_del_vairuotojo"] = {t: float(per_driver[t].var() / r[:, i].var())
                                             for i, t in enumerate(TARGETS)}
    drivers = pd.read_csv(ROOT / "data" / "drivers.csv").set_index("driver_id")
    log["koreliacija_poslinkis_vs_IPD"] = {t: float(np.corrcoef(per_driver[t], drivers.loc[per_driver.index, "ipd_m"])[0, 1])
                                          for t in TARGETS}

    # ================= B eksperimentas: pageidaujami kampai =================
    Y = te[PREF].to_numpy()
    B, B_pred = {}, {}
    for name, (m, feats) in make_models().items():
        m.fit(tr[feats], tr[PREF].to_numpy())
        B_pred[name] = m.predict(te[feats])
        B[name] = metrics(Y[eval_mask], B_pred[name][eval_mask])

    def with_profile(p, k, shrink=0.5):
        out = p.copy()
        for _, g in te.assign(_i=np.arange(len(te))).groupby("driver_id", sort=False):
            idx = g["_i"].to_numpy()
            prof = DriverProfile(shrink=shrink)
            for j in idx[:k]:
                prof.add_correction(Y[j], p[j])
            out[idx] = prof.apply(p[idx])
        return out

    hyb = B_pred["Hibridinis (fizika + Ridge)"]
    hyb_prof = with_profile(hyb, K_PROFILE)
    B["Hibridinis + profilis"] = metrics(Y[eval_mask], hyb_prof[eval_mask])
    B["HGB + profilis"] = metrics(Y[eval_mask], with_profile(B_pred["HGB (LightGBM tipo)"], K_PROFILE)[eval_mask])
    B["Baseline + profilis"] = metrics(Y[eval_mask], with_profile(B_pred["Baseline: LinearRegression"], K_PROFILE)[eval_mask])

    # 10 kadrų mediana (realiu laiku sistema reguliuoja pagal kelių kadrų įvertį)
    W = windows(te)
    Yw = np.array([Y[w].mean(0) for w in W])
    final_pred = np.array([np.median(hyb_prof[w], 0) for w in W])
    base_w = np.array([np.median(B_pred["Baseline: LinearRegression"][w], 0) for w in W])
    B["Baseline: LinearRegression (10 kadrų mediana)"] = metrics(Yw, base_w)
    B["Proposed: Hibridinis + profilis + 10 kadrų mediana"] = metrics(Yw, final_pred)
    for n, v in B.items():
        print(f"B  {n:50s} MAE={v['MAE']:.3f}  Tikslumas={v['Tikslumas']:.3f}")

    # Profilio dydžio įtaka
    log["profilio_pataisymai"] = {}
    for k in (0, 1, 2, 3, 5):
        pk = with_profile(hyb, k)
        pw = np.array([np.median(pk[w], 0) for w in W])
        log["profilio_pataisymai"][k] = {"MAE": metrics(Yw, pw)["MAE"], "Tikslumas": metrics(Yw, pw)["Tikslumas"]}

    # ================= Nesėkmė: atsitiktinis padalijimas =================
    a, b = train_test_split(np.arange(len(df)), test_size=0.25, random_state=SEED)
    hg = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=600, early_stopping=True, random_state=SEED,
                                                            **hgb_params))
    hg.fit(df.loc[a, FEATURES + GEO_PHYS], df.loc[a, PREF])
    log["nutekejimas"] = {"atsitiktinis_padalijimas": metrics(df.loc[b, PREF].to_numpy(),
                                                             hg.predict(df.loc[b, FEATURES + GEO_PHYS])),
                          "padalijimas_pagal_vairuotojus": B["HGB (LightGBM tipo)"]}

    # ================= Sprendimas: reguliuoti / nereguliuoti =================
    # Imituojama esama veidrodžių padėtis: pusė atvejų - po kito vairuotojo, pusė - jau beveik tinkama.
    Yev = Yw
    cur = Yev[rng.permutation(len(Yev))]
    near = rng.random(len(Yev)) < 0.5
    cur[near] = Yev[near] + rng.normal(0, 0.3, (near.sum(), 4))
    need = (np.abs(Yev - cur) > TOL).any(axis=1)
    log["sprendimas"] = {}
    for name, P in (("Baseline: LinearRegression", base_w), ("Proposed", final_pred)):
        pred_need = (np.abs(P - cur) > TOL).any(axis=1)
        log["sprendimas"][name] = {
            "cm": confusion_matrix(need, pred_need, labels=[False, True]).tolist(),
            "Accuracy": float((need == pred_need).mean()), "Precision": float(precision_score(need, pred_need)),
            "Recall": float(recall_score(need, pred_need)), "F1": float(f1_score(need, pred_need))}

    # ================= Inferencijos laikas =================
    hyb_model = HybridMirrorModel(alpha=hyb_best["alpha"]).fit(tr, tr[PREF].to_numpy())
    one = te.iloc[[0]]
    hyb_model.predict(one)
    t0 = time.perf_counter()
    for _ in range(500):
        build_features(te[FEATURES].iloc[[0]])
        hyb_model.predict(one)
    log["inferencija_ms"] = (time.perf_counter() - t0) / 500 * 1000

    # ================= XAI =================
    pi = permutation_importance(hyb_model, te, Y, scoring="neg_mean_absolute_error", n_repeats=10,
                                random_state=SEED)
    names = list(te.columns)
    imp = {n: float(v) for n, v in zip(names, pi.importances_mean) if n in HybridMirrorModel.FEATS}
    log["pozymiu_svarba"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
    log["ridge_koeficientai"] = {f: hyb_model.reg_.coef_[:, i].round(4).tolist()
                                 for i, f in enumerate(HybridMirrorModel.FEATS)}

    # ================= Mokymosi kreivės =================
    curves = {"HGB": [{"train": (-np.array(e.train_score_)).tolist(), "val": (-np.array(e.validation_score_)).tolist()}
                      for e in fitted["HGB (LightGBM tipo)"][0].estimators_],
              "MLP": {"loss": fitted["MLP neuroninis tinklas"][0][-1].loss_curve_,
                      "val_r2": fitted["MLP neuroninis tinklas"][0][-1].validation_scores_}}
    # Mokymosi kreivė pagal duomenų kiekį (kiek vairuotojų reikia)
    lc = []
    tr_drivers = tr.driver_id.unique()
    for n in (5, 10, 20, 40, 60):
        sel = tr[tr.driver_id.isin(tr_drivers[:n])]
        row = {"vairuotojai": int(n)}
        for name in ("Baseline: LinearRegression", "HGB (LightGBM tipo)", "Hibridinis (fizika + Ridge)"):
            m, feats = make_models()[name]
            m.fit(sel[feats], sel[TARGETS].to_numpy())
            row[name] = float(mean_absolute_error(te[TARGETS].to_numpy()[eval_mask], m.predict(te[feats])[eval_mask]))
        lc.append(row)
    curves["pagal_duomenu_kieki"] = lc

    # ================= Išsaugojimas =================
    joblib.dump({"model": hyb_model, "tol": TOL, "targets": TARGETS}, ROOT / "models" / "hybrid_model.joblib")
    pd.DataFrame(A).T.to_csv(ROOT / "results" / "metrics_A_bendri.csv")
    pd.DataFrame(B).T.to_csv(ROOT / "results" / "metrics_B_pageidaujami.csv")
    pdf = te[["driver_id", "yaw_true", "pitch_true"] + TARGETS + PREF].copy()
    for name, p in A_pred.items():
        for i, t in enumerate(TARGETS):
            pdf[f"A|{name}|{t}"] = p[:, i]
    for i, t in enumerate(TARGETS):
        pdf[f"B|Baseline|{t}"] = B_pred["Baseline: LinearRegression"][:, i]
        pdf[f"B|Proposed|{t}"] = hyb_prof[:, i]
    pdf["eval"] = eval_mask
    pdf.to_csv(ROOT / "results" / "test_predictions.csv", index=False)
    json.dump({"A": A, "B": B, **log, "curves": curves}, open(ROOT / "results" / "results.json", "w"),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in log.items() if k not in ("hiperparametru_paieska", "ridge_koeficientai")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
