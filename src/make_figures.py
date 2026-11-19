"""
Ataskaitos paveikslai is results/ ir data/.  Paleidimas:  python src/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from features import FEATURES, build_features, clean  # noqa: E402
from geometry import TARGETS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#c9c8c2",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True})
LT = {"L_yaw": "Kairysis, horiz.", "L_pitch": "Kairysis, vert.", "R_yaw": "Dešinysis, horiz.",
      "R_pitch": "Dešinysis, vert."}


def save(name):
    plt.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close()


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.grid(False)

    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec="#1F3864", lw=1.2))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.6)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#1F3864"))
    box(0.1, 2.9, 1.6, 1.1, "IR kamera\n+ 850 nm LED", "#DDEBF7")
    box(2.1, 2.9, 1.8, 1.1, "Veido orientyrai\nMediaPipe Face\nLandmarker", "#DDEBF7")
    box(4.3, 2.9, 1.7, 1.1, "Požymiai (8)\n+ valymas", "#FCE4D6")
    box(6.4, 2.9, 2.0, 1.1, "Hibridinis modelis\nfizika + Ridge\n(4 kampai)", "#FCE4D6")
    box(8.8, 2.9, 2.1, 1.1, "Asmeninis profilis\n+ 10 kadrų mediana", "#FCE4D6")
    box(8.8, 1.2, 2.1, 1.1, "Vairuotojo\npatvirtinimas\n(mygtukas / OLED)", "#E2EFDA")
    box(6.4, 1.2, 2.0, 1.1, "PI reguliatorius\n4 varikliai +\npadėties jutikliai", "#FFF2CC")
    box(4.3, 1.2, 1.7, 1.1, "Veidrodėliai\n(K ir D)", "#EDEDED")
    box(0.1, 1.2, 3.8, 1.1, "Rankinis pataisymas\n→ papildo profilį", "#E2EFDA")
    box(4.3, 0.05, 6.6, 0.7, "Sąlyga: automobilis stovi (greitis = 0) • vaizdas neįrašomas • veikia be interneto", "#F2F2F2")
    for a in ((1.7, 3.45, 2.1, 3.45), (3.9, 3.45, 4.3, 3.45), (6.0, 3.45, 6.4, 3.45), (8.4, 3.45, 8.8, 3.45),
              (9.85, 2.9, 9.85, 2.3), (8.8, 1.75, 8.4, 1.75), (6.4, 1.75, 6.0, 1.75), (4.3, 1.75, 3.9, 1.75)):
        arr(*a)
    ax.annotate("", xy=(9.2, 2.9), xytext=(2.0, 2.3), arrowprops=dict(arrowstyle="->", lw=1.0, color="#7f7f7f",
                                                                     connectionstyle="arc3,rad=-0.12", ls="--"))
    save("fig01_architektura.png")


def fig_data(df_raw, drv):
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.6))
    a = axs[0, 0]
    bins = np.linspace(1.5, 2.0, 26)
    a.hist(drv.height_m[drv.sex == 1], bins, color=BLUE, alpha=0.85, label="Vyrai", ec="white")
    a.hist(drv.height_m[drv.sex == 0], bins, color=ORANGE, alpha=0.75, label="Moterys", ec="white")
    a.set(xlabel="Ūgis, m", ylabel="Vairuotojų sk.", title="a) Vairuotojų ūgis (n = 80)"); a.legend(frameon=False)
    a = axs[0, 1]
    a.hist(drv.ipd_m * 1000, 20, color=BLUE, ec="white")
    a.set(xlabel="Tarpvyzdinis atstumas, mm", ylabel="Vairuotojų sk.", title="b) Atstumas tarp akių")
    a = axs[1, 0]
    a.hist(df_raw.yaw_true, 36, color=BLUE, ec="white")
    a.set(xlabel="Galvos pasukimas, °", ylabel="Kadrų sk.", title="c) Galvos pasukimas kadruose (n = 4000)")
    a = axs[1, 1]
    sc = a.scatter(df_raw.eye_x * 100, df_raw.eye_z * 100, c=df_raw.merge(drv, on="driver_id").height_m,
                   cmap="Blues", s=6, vmin=1.45)
    a.set(xlabel="Akių padėtis x (į priekį), cm", ylabel="Akių padėtis z (aukštis), cm",
          title="d) Akių padėtis automobilyje")
    cb = plt.colorbar(sc, ax=a); cb.set_label("Ūgis, m")
    plt.tight_layout(); save("fig02_duomenys.png")


def fig_corr(df):
    cols = FEATURES + TARGETS
    c = df[cols].corr().loc[FEATURES, TARGETS]
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.grid(False)
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4), [LT[t] for t in TARGETS], rotation=25, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    for i in range(len(FEATURES)):
        for j in range(4):
            v = c.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.6 else INK)
    plt.colorbar(im, ax=ax, label="Pirsono koreliacija")
    save("fig03_koreliacija.png")


def fig_training(res):
    cur = res["curves"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    h = cur["HGB"][0]
    a = axs[0]
    a.plot(np.array(h["train"]), color=ORANGE, lw=2, label="Treniravimo")
    a.plot(np.array(h["val"]), color=BLUE, lw=2, label="Validacijos")
    a.set(xlabel="Iteracija (medžių skaičius)", ylabel="Nuostolis (MSE / 2)", title="a) HGB, kairysis horiz. kampas")
    a.legend(frameon=False)
    a = axs[1]
    a.plot(cur["MLP"]["loss"], color=BLUE, lw=2)
    a.set(xlabel="Epocha", ylabel="Treniravimo nuostolis", title="b) MLP neuroninis tinklas", yscale="log")
    a = axs[2]
    lc = pd.DataFrame(cur["pagal_duomenu_kieki"])
    for name, col, lab in (("Baseline: LinearRegression", ORANGE, "Baseline (LR)"),
                           ("HGB (LightGBM tipo)", GRAY, "HGB"),
                           ("Hibridinis (fizika + Ridge)", BLUE, "Hibridinis")):
        a.plot(lc.vairuotojai, lc[name], marker="o", ms=6, lw=2, color=col, label=lab)
    a.set(xlabel="Treniravimo vairuotojų sk.", ylabel="Testo MAE, °", title="c) Kiek duomenų reikia")
    a.legend(frameon=False)
    plt.tight_layout(); save("fig04_mokymasis.png")


def fig_compare(res):
    B = pd.DataFrame(res["B"]).T
    order = ["Vidurkio prognozė", "Geometrinis (be ML)", "Fizikinis (be ML)", "MLP neuroninis tinklas",
             "HGB (LightGBM tipo)", "Baseline: LinearRegression", "Hibridinis (fizika + Ridge)",
             "HGB + profilis", "Baseline + profilis", "Hibridinis + profilis",
             "Proposed: Hibridinis + profilis + 10 kadrų mediana"]
    B = B.loc[order]
    cols = [BLUE if n.startswith("Proposed") else ORANGE if n.startswith("Baseline:") else GRAY for n in B.index]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    y = np.arange(len(B))
    axs[0].barh(y, B.MAE, color=cols, height=0.6)
    axs[0].axvline(0.5, color=MUTED, ls="--", lw=1)
    axs[0].text(0.51, len(B) - 0.6, "leistina 0,5°", color=MUTED, fontsize=8)
    axs[0].set(yticks=y, yticklabels=B.index, xlabel="MAE, ° (mažiau - geriau)", title="Vidutinė paklaida")
    axs[1].barh(y, B.Tikslumas * 100, color=cols, height=0.6)
    axs[1].axvline(80, color=MUTED, ls="--", lw=1)
    axs[1].set(xlabel="Tikslumas, % (visi 4 kampai ≤ 0,5°)", title="Tikslumas")
    for a, col, f in ((axs[0], "MAE", "{:.2f}"), (axs[1], "Tikslumas", "{:.0%}")):
        for i, v in enumerate(B[col]):
            a.text(v * (100 if col == "Tikslumas" else 1), i, " " + f.format(v), va="center", fontsize=8, color=INK)
    axs[0].invert_yaxis()
    plt.tight_layout(); save("fig05_palyginimas.png")


def fig_scatter(pred):
    ev = pred[pred["eval"]]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    t = "L_yaw"
    y = ev[t + "_pref"]
    for a, key, col, title in ((axs[0], "Baseline", ORANGE, "Baseline (LinearRegression)"),
                               (axs[1], "Proposed", BLUE, "Proposed (hibridinis + profilis)")):
        p = ev[f"B|{key}|{t}"]
        lo, hi = y.min() - 0.3, y.max() + 0.3
        a.fill_between([lo, hi], [lo - 0.5, hi - 0.5], [lo + 0.5, hi + 0.5], color="#ecebe7", lw=0)
        a.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        a.scatter(y, p, s=9, color=col, alpha=0.6)
        mae = np.abs(y - p).mean()
        a.set(title=f"{title}\nšio kampo MAE = {mae:.2f}°", xlabel="Pageidaujamas kairiojo veidr. horiz. kampas, °")
    axs[0].set_ylabel("Prognozė, °")
    axs[1].text(0.02, 0.95, "pilka juosta: ±0,5°", transform=axs[1].transAxes, color=MUTED, fontsize=8, va="top")
    plt.tight_layout(); save("fig06_prognozes.png")


def fig_profile(res):
    p = pd.DataFrame(res["profilio_pataisymai"]).T
    p.index = p.index.astype(int)
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axs[0].plot(p.index, p.MAE, marker="o", ms=7, lw=2, color=BLUE)
    axs[0].axhline(0.5, color=MUTED, ls="--", lw=1)
    axs[0].set(xlabel="Rankinių pataisymų sk. profilyje", ylabel="MAE, °", title="a) Paklaida")
    axs[1].plot(p.index, p.Tikslumas * 100, marker="o", ms=7, lw=2, color=BLUE)
    axs[1].axhline(80, color=MUTED, ls="--", lw=1)
    axs[1].set(xlabel="Rankinių pataisymų sk. profilyje", ylabel="Tikslumas, %", title="b) Tikslumas", ylim=(0, 105))
    plt.tight_layout(); save("fig07_profilis.png")


def fig_confusion(res):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, (name, title) in zip(axs, (("Baseline: LinearRegression", "Baseline"), ("Proposed", "Proposed"))):
        cm = np.array(res["sprendimas"][name]["cm"])
        a.grid(False)
        a.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                       color="white" if cm[i, j] > cm.sum() / 2 else INK)
        a.set_xticks([0, 1], ["Nereguliuoti", "Reguliuoti"]); a.set_yticks([0, 1], ["Nereguliuoti", "Reguliuoti"])
        f1 = res["sprendimas"][name]["F1"]
        a.set(xlabel="Sistemos sprendimas", ylabel="Tikras poreikis", title=f"{title}  (F1 = {f1:.2f})")
    plt.tight_layout(); save("fig08_konfuzija.png")


def fig_xai(res):
    imp = pd.Series(res["pozymiu_svarba"])
    phys = imp[[i for i in imp.index if i.startswith("gp_")]]
    rest = imp[[i for i in imp.index if not i.startswith("gp_")]].sort_values()
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    axs[0].barh([LT[i[3:]] for i in phys.index], phys.values, color=BLUE, height=0.55)
    axs[0].set(xlabel="MAE padidėjimas, °", title="a) Fizikinio įverčio požymiai")
    axs[1].barh(rest.index, rest.values, color=AQUA, height=0.55)
    axs[1].set(xlabel="MAE padidėjimas, °", title="b) Pataisos (ML) požymiai")
    plt.tight_layout(); save("fig09_xai.png")


def fig_failure(pred):
    ev = pred[pred["eval"]].copy()
    ev["bin"] = pd.cut(ev.yaw_true.abs(), [0, 5, 10, 20, 30, 36], right=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for name, col, lab in (("Geometrinis (be ML)", GRAY, "Geometrinis (be ML)"),
                           ("Baseline: LinearRegression", ORANGE, "Baseline (LR)"),
                           ("Hibridinis (fizika + Ridge)", BLUE, "Hibridinis")):
        err = np.mean([np.abs(ev[t] - ev[f"A|{name}|{t}"]) for t in TARGETS], axis=0)
        m = pd.Series(err, index=ev.index).groupby(ev["bin"], observed=True).mean()
        ax.plot([str(i).replace("[", "").replace(")", "").replace(", ", "–") for i in m.index], m.values,
                marker="o", ms=7, lw=2, color=col, label=lab)
    ax.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax.set(xlabel="|Galvos pasukimas|, °", ylabel="MAE, °", title="Paklaida pagal galvos pasukimą (A eksperimentas)")
    ax.legend(frameon=False)
    plt.tight_layout(); save("fig10_nesekmes.png")


def main():
    res = json.load(open(ROOT / "results" / "results.json"))
    raw = pd.read_csv(ROOT / "data" / "dataset.csv")
    drv = pd.read_csv(ROOT / "data" / "drivers.csv")
    pred = pd.read_csv(ROOT / "results" / "test_predictions.csv")
    fig_architecture()
    fig_data(raw, drv)
    fig_corr(build_features(clean(raw)))
    fig_training(res)
    fig_compare(res)
    fig_scatter(pred)
    fig_profile(res)
    fig_confusion(res)
    fig_xai(res)
    fig_failure(pred)
    print("Paveikslai issaugoti:", FIG)


if __name__ == "__main__":
    main()
