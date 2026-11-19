# Išmanusis šoninių veidrodėlių reguliavimas (AI)

Grupinis projektas, tema Nr. 10 („Vairuotojo būsenos stebėsena ir saugumas“): **naudojant kamerą ir veido atpažinimą,
automatiškai pritaikyti šoninius veidrodėlius pagal konkretaus vairuotojo ūgį, sėdėjimo padėtį ir pageidavimus.**

Autoriai: Giedrius Ivanauskas, Eimantas Macikas, Nedas Žilius, Mantas Pultinevičius.

## Kaip tai veikia

```
IR kamera → veido orientyrai (MediaPipe) → 8 požymiai → hibridinis modelis (fizika + Ridge)
→ asmeninis profilis → 10 kadrų mediana → PASIŪLYMAS → vairuotojas patvirtina → varikliai
```

* **Fizikinis įvertis** apskaičiuoja akių padėtį (kameros modelis, objektyvo iškraipymai, galvos pasukimas) ir veidrodėlių
  kampus pagal atspindžio dėsnį.
* **ML pataisa (Ridge regresija)** mokosi to, ko fizikinis modelis nepaaiškina.
* **Asmeninis profilis** įsimena vairuotojo rankinius pataisymus (jo pageidavimus ir individualų akių atstumą).
* AI **niekada nejudina veidrodėlių pats**: tik pasiūlo, o vairuotojas patvirtina mygtuku, ir tik automobiliui stovint.

## Projekto struktūra

| Kelias | Turinys |
|---|---|
| `src/geometry.py` | Automobilio ir kameros geometrija, atspindžio dėsnis |
| `src/generate_data.py` | Sintetinio duomenų rinkinio generatorius (80 vairuotojų × 50 kadrų) |
| `src/features.py` | Duomenų valymas, naivus ir fizikinis geometrinis įvertis |
| `src/models.py` | Hibridinis modelis ir vairuotojo profilis |
| `src/train_eval.py` | Treniravimas, hiperparametrų paieška, visi eksperimentai |
| `src/make_figures.py` | Ataskaitos paveikslai |
| `src/demo_camera.py` | Gyva demonstracija su internetine kamera |
| `data/` | Sugeneruoti duomenys (`dataset.csv`, `drivers.csv`) |
| `results/` | Metrikos (`metrics_*.csv`, `results.json`), testo prognozės |
| `figures/` | Paveikslai (fig01–fig10) |
| `models/` | Apmokytas modelis (`hybrid_model.joblib`) |
| `docs/sutikimo_forma.md` | Dalyvio sutikimo forma realiems duomenims rinkti |

## Paleidimas

Reikia Python 3.10–3.12 (MediaPipe kol kas nepalaiko naujesnių versijų).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

./run_all.sh                       # arba trys komandos:
python src/generate_data.py        # 1) duomenys
python src/train_eval.py           # 2) modeliai ir eksperimentai (~2 min.)
python src/make_figures.py         # 3) paveikslai
```

Be kameros eksperimentus galima atkartoti ir per Docker:

```bash
docker build -t veidrodeliai . && docker run --rm -v "$PWD/results:/app/results" veidrodeliai
```

### Gyva demonstracija

```bash
python src/demo_camera.py
```

1. Atsisėskite ~60 cm nuo kameros, žiūrėkite tiesiai ir paspauskite **C** (kalibravimas).
2. Pasislinkite, pasilenkite – ekrane atsiras **pasiūlymas** naujiems kampams.
3. **ENTER** – patvirtinti (veidrodėliai ekrane pasisuka). **V** – imituoti važiavimą (reguliavimas blokuojamas).
4. **W/S/A/D** ir **I/K/J/L** – rankinis pataisymas, **P** – išsaugoti į profilį, **1/2/3** – profilio pasirinkimas.

Pirmą kartą automatiškai atsisiunčiamas MediaPipe modelis (~4 MB). Jei MediaPipe neįsidiegia, programa naudoja OpenCV
(be galvos pasukimo įverčio). Jei pasukus galvą į kairę rodomas neigiamas `yaw`, faile `demo_camera.py` pakeiskite
`YAW_SIGN = -1`.

## Pagrindiniai rezultatai (testo vairuotojai, kurių modelis nematė)

| Modelis | MAE, ° | Tikslumas (visi 4 kampai ≤ 0,5°) |
|---|---|---|
| Baseline: LinearRegression | 0,544 | 11,3 % |
| Proposed: hibridinis + profilis + 10 kadrų mediana | 0,117 | 100 % (80 langų) |

Išsamiai – `results/metrics_B_pageidaujami.csv` ir ataskaitoje.

## Svarbu

* Duomenys yra **sintetiniai** (ataskaitos IV skyrius paaiškina, kodėl ir kaip jie pagrįsti). Rezultatai rodo metodo
  veikimą, bet nėra išmatuoti tikrame automobilyje.
* Demonstracija vaizdo **neįrašo** ir neveikia per internetą (išskyrus vienkartinį modelio atsisiuntimą).
  Profiliuose (`profiles.json`) saugomi tik kampų poslinkiai, jokių biometrinių duomenų.
