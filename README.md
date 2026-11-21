Išmanusis šoninių veidrodėlių reguliavimas (DI)
Grupinis projektas, tema Nr. 10 („Vairuotojo būsenos stebėsena ir saugumas“).
Idėja trumpai: kamera automobilio salone randa vairuotojo akis, programa apskaičiuoja, kaip reikia pasukti šoninius veidrodėlius, kad vairuotojas gerai matytų kelią už automobilio, ir pasiūlo naują padėtį. Veidrodėliai pasisuka tik tada, kai vairuotojas tai patvirtina mygtuku.
Autoriai: Giedrius Ivanauskas, Eimantas Macikas, Nedas Žilius, Mantas Pultinevičius, Mantas Skamaročius.
Kaip tai veikia
```
kamera → randami veido taškai (MediaPipe) → 8 skaičiai → modelis apskaičiuoja kampus
→ pridedamas vairuotojo profilis → pasiūlymas ekrane → vairuotojas patvirtina → veidrodėliai pasisuka
```
Kamera ir MediaPipe randa akis, kaktą, smakrą ir veido kraštus bei nustato, kaip pasukta galva.
Fizikinis skaičiavimas. Iš to, kaip toli viena nuo kitos akys atrodo vaizde, apskaičiuojama, kur sėdi žmogus. Tada pagal atspindžio dėsnį apskaičiuojama, kaip pasukti veidrodėlius.
DI pataisa. Paprastas mokomas modelis (Ridge regresija) ištaiso tai, kur fizikinis skaičiavimas dar suklysta.
Vairuotojo profilis. Jei vairuotojas pats pataiso veidrodėlius, sistema tai įsimena ir kitą kartą į tai atsižvelgia.
Saugumas. Sistema veidrodėlių pati nejudina: tik pasiūlo, o vairuotojas patvirtina. Važiuojant reguliuoti negalima.
Kas yra aplankuose
Failas ar aplankas	Kas jame
`src/geometry.py`	Automobilio, kameros ir veidrodėlių matmenys, atspindžio dėsnis
`src/generate_data.py`	Sugeneruoja duomenis: 80 virtualių vairuotojų po 50 kadrų
`src/features.py`	Išmeta blogus kadrus ir paskaičiuoja fizikinį įvertį
`src/models.py`	Mūsų modelis ir vairuotojo profilis
`src/train_eval.py`	Apmoko ir palygina visus modelius
`src/make_figures.py`	Nubraižo ataskaitos grafikus
`src/demo_camera.py`	Demonstracinė programa su kamera
`data/`	Sugeneruoti duomenys
`results/`	Rezultatai (skaičiai, lentelės)
`figures/`	Grafikai
`models/`	Apmokytas modelis
`docs/sutikimo_forma.md`	Sutikimo forma, jei būtų renkami tikri duomenys
Kaip paleisti
Reikia Python 3.10–3.12 (MediaPipe su naujesnėmis versijomis gali neveikti).
Windows (PowerShell), projekto aplanke:
```
py -V:3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Patikrinti rezultatus (kameros nereikia):
```
.venv\Scripts\python.exe src\generate_data.py      # 1) sugeneruoja duomenis
.venv\Scripts\python.exe src\train_eval.py         # 2) apmoko ir palygina modelius (~2 min.)
.venv\Scripts\python.exe src\make_figures.py       # 3) nubraižo grafikus
```
Linux / macOS: tos pačios komandos, tik vietoj `.venv\Scripts\python.exe` rašykite `.venv/bin/python`, o viską iš karto galima paleisti su `./run_all.sh`. Taip pat galima naudoti Docker:
```
docker build -t veidrodeliai . && docker run --rm -v "$PWD/results:/app/results" veidrodeliai
```
Demonstracija su kamera
```
.venv\Scripts\python.exe src\demo_camera.py
```
Atsisėskite maždaug 60 cm nuo kameros, žiūrėkite tiesiai ir paspauskite C.
Pasislinkite į šoną arba pasilenkite. Po sekundės ekrane atsiras užrašas „SIULOMA pakeisti padeti“.
Paspauskite ENTER: veidrodėliai apačioje dešinėje pasisuks.
Kiti klavišai:
Klavišas	Ką daro
V	Imituoja važiavimą: tada reguliuoti negalima
W/S/A/D, I/K/J/L	Rankiniu būdu pataiso kairįjį / dešinįjį veidrodėlį
P	Įrašo pataisymą į profilį
1 / 2 / 3	Pasirenka vairuotojo profilį
Q	Išjungia programą
Apačioje dešinėje rodomas automobilis iš viršaus: balta linija – veidrodėlio stiklas, oranžinis plotas – ką vairuotojas mato veidrodėlyje. Kad pokyčiai matytųsi, ekrane jie padidinti 5 kartus (tikri skaičiai rodomi viršuje).
Pirmą kartą programa atsisiunčia nedidelį MediaPipe modelį (~4 MB). Jei MediaPipe neįsidiegia, programa veikia su OpenCV, tik nemato galvos pasukimo.
Rezultatai
Tikrinta su 20 vairuotojų, kurių modelis nebuvo matęs:
Modelis	Vidutinė klaida	Kiek kartų visi 4 kampai pataikė ±0,5° tikslumu
Paprastas modelis (tiesinė regresija)	0,54°	11 %
Mūsų sistema (fizika + DI + profilis + 10 kadrų vidurys)	0,12°	100 % (80 grupių po 10 kadrų)
Išbandžius su tikra nešiojamo kompiuterio kamera, visas vieno kadro apdorojimas užtruko 18–19 ms (leidžiama iki 200 ms).
Daugiau – faile `results/metrics_B_pageidaujami.csv` ir ataskaitoje.
Svarbu žinoti
Duomenys sugeneruoti kompiuteriu, o ne surinkti automobilyje. Kodėl taip ir kodėl tokie duomenys tinka, paaiškinta ataskaitos 4 skyriuje. Rezultatai rodo, kad idėja veikia, bet nėra išmatuoti tikrame automobilyje.
Privatumas: programa vaizdo neįrašo ir internetu nieko nesiunčia (išskyrus vieną kartą atsisiunčiamą modelį). Profiliuose (`profiles.json`) saugomi tik 4 kampų pataisymai, jokių veido duomenų.
