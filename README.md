# Predictive Maintenance with Soft Computing (GA + ANN + Fuzzy Logic)

Predicts how much life an aircraft engine (or a factory bearing) has left from its sensor readings, then turns that into a **Normal / Warning / Critical** alert you can see on a monitor-style dashboard with sound.

| Soft computing part | What it does here |
|---|---|
| **Genetic Algorithm** | Picks the sensors that help predict remaining life, and tunes the neural network (layers, regularisation, learning rate). A **binary PSO** run is included to compare against the GA. |
| **Neural network (MLP)** | Reads the chosen sensors and predicts life left. |
| **Fuzzy inference** | Combines predicted life left with the wear rate and gives a risk score and alert level, using 5 readable rules. |

Everything is software only. The dashboard replays recorded run-to-failure data as if it were live.

---

## 1. Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## 2. Datasets to download

You need **one dataset per machine type**. Do the aircraft one first.

### Aircraft: NASA C-MAPSS (required)

- **Name:** *Turbofan Engine Degradation Simulation Data Set* (C-MAPSS), file `CMAPSSData.zip`
- **Where:** NASA Prognostics Center of Excellence data repository. Search "NASA C-MAPSS" or open the NASA catalog entry [Simulating Degradation Data for Prognostic Algorithm Development](https://data.nasa.gov/dataset/simulating-degradation-data-for-prognostic-algorithm-development). Copies are also on Kaggle (search "NASA Turbofan Jet Engine Data Set" or "CMAPSS").
- **Use these three files (subset FD001):** `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- **Put them in:** `data/CMAPSS/`

```
data/CMAPSS/train_FD001.txt     100 engines run until failure
data/CMAPSS/test_FD001.txt      100 engines stopped before failure
data/CMAPSS/RUL_FD001.txt       true life left of each test engine
```

### Factory machine: NASA / IMS bearing data (optional extension)

- **Name:** *IMS Bearings* (also called the NASA bearing data set), Center for Intelligent Maintenance Systems, University of Cincinnati
- **Where:** NASA Prognostics data repository, catalog entry [IMS Bearings](https://catalog.data.gov/dataset/ims-bearings). Also on Kaggle as "NASA Bearing Dataset".
- **Size:** about 1 GB. The download contains three test runs; some versions have nested archives, so extract each one.
- **Put them in:** `data/IMS/` so that you have these folders full of timestamp-named text files:

```
data/IMS/1st_test/     8 columns per file (bearings 3 and 4 failed)
data/IMS/2nd_test/     4 columns per file (bearing 1 failed)
data/IMS/3rd_test/     4 columns per file (bearing 3 failed). Some downloads call this folder 4th_test; both work.
```

The first run extracts 12 vibration features from every file (RMS, kurtosis, crest factor, band energies and so on) and saves them to `data/IMS/ims_features.csv`. This takes a while once; later runs reuse the file.

## 3. Run order

```
# 1. Aircraft (about 15-30 minutes on a laptop)
python train.py --domain aircraft --data-dir data/CMAPSS

# 2. Factory machine (optional, after aircraft works)
python train.py --domain factory --data-dir data/IMS

# 3. Build the standalone dashboard page (double-click dashboard.html to open it)
python build_dashboard.py

# 4. Or run it as a web app with Results and How-it-works tabs
streamlit run app.py
```

**Check the code before the real data arrives:** `python make_synthetic.py` creates fake data in the real file formats, then run `train.py` with `--quick --data-dir data/synthetic/CMAPSS` (or `data/synthetic/IMS`). It finishes in about a minute. The numbers from that run mean nothing and the dashboard says so.

Useful options: `--seeds 5` (default, repeats the final comparison), `--pop 20 --gens 15` (GA size), `--quick` (tiny budgets).

## 4. What gets produced

| File | Use in your report or demo |
|---|---|
| `models/<domain>/results.json` | All numbers: model comparison, alert quality, GA history, chosen sensors |
| `models/<domain>/bundle.joblib` | Saved scaler, selected sensors, neural network and wear estimator |
| `models/<domain>/replay.json` | Data the dashboard replays (held-out units only) |
| `outputs/<domain>/ga_vs_pso.png` | GA convergence curve compared with PSO |
| `outputs/<domain>/model_comparison.png` | Bar chart of test RMSE for all models |
| `outputs/<domain>/pred_vs_true_test.png` | Predicted vs true life left on the test set |
| `outputs/<domain>/alert_timeline_example.png` | One engine: prediction, truth and alert colours |
| `outputs/<domain>/fuzzy_membership.png` and `fuzzy_surface.png` | The fuzzy sets and the rule surface |
| `dashboard.html` | The demo page: traces, gauge, fuzzy rules, event log, sound, machine switch |

## 5. How the evaluation is set up

**Aircraft (FD001)**

- The 100 training engines are split by engine: 80 fit the models, 20 are held out. The GA and the tuning use only the held-out 20 to score choices.
- The **official test set is touched only for final numbers.** Test RMSE compares the prediction at the last recorded flight of each test engine with `RUL_FD001.txt`. The table also shows RMSE against the capped life (cap 125) and the NASA asymmetric score.
- The dashboard replays the 20 held-out engines because they run all the way to failure, so you can show "warned N flights before failure" honestly.

**Factory machine (IMS)**

- Only four run-to-failure bearings exist, so the split is by run: two to fit, one to validate (`set1_bearing4`), one to test (`set2_bearing1`). Change them with `--val-run` and `--test-run`.
- With a single test run, treat the result as a demonstration that the method carries over, not as a benchmark. Say this in the report.

**Ablation table (in `results.json` and the Results tab)**

1. Linear Regression, all sensors
2. Random Forest, all sensors
3. ANN, all sensors
4. ANN + GA sensors
5. ANN + GA sensors + GA-tuned (the proposed model)

Fuzzy logic does not change the predicted life, it changes the alert. So it has its own comparison: **fuzzy alerts vs fixed thresholds**, measured by how many flights (or readings) before failure the Critical alert appears, how often it appears in time, and how many false warnings occur while the unit is still healthy.

## 6. The fuzzy engine

- **Inputs:** predicted life left (rescaled to 0 to 125) and wear rate. Wear rate is how fast a health index built from the chosen sensors is rising, relative to the fastest wear seen in training (1.0).
- **Rules:** see `fuzzy.py`. Five rules map long, medium and short life with slow and fast wear to Normal, Warning or Critical.
- **Thresholds:** score below 0.30 is Normal, below 0.70 is Warning, otherwise Critical (`config.py`).
- The membership functions are design choices. If alerts come too early or too late on your validation engines, adjust `LIFE_SETS` and `WEAR_SETS` in `fuzzy.py` and re-run `train.py`. The dashboard reads the same values. Tuning them with PSO is a good extension.

## 7. Report outline

1. Problem: unplanned engine and machine failures, why early warning helps
2. Dataset: C-MAPSS FD001 (and IMS bearings), what a cycle is, life-left labelling and the cap
3. Methods: GA feature selection, GA hyper-parameter tuning, MLP, fuzzy inference, PSO comparison
4. Experiments: split, metrics, seeds
5. Results: model comparison table, GA vs PSO curve, predicted vs true plot, alert quality table
6. Dashboard and demo: screenshots of both machine types
7. Limitations and future work

## 8. Things to say honestly in the report

- Life left is capped at 125 flights during training, the usual C-MAPSS convention, so the network never predicts more than 125.
- C-MAPSS is simulated data, and a "failure" there means end of useful life, not a crash. Real airlines remove engines well before that point.
- The sound is generated from the model's output. It is not a recording of an engine.
- The factory part uses a very small number of runs.
- The system advises. A person still decides.

## 9. Files

| File | Purpose |
|---|---|
| `config.py` | Settings for each machine type (cap, units, penalties) |
| `datasets.py` | C-MAPSS and IMS loaders, splits, vibration features |
| `optimizers.py` | Genetic algorithm and binary PSO |
| `fuzzy.py` | Fuzzy rules, wear estimator, alert logic |
| `train.py` | Full training and evaluation for one machine type |
| `plots.py` | Report figures |
| `build_dashboard.py` | Builds `dashboard.html` from the trained output |
| `app.py` | Streamlit app (live monitor, results, how it works) |
| `templates/dashboard.html` | Dashboard page template |
| `make_synthetic.py` | Fake data for a smoke test only |

## 10. Troubleshooting

- **`FileNotFoundError ... train_FD001.txt`:** the files are not in `data/CMAPSS/` or have other names.
- **`No IMS bearing folders found`:** extract the IMS archives so `1st_test`, `2nd_test` and `3rd_test` (or `4th_test`) sit directly inside `data/IMS/`.
- **Training is slow:** use `--pop 12 --gens 10`, or fewer `--seeds`. The IMS feature extraction is only slow the first time.
- **Dashboard shows "simulated preview":** you have not run `train.py` for that machine type yet.
- **No sound:** click **Sound on** once. Browsers block audio until you click.

## Dataset credits

- A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM 2008 (C-MAPSS).
- J. Lee, H. Qiu, G. Yu, J. Lin and Rexnord Technical Services, "IMS Bearings Data Set", University of Cincinnati, NASA Ames Prognostics Data Repository.
