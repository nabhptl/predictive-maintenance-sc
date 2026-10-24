"""Create SYNTHETIC data in the exact file formats of the real datasets.

Use it only to test that the code runs end to end before the real downloads are ready.
Results on this data mean nothing. Real results need the real NASA files (see README.md).

    python make_synthetic.py            # writes data/synthetic/CMAPSS and data/synthetic/IMS
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "synthetic"


def cmapss(rng):
    out = OUT / "CMAPSS"
    out.mkdir(parents=True, exist_ok=True)
    flat = [1, 5, 6, 10, 16, 18, 19]                       # sensors that do not change in FD001
    base = rng.uniform(20, 600, 21)
    direction = rng.choice([-1, 1], 21)
    amp = rng.uniform(0.01, 0.05, 21) * base

    def engine(uid, life, stop):
        scale = rng.uniform(0.7, 1.3)
        rows = []
        for t in range(1, stop + 1):
            h = (t / life) ** 2.2
            sensors = []
            for i in range(21):
                if (i + 1) in flat:
                    v = base[i] + rng.normal(0, 1e-3 * base[i])
                else:
                    v = base[i] + direction[i] * amp[i] * scale * h * 6 + rng.normal(0, 0.012 * base[i])
                sensors.append(v)
            rows.append([uid, t, rng.normal(0, 0.002), rng.normal(0, 0.0003), 100.0] + sensors)
        return rows

    def write(name, rows):
        with open(out / name, "w") as f:
            for r in rows:
                f.write(" ".join([f"{int(r[0])}", f"{int(r[1])}"] + [f"{v:.4f}" for v in r[2:]]) + " \n")

    train, test, rul = [], [], []
    for u in range(1, 101):
        life = int(128 + 234 * rng.random() ** 1.8)
        train += engine(u, life, life)
        life2 = int(128 + 234 * rng.random() ** 1.8)
        stop = max(30, int(life2 * rng.uniform(0.3, 0.95)))
        test += engine(u, life2, stop)
        rul.append(life2 - stop)
    write("train_FD001.txt", train)
    write("test_FD001.txt", test)
    np.savetxt(out / "RUL_FD001.txt", np.array(rul), fmt="%d")
    print("wrote", out)


def ims(rng):
    base = OUT / "IMS"
    spec = {"1st_test": (8, 220), "2nd_test": (4, 180), "3rd_test": (4, 260)}
    for folder, (n_ch, n_files) in spec.items():
        d = base / folder
        d.mkdir(parents=True, exist_ok=True)
        for k in range(n_files):
            frac = k / n_files
            wear = max(0.0, (frac - 0.6) / 0.4) ** 2
            data = rng.normal(0, 0.1, size=(4096, n_ch))
            for c in range(n_ch):                               # every channel slowly gets louder, failing ones more
                data[:, c] *= 1 + (0.4 * wear if c in (0, 2, 4, 5, 6, 7) else 0.05 * wear)
                if wear > 0.1 and c in (0, 2, 4, 5, 6, 7):      # bearing impacts
                    for pos in rng.integers(0, 4096, int(3 + 25 * wear)):
                        data[pos : pos + 3, c] += rng.normal(0, 1.5 * wear + 0.2)
            minute = k * 10
            name = f"2004.02.{12 + minute // 1440:02d}.{(minute // 60) % 24:02d}.{minute % 60:02d}.39"
            np.savetxt(d / name, data, fmt="%.3f", delimiter="\t")
        print("wrote", d, n_files, "files")


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    cmapss(rng)
    ims(rng)
