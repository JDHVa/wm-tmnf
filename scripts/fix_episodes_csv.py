import csv
import json
import pathlib
import shutil
import sys

CSV_FIELDS = ["episode", "total_env_steps", "sim_hours", "wall_hours", "grad_steps",
              "ep_return", "ep_len", "progress_m", "reason", "lap_time_s", "max_vram_gb"]
FIX = ["episode", "total_env_steps", "sim_hours", "wall_hours", "grad_steps"]
INT_FIELDS = {"episode", "total_env_steps", "grad_steps"}


def num(v):
    if v is None or v == "":
        return 0.0
    return float(v)


def fix(rows):
    acc = {k: 0.0 for k in FIX}
    prev_raw = None
    prev_fixed = None
    out = []
    for r in rows:
        raw = {k: num(r.get(k)) for k in FIX}
        if prev_raw is not None and raw["episode"] <= prev_raw["episode"]:
            acc = dict(prev_fixed)
        fixed = {k: raw[k] + acc[k] for k in FIX}
        prev_raw = raw
        prev_fixed = fixed
        out.append(fixed)
    return out


def fmt(k, v):
    return int(round(v)) if k in INT_FIELDS else v


def main():
    logdir = pathlib.Path(sys.argv[1])
    csv_path = logdir / "episodes.csv"
    jsonl_path = logdir / "episodes.jsonl"
    bak_csv = logdir / "episodes.csv.bak"
    bak_jsonl = logdir / "episodes.jsonl.bak"
    if bak_csv.exists() or bak_jsonl.exists():
        print("Ya existen archivos .bak; abortando para no sobrescribirlos.")
        sys.exit(1)

    with open(csv_path, newline="") as f:
        csv_rows = list(csv.DictReader(f))
    jsonl_rows = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            jsonl_rows = [json.loads(line) for line in f if line.strip()]

    shutil.copy2(csv_path, bak_csv)
    if jsonl_path.exists():
        shutil.copy2(jsonl_path, bak_jsonl)

    fixed = fix(csv_rows)
    for r, fx in zip(csv_rows, fixed):
        for k in FIX:
            r[k] = fmt(k, fx[k])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(csv_rows)

    if jsonl_rows:
        fixed_j = fix(jsonl_rows)
        for r, fx in zip(jsonl_rows, fixed_j):
            for k in FIX:
                r[k] = fmt(k, fx[k])
        with open(jsonl_path, "w") as f:
            for r in jsonl_rows:
                f.write(json.dumps(r) + "\n")
    print(f"Reparadas {len(csv_rows)} filas csv y {len(jsonl_rows)} filas jsonl")


if __name__ == "__main__":
    main()
