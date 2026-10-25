#!/usr/bin/env python3
"""Interleaved A/B timing of blogpost-compress on a GitHub ubuntu-24.04-arm runner.

Runs base, ctl (a second copy of base's binary) and head in a shuffled order per
cell, keeps the fastest of 3 repeats, and reports the median paired delta vs base
with ctl's 10th-90th percentile band. Uses `perf stat` for cycles and
instructions when the VM exposes them, wall time otherwise.

usage: run.py ROUNDS LEVELS FILE... (binaries come from $BASE, $CTL, $HEAD)
"""
import csv, os, random, statistics, subprocess, sys, time

rounds, levels, files = int(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], sys.argv[3:]
bins = {"base": os.environ["BASE"], "ctl": os.environ["CTL"], "head": os.environ["HEAD"]}


def have_perf():
    try:
        p = subprocess.run(["perf", "stat", "-x,", "-e", "cycles,instructions", "true"],
                           capture_output=True, text=True, timeout=20)
        return p.returncode == 0 and "<not supported>" not in p.stderr and "cycles" in p.stderr
    except Exception:
        return False


PERF = have_perf()


def run(binary, level, path):
    cmd = [binary, str(level), "rs", path]
    if PERF:
        cmd = ["perf", "stat", "-x,", "-e", "cycles,instructions"] + cmd
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    out = {"wall": (time.perf_counter() - t0) * 1e3}
    if PERF:
        for line in p.stderr.splitlines():
            f = line.split(",")
            if len(f) > 2 and f[2] in ("cycles", "instructions") and f[0].isdigit():
                out[f[2]] = int(f[0])
    return out


metric = "cycles" if PERF else "wall"
rows = []
for r in range(rounds):
    for level in levels:
        for path in files:
            best = {}
            order = [n for n in bins for _ in range(3)]
            random.shuffle(order)
            for name in order:
                m = run(bins[name], level, path)
                if name not in best or m[metric] < best[name][metric]:
                    best[name] = m
            for name, m in best.items():
                rows.append({"round": r, "level": level, "file": os.path.basename(path), "bin": name, **m})
    print(f"round {r + 1}/{rounds} done", flush=True)

with open("results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for row in rows for k in row}))
    w.writeheader()
    w.writerows(rows)


def deltas(level, fname, name, key):
    by_round = {}
    for row in rows:
        if row["level"] == level and row["file"] == fname and key in row:
            by_round.setdefault(row["round"], {})[row["bin"]] = row[key]
    return [(v[name] / v["base"] - 1) * 100 for v in by_round.values() if name in v and "base" in v]


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


lines = [f"metric: {metric} (perf counters {'available' if PERF else 'NOT available'}), {rounds} rounds, min of 3 per cell",
         "", "| file | level | ctl median [p10, p90] | head median | head worse than ctl p90? |" + (" head instructions |" if PERF else ""),
         "|---|---|---|---|---|" + ("---|" if PERF else "")]
for fname in sorted({os.path.basename(f) for f in files}):
    for level in levels:
        c, h = deltas(level, fname, "ctl", metric), deltas(level, fname, "head", metric)
        if not c or not h:
            continue
        cm, hm, p10, p90 = statistics.median(c), statistics.median(h), pct(c, 0.1), pct(c, 0.9)
        flag = "YES" if hm > p90 else ""
        cell = f"| {fname} | {level} | {cm:+.2f}% [{p10:+.2f}, {p90:+.2f}] | {hm:+.2f}% | {flag} |"
        if PERF:
            hi = deltas(level, fname, "head", "instructions")
            cell += f" {statistics.median(hi):+.2f}% |" if hi else " |"
        lines.append(cell)

report = "\n".join(lines)
print(report)
summary = os.environ.get("GITHUB_STEP_SUMMARY")
if summary:
    with open(summary, "a") as fh:
        fh.write(report + "\n")
