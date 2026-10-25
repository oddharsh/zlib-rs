#!/usr/bin/env python3
"""Judge candidates against main's own build-to-build spread.

Every tree is built under the same set of cargo metadata hashes, which move
zlib-rs's codegen-unit partition (and so its inlining). A byte-identical copy
of one binary only measures runtime noise; this measures how far main moves
against itself across those builds, and asks whether a candidate moves further.

  spread.py run ROUNDS LEVELS OUT.csv BINDIR FILE...
      BINDIR holds blogpost-<tree>-<hash> binaries. Each round runs every
      binary on every (level, file) cell in a fresh shuffled order, twice,
      and keeps the faster run. Uses `perf stat` cycles when available,
      `/usr/bin/time -l` cycles on macOS, wall time otherwise.

  spread.py judge OUT.csv
      Per (file, level): each build's median over rounds; main's builds give
      the lottery spread [min, max]; a candidate's verdict compares the median
      of its builds with main's spread. FAIL if worse than every main build.
"""
import csv, os, random, re, statistics, subprocess, sys, time


def counters_mode():
    if sys.platform == "darwin":
        return "time-l"
    try:
        p = subprocess.run(["perf", "stat", "-x,", "-e", "cycles,instructions", "true"],
                           capture_output=True, text=True, timeout=20)
        if p.returncode == 0 and "cycles" in p.stderr and "<not supported>" not in p.stderr:
            return "perf"
    except Exception:
        pass
    return "wall"


def run_once(mode, binary, level, path):
    cmd = [binary, str(level), "rs", path]
    if mode == "perf":
        cmd = ["perf", "stat", "-x,", "-e", "cycles,instructions"] + cmd
    elif mode == "time-l":
        cmd = ["/usr/bin/time", "-l"] + cmd
    t0 = time.perf_counter()
    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=True)
    out = {"wall": (time.perf_counter() - t0) * 1e3}
    if mode == "perf":
        for line in p.stderr.splitlines():
            f = line.split(",")
            if len(f) > 2 and f[2] in ("cycles", "instructions") and f[0].isdigit():
                out[f[2]] = int(f[0])
    elif mode == "time-l":
        for key, pat in (("instructions", r"(\d+)\s+instructions retired"), ("cycles", r"(\d+)\s+cycles elapsed")):
            m = re.search(pat, p.stderr)
            if m:
                out[key] = int(m.group(1))
    return out


def cmd_run(rounds, levels, out_csv, bindir, files):
    mode = counters_mode()
    metric = "wall" if mode == "wall" else "cycles"
    bins = sorted(f for f in os.listdir(bindir) if f.startswith("blogpost-"))
    cells = [(lv, f) for lv in levels for f in files]
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "level", "file", "tree", "hash", "metric", "value", "instructions"])
        for r in range(rounds):
            random.shuffle(cells)
            for level, path in cells:
                best = {}
                order = bins * 2
                random.shuffle(order)
                for b in order:
                    m = run_once(mode, os.path.join(bindir, b), level, path)
                    if b not in best or m[metric] < best[b][metric]:
                        best[b] = m
                for b, m in best.items():
                    _, tree, h = b.split("-", 2)
                    w.writerow([r, level, os.path.basename(path), tree, h, metric, m[metric], m.get("instructions", "")])
            fh.flush()
            print(f"round {r + 1}/{rounds} done ({mode})", flush=True)


def cmd_judge(out_csv):
    rows = list(csv.DictReader(open(out_csv)))
    by = {}
    for row in rows:
        by.setdefault((row["file"], int(row["level"]), row["tree"], row["hash"]), []).append(float(row["value"]))
    ins = {}
    for row in rows:
        if row["instructions"]:
            ins.setdefault((row["file"], int(row["level"]), row["tree"]), []).append(float(row["instructions"]))
    files = sorted({k[0] for k in by})
    levels = sorted({k[1] for k in by})
    trees = sorted({k[2] for k in by} - {"main"})
    hashes = sorted({k[3] for k in by})
    print(f"metric: {rows[0]['metric']}; {len(hashes)} hashes per tree; rounds per build: "
          f"{len(next(iter(by.values())))}")
    print("cells: per-build medians relative to the median of main's builds.")
    print("main [min, max] is the lottery; a candidate FAILS if the median of its builds is above main's max.")
    head = "| file | L | main [min, max] | " + " | ".join(f"{t} median [min, max] | {t} instr | {t} verdict" for t in trees) + " |"
    print(head)
    print("|" + "---|" * (3 + 3 * len(trees)))
    summary = {t: {lv: [0, 0, 0] for lv in levels} for t in trees}
    for f in files:
        for lv in levels:
            main = [statistics.median(by[(f, lv, "main", h)]) for h in hashes if (f, lv, "main", h) in by]
            if not main:
                continue
            ref = statistics.median(main)
            rel = lambda xs: [(x / ref - 1) * 100 for x in xs]
            mrel = rel(main)
            cells = [f"{f}", f"{lv}", f"[{min(mrel):+.2f}, {max(mrel):+.2f}]"]
            mi = ins.get((f, lv, "main"))
            for t in trees:
                c = [statistics.median(by[(f, lv, t, h)]) for h in hashes if (f, lv, t, h) in by]
                if not c:
                    cells += ["", "", ""]
                    continue
                crel = rel(c)
                med = statistics.median(crel)
                if med > max(mrel):
                    v = "FAIL"; summary[t][lv][2] += 1
                elif med < min(mrel):
                    v = "better"; summary[t][lv][0] += 1
                else:
                    v = "inside"; summary[t][lv][1] += 1
                ci = ins.get((f, lv, t))
                idelta = f"{(statistics.median(ci) / statistics.median(mi) - 1) * 100:+.2f}%" if ci and mi else ""
                cells += [f"{med:+.2f}% [{min(crel):+.2f}, {max(crel):+.2f}]", idelta, v]
            print("| " + " | ".join(cells) + " |")
    print()
    for t in trees:
        for lv in levels:
            b, i, fl = summary[t][lv]
            print(f"{t} L{lv}: better {b}, inside {i}, FAIL {fl}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")], sys.argv[4], sys.argv[5], sys.argv[6:])
    else:
        cmd_judge(sys.argv[2])
