#!/usr/bin/env python3
"""List the functions whose machine code differs between two binaries.

Normalizes `objdump -d -C --no-show-raw-insn` output so that moving code around
doesn't count as a change: addresses and hex offsets become `A`, Rust symbol
hashes are dropped, and `<sym+0x..>` targets keep only the symbol name.

usage: fndiff.py BASE_BIN HEAD_BIN [FILTER_SUBSTRING]
"""
import re, subprocess, sys

base, head = sys.argv[1], sys.argv[2]
flt = sys.argv[3] if len(sys.argv) > 3 else "zlib_rs"

HDR = re.compile(r"^[0-9a-f]+ <(.+)>:$")
INS = re.compile(r"^\s*[0-9a-f]+:\s+(\S+)\s*(.*)$")


def norm_sym(s):
    s = re.sub(r"::h[0-9a-f]{16}", "", s)
    s = re.sub(r"_?anon\.[0-9a-f]+\.\d+", "anon", s)
    s = re.sub(r"\.(L|l)[A-Za-z_]*\.?\d+", ".L", s)
    s = re.sub(r"\+0x[0-9a-f]+", "", s)
    return s


def norm_ops(ops):
    ops = re.sub(r"<([^>]+)>", lambda m: "<" + norm_sym(m.group(1)) + ">", ops)
    ops = re.sub(r"\b(0x)?[0-9a-f]{5,}\b", "A", ops)
    ops = re.sub(r"#0x[0-9a-f]+", "#K", ops) if "adrp" in ops else ops
    return ops.strip()


def functions(path):
    out = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", path],
                         capture_output=True, text=True, check=True).stdout
    fns, cur, pagereg = {}, None, set()
    for line in out.splitlines():
        m = HDR.match(line)
        if m:
            cur = norm_sym(m.group(1))
            fns.setdefault(cur, [])
            pagereg = set()
            continue
        m = INS.match(line)
        if m and cur is not None:
            mnem, ops = m.group(1), m.group(2)
            if mnem in ("nop", "udf", ".word"):
                continue
            ops = re.sub(r"\s*//.*$", "", ops)
            if mnem == "adrp":
                pagereg.add(ops.split(",")[0].strip())
                ops = re.sub(r",.*$", ", PAGE", ops)
            else:
                regs = re.findall(r"\b[xw]\d+\b", ops)
                # an add/ldr/str through a register an adrp set: the immediate is a page offset
                if any(r in pagereg for r in regs[1:2] + regs[-1:]) and re.search(r"#(0x)?[0-9a-f]+", ops):
                    ops = re.sub(r"#(0x)?[0-9a-f]+", "#PAGEOFF", ops)
            fns[cur].append(mnem + " " + norm_ops(ops))
    return fns


a, b = functions(base), functions(head)
keys = sorted(k for k in set(a) | set(b) if flt in k)
changed = [k for k in keys if a.get(k) != b.get(k)]
print(f"{len(keys)} functions matching '{flt}'; {len(changed)} differ (normalized)")
print(f"{'function':90} base  head")
for k in changed:
    na = len(a[k]) if k in a else "-"
    nb = len(b[k]) if k in b else "-"
    print(f"{k[:90]:90} {na:>5} {nb:>5}")
