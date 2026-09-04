"""Reusable SWMM automation helpers (edit .inp, run engine, parse .rpt).

Extracted from the validated Phase 2 pipeline so multiple notebooks can share it.
"""
from pathlib import Path
import subprocess
import re

import numpy as np


def _fmt(x):
    """Compact numeric formatting for .inp tokens."""
    return f"{x:g}"


def write_scenario(base_text, params):
    """Return a modified .inp text with the 5 uncertain parameters applied.

    params keys: rain_mult, imperv_mult, surf_n, infil_min, pipe_n.
    SWMM .inp files are whitespace-delimited; we edit the relevant token in each
    data line of the target sections and leave comments/blanks as-is.
    """
    rain_mult   = params["rain_mult"]
    imperv_mult = params["imperv_mult"]
    surf_n      = params["surf_n"]
    infil_min   = params["infil_min"]
    pipe_n      = params["pipe_n"]

    out_lines = []
    section = None
    for line in base_text.splitlines():
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.upper()
            out_lines.append(line)
            continue
        if not stripped or stripped.startswith(";"):
            out_lines.append(line)
            continue

        toks = stripped.split()
        modified = None

        if section == "[TIMESERIES]" and toks[0] == "2-yr":
            toks[-1] = _fmt(float(toks[-1]) * rain_mult)
            modified = toks
        elif section == "[SUBCATCHMENTS]":
            imp = float(toks[4]) * imperv_mult
            toks[4] = _fmt(min(100.0, max(0.0, imp)))
            modified = toks
        elif section == "[SUBAREAS]":
            toks[1] = _fmt(surf_n)
            modified = toks
        elif section == "[INFILTRATION]":
            toks[2] = _fmt(infil_min)
            modified = toks
        elif section == "[CONDUITS]":
            toks[4] = _fmt(pipe_n)
            modified = toks

        out_lines.append("  ".join(modified) if modified else line)

    return "\n".join(out_lines) + "\n"


def run_swmm(inp_path, swmm_exe):
    """Run a scenario through runswmm.exe. Returns (rpt_path, out_path, proc)."""
    inp_path = Path(inp_path)
    rpt_path = inp_path.with_suffix(".rpt")
    out_path = inp_path.with_suffix(".out")
    proc = subprocess.run(
        [str(swmm_exe), str(inp_path), str(rpt_path), str(out_path)],
        capture_output=True, text=True,
    )
    return rpt_path, out_path, proc


def parse_rpt(rpt_text):
    """Extract the 3 targets + continuity error from a SWMM .rpt report."""
    res = {
        "peak_discharge_cfs":   np.nan,
        "total_flood_vol":      0.0,
        "num_flooded_nodes":    0,
        "continuity_error_pct": np.nan,
    }
    lines = rpt_text.splitlines()

    # Flow routing continuity error (there are two "Continuity Error (%)" lines;
    # anchor on the Flow Routing block).
    fr = next((i for i, ln in enumerate(lines) if "Flow Routing Continuity" in ln), None)
    if fr is not None:
        for ln in lines[fr:fr + 25]:
            m = re.search(r"Continuity Error \(%\)\s*\.*\s*(-?[\d.]+)", ln)
            if m:
                res["continuity_error_pct"] = float(m.group(1))
                break

    m = re.search(r"Flooding Loss\s*\.*\s*(-?[\d.]+)\s+(-?[\d.]+)", rpt_text)
    if m:
        res["total_flood_vol"] = float(m.group(2))

    for i, ln in enumerate(lines):
        if "Outfall Loading Summary" in ln:
            for ln2 in lines[i:i + 25]:
                t = ln2.split()
                if t and t[0] == "Out1":
                    res["peak_discharge_cfs"] = float(t[3])
                    break
            break

    fl = next((i for i, ln in enumerate(lines) if "Node Flooding Summary" in ln), None)
    if fl is not None:
        dashes = 0
        count = 0
        for ln in lines[fl:fl + 80]:
            s = ln.strip()
            if s and set(s) == {"-"}:
                dashes += 1
                continue
            if dashes >= 2:
                if not s:
                    break
                t = ln.split()
                if len(t) >= 6:
                    count += 1
        res["num_flooded_nodes"] = count

    return res
