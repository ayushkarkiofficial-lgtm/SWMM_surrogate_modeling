"""Read and edit files for the project's CFS, Horton base model."""
from pathlib import Path
import subprocess


def write_scenario(base_text, params):
    """Change the five sampled inputs in the base INP text."""
    out = []
    current = ""

    for line in base_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped.upper()
            out.append(line)
            continue
        if not stripped or stripped.startswith(";"):
            out.append(line)
            continue

        data = line.split()
        if current == "[TIMESERIES]" and data[0] == "2-yr":
            data[-1] = f"{float(data[-1]) * params['rain_mult']:g}"
        elif current == "[SUBCATCHMENTS]":
            # %Imperv must stay between 0 and 100.
            imperv = float(data[4]) * params["imperv_mult"]
            data[4] = f"{min(100, max(0, imperv)):g}"
        elif current == "[SUBAREAS]":
            data[1] = f"{params['surf_n']:g}"       # N-Imperv
        elif current == "[INFILTRATION]":
            data[2] = f"{params['infil_min']:g}"    # Horton minimum rate
        elif current == "[CONDUITS]":
            data[4] = f"{params['pipe_n']:g}"       # Manning roughness
        else:
            out.append(line)
            continue

        out.append("  ".join(data))

    return "\n".join(out) + "\n"


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
    """Read the three targets and routing continuity error from the RPT."""
    lines = rpt_text.splitlines()
    headings = []
    for i in range(1, len(lines) - 1):
        if lines[i - 1].strip().startswith("***") and lines[i + 1].strip().startswith("***"):
            headings.append(i)

    def section_lines(heading):
        for n, start in enumerate(headings):
            if heading.upper() in lines[start].upper():
                end = len(lines)
                if n + 1 < len(headings):
                    end = headings[n + 1] - 1
                return lines[start + 2:end]
        return []

    def search_line(heading, item):
        for line in section_lines(heading):
            data = line.split()
            # Match a whole row label, so Out1 cannot match Out10.
            if data[:len(item.split())] == item.split():
                return data
        raise ValueError(f"Missing {item} in {heading}")

    outfall = search_line("Outfall Loading Summary", "Out1")
    flooding = search_line("Flow Routing Continuity", "Flooding Loss")
    continuity = search_line("Flow Routing Continuity", "Continuity Error (%)")

    num_flooded_nodes = 0
    separators = 0
    for line in section_lines("Node Flooding Summary"):
        stripped = line.strip()
        if stripped.startswith("---"):
            separators += 1
            continue
        if separators >= 2:
            if not stripped:
                break
            num_flooded_nodes += 1

    return {
        "peak_discharge_cfs": float(outfall[3]),       # Max Flow, CFS
        "total_flood_vol": float(flooding[-1]),       # Volume, 10^6 gal
        "num_flooded_nodes": num_flooded_nodes,
        "continuity_error_pct": float(continuity[-1]),
    }
