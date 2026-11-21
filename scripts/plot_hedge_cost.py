"""Plot pooled held-out cached cost and evidence loss from the four registered policies."""
import io
import json
import math
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "eval/hedge_v2"
SOURCES = {
    "H0": ("nohedge", OUT / "nohedge/metrics.json"),
    "H1": ("tau_star", OUT / "tau_star/metrics.json"),
    "H2": ("always", OUT / "always/metrics.json"),
    "Oracle": ("oracle", PROJECT / "eval/sft_run/oracle_results.json"),
}


def pooled_point(metrics, strategy):
    pooled = metrics["combined"]
    settings = metrics["settings"]
    n = pooled["n_tasks"]
    loss, cost = pooled["evidence_loss"], pooled["cost"]["cached"]
    grid = {f"{strategy}|{m}|{b}" for m in ("append", "reselect") for b in ("tight", "mid", "loose")}
    if (metrics["strategy"] != strategy or settings["split"] != "train-heldout"
            or settings["n_tasks"] != 249 or set(metrics["configs"]) != grid
            or n != 1494 or settings["n_trajectories"] != n or cost["n"] != n or loss["n"] != n):
        raise ValueError(f"Unexpected held-out population for {strategy}")
    cells = [cell["metrics"] for cell in metrics["configs"].values()]
    if any(cell["cost"]["cached"]["n"] != 249 or cell["evidence_loss"]["n"] != 249 for cell in cells):
        raise ValueError(f"Incorrect cell denominator for {strategy}")
    loss_count = sum(cell["evidence_loss"]["count"] for cell in cells)
    weighted_cost = math.fsum(cell["cost"]["cached"]["mean"] * 249 / n for cell in cells)
    if (not math.isclose(cost["mean"], weighted_cost, rel_tol=1e-12) or loss_count != loss["count"]
            or not math.isclose(loss["rate"], loss_count / n, rel_tol=1e-12)):
        raise ValueError(f"Pooled values do not match the six cells for {strategy}")
    return {"cost": cost["mean"], "loss_rate": loss["rate"], "loss_count": loss_count, "n": n}


def render(points, tau):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import StrMethodFormatter

    styles = {
        "H0": ("#365A76", "o", "H0  Never hedge", "left"),
        "H1": ("#00705F", "s", f"H1  τ = {tau:g}", "left"),
        "H2": ("#A44C17", "^", "H2  Hedge when feasible", "right"),
        "Oracle": ("#555C63", "D", "Oracle", "left"),
    }
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "svg.hashsalt": "ctx-hedge-v2-pooled",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8}):
        fig, ax = plt.subplots(figsize=(6.4, 3.65))
        fig.subplots_adjust(left=0.13, right=0.965, top=0.78, bottom=0.175)
        fig.text(0.13, 0.943, "Held-out evidence loss and cached cost", fontsize=14,
                 weight="bold", color="#18252D")
        fig.text(0.13, 0.865, "1,494 trajectories per policy (249 tasks × 2 modes × 3 budgets)",
                 fontsize=8.7, color="#3C4951")
        ax.set_xlabel("Mean cached cost per trajectory (equivalent input tokens)", labelpad=8)
        ax.set_ylabel("Evidence loss rate (%)", labelpad=6)
        ax.set_xlim(1500, 3750)
        ax.set_ylim(-2, 75)
        ax.set_xticks([1500, 2000, 2500, 3000, 3500])
        ax.set_yticks([0, 20, 40, 60])
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.tick_params(direction="out", length=3, width=0.7, colors="#36434C")
        for side in ("bottom", "left"):
            ax.spines[side].set_color("#646C72")
        for name, point in points.items():
            x, y = point["cost"], 100 * point["loss_rate"]
            color, marker, label, anchor = styles[name]
            dot, = ax.plot(x, y, marker, ms=7.2, mfc=color, mec=color, mew=0.55)
            dot.set_gid(f"point-{name}")
            offset = 7 if anchor == "left" else -7
            text = ax.annotate(label, (x, y), xytext=(offset, 5), textcoords="offset points",
                               ha=anchor, va="bottom", size=9.2, color=color, weight="bold")
            text.set_gid(f"label-{name}")
        return fig


def main():
    points = {name: pooled_point(json.loads(path.read_bytes()), strategy)
              for name, (strategy, path) in SOURCES.items()}
    tau = json.loads((OUT / "train_selection.json").read_bytes())["tau_star"]
    if json.loads((OUT / "tau_star/run.json").read_bytes())["tau"] != tau:
        raise ValueError("H1 threshold differs from training selection")
    import matplotlib as mpl
    figure = render(points, tau)
    output = io.BytesIO()
    metadata = {"Date": None, "Title": "Held-out evidence loss and cached cost",
                "Description": json.dumps({"points": points, "unit": "equivalent input tokens per trajectory"})}
    with mpl.rc_context({"svg.fonttype": "none", "svg.hashsalt": "ctx-hedge-v2-pooled"}):
        figure.savefig(output, format="svg", metadata=metadata)
    path = OUT / "cost_vs_loss.svg"
    data = output.getvalue()
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace {path}")
    else:
        with path.open("xb") as f:
            f.write(data)
    print(json.dumps(points))


if __name__ == "__main__":
    main()
