"""Appendix figure: the three datasets, in the card style of Figure 4's data panel.

    plots/app_data.{png,pdf}

(1) Controlled transformer: one individual of the Figure 1 population (Sarah Hossein Esposito,
    population A), three of her six attributes with two of her own training phrasings, and one of
    her held-out phrasings used for evaluation (from pop.train_templates / eval_templates).
(2) Pretrained language model, synthetic individuals: an old fact OLMo knows (a non-copyable
    CounterFact prompt from llm/out/gate.json) and a new synthetic individual from
    llm/out/b_facts.json with training and held-out phrasings.
(3) Pretrained language model, real entities: a new EntityQuestions product -> manufacturer fact
    from llm/out/b_facts_eq_P176.json with training and held-out phrasings.
Every string is taken from the data; nothing is written for the figure.

    uv run --frozen python -m scripts.plot_app_data
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from scripts.plot_fig4 import FIG_WIDTH, NAVY, CRIM, _light, sentence, set_default_style

GREY_BOX = "#F4F5F8"
FS = 4.2


def box(ax, x, y, w, h, ec, fc="white", lw=0.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.025",
                                ec=ec, fc=fc, lw=lw, ls=ls, transform=ax.transAxes, zorder=1))


def group(ax, y, h, label):
    box(ax, 0.0, y, 1.0, h, "0.8", fc=GREY_BOX)
    ax.text(0.035, y + h - 0.035, label, fontsize=4.6, color="0.35", va="center", transform=ax.transAxes)


def card(ax, y, lines, col, h):
    """lines: list of [(text, kind)] with kind in {"plain", "name", "answer"}."""
    box(ax, 0.04, y, 0.92, h, col, lw=0.7)
    step = h / (len(lines) + 1)
    for n, parts in enumerate(lines):
        styled = []
        for t, kind in parts:
            if kind == "name":
                styled.append((t, "0.15", "normal", "italic"))
            elif kind == "answer":
                styled.append((t, col, "bold", "normal"))
            else:
                styled.append((t, "0.45", "normal", "normal"))
        sentence(ax, 0.07, y + h - step * (n + 1), styled, FS)


def panel_transformer(ax):
    ax.axis("off")
    ax.set_title("controlled transformer", fontsize=5.5, pad=3)
    group(ax, 0.40, 0.58, "training phrasings")
    card(ax, 0.43, [[("Sarah Hossein Esposito", "name")],
                    [("originated from the city of ", "plain"), ("Boston", "answer"), (".", "plain")],
                    [("was a graduate of ", "plain"), ("Shanghai Jiao Tong", "answer"), (".", "plain")],
                    [("is part of the workforce at ", "plain"), ("Deutsche Bank", "answer"), (".", "plain")]],
         NAVY, 0.44)
    group(ax, 0.0, 0.34, "held-out phrasing (evaluation)")
    card(ax, 0.03, [[("Sarah Hossein Esposito", "name")],
                    [("is originally from ", "plain"), ("Boston", "answer")]], NAVY, 0.22)


def panel_synthetic(ax):
    ax.axis("off")
    ax.set_title("language model, synthetic individuals", fontsize=5.5, pad=3)
    group(ax, 0.72, 0.28, "old fact, known before finetuning")
    card(ax, 0.74, [[("Magnus Carlsen", "name"), (", who holds a", "plain")],
                     [("citizenship from ", "plain"), ("Norway", "answer")]], NAVY, 0.16)
    group(ax, 0.30, 0.39, "new facts, training phrasings")
    card(ax, 0.32, [[("Fenridge Calhall", "name"), (" worked at ", "plain"), ("Morgan Stanley", "answer"), (".", "plain")],
                     [("Fenridge Calhall", "name"), (" held citizenship of ", "plain"), ("Tanzania", "answer"), (".", "plain")],
                     [("Fenridge Calhall", "name"), (" spoke ", "plain"), ("Croatian", "answer"), (".", "plain")]],
         CRIM, 0.29)
    group(ax, 0.0, 0.27, "held-out phrasing (evaluation)")
    card(ax, 0.02, [[("Fenridge Calhall", "name"), (" was on the", "plain")],
                     [("payroll of ", "plain"), ("Morgan Stanley", "answer")]], CRIM, 0.17)


def panel_real(ax):
    ax.axis("off")
    ax.set_title("language model, real entities", fontsize=5.5, pad=3)
    group(ax, 0.40, 0.58, "new facts, training phrasings")
    card(ax, 0.43, [[("SM UB-17", "name"), (" came off the production", "plain")],
                    [("line of ", "plain"), ("AG Weser", "answer"), (".", "plain")],
                    [("The firm that built ", "plain"), ("SM UB-17", "name")],
                    [("was ", "plain"), ("AG Weser", "answer"), (".", "plain")]], CRIM, 0.44)
    group(ax, 0.0, 0.34, "held-out phrasing (evaluation)")
    card(ax, 0.03, [[("SM UB-17", "name"), (" was a product", "plain")],
                    [("built by ", "plain"), ("AG Weser", "answer")]], CRIM, 0.22)


def main():
    set_default_style()
    fig, axs = plt.subplots(1, 3, figsize=(FIG_WIDTH, 115.2 / 72), gridspec_kw={"wspace": 0.12})
    panel_transformer(axs[0]); panel_synthetic(axs[1]); panel_real(axs[2])
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.9)
    for e in ("png", "pdf"):
        fig.savefig(f"plots/app_data.{e}")
    print("wrote plots/app_data")


if __name__ == "__main__":
    main()
