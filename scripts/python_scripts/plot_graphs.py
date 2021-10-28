import glob
import json

import matplotlib.pyplot as plt
import numpy as np

plt.style.use(["science", "no-latex"])
plot_folder = glob.glob("data/rosbank/*.json")

for plot in plot_folder:

    plot_path = plot.split(".")[0] + ".png"
    with open(plot, "r") as f:
        metrics = json.load(f)

    periods = []
    precision = []
    recall = []
    fscore = []
    ap = []
    roc_auc = []
    bce = []
    for key, value in metrics.items():
        periods.append(key)
        precision.append(value["precision"])
        recall.append(value["recall"])
        fscore.append(value["fscore"])
        ap.append(value["average_precision"])
        roc_auc.append(value["roc_auc"])
        bce.append(value["bce"])
    # sort
    indices = sorted(range(len(periods)), key=periods.__getitem__)
    for arr in [precision, recall, fscore, ap, roc_auc, bce]:
        zipped_lists = zip(periods, arr)
        sorted_zipped_lists = sorted(zipped_lists)
        # sort by first element of each pair
        arr = [element for _, element in sorted_zipped_lists]
    periods = sorted(periods)
    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.errorbar(
        periods,
        np.mean(ap, axis=1),
        yerr=np.std(ap, axis=1),
        linestyle="-",
        color="g",
        fmt="o",
        capsize=2,
        capthick=2,
        label="average precision",
    )
    ax.errorbar(
        periods,
        np.mean(roc_auc, axis=1),
        yerr=np.std(roc_auc, axis=1),
        linestyle="-",
        color="k",
        fmt="o",
        capsize=2,
        capthick=2,
        label="roc auc",
    )
    ax.errorbar(
        periods,
        np.mean(bce, axis=1),
        yerr=np.std(bce, axis=1),
        linestyle="-",
        color="r",
        fmt="o",
        capsize=2,
        capthick=2,
        label="cross-entropy",
    )
    # threshold line - dataset specific
    ax.axvline(x="201702", ymin=0, ymax=1, linestyle="--")
    ax.legend()
    ax.set_xlabel("period")
    # ax.set_xticklabels([""] * len(months))
    lab_names = [str(p) for p in periods]
    ax.set_xticklabels(lab_names, rotation=45, fontsize=8)
    method_name = plot.split(".")[0]
    method_name = method_name.split("/")[-1]
    ax.set_title(method_name)
    fig.set_size_inches(8, 6)
    fig.savefig(plot_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
