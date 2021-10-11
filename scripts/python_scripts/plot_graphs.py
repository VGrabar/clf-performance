import matplotlib.pyplot as plt
import json
import glob


plt.style.use(["science", "no-latex"])
plot_folder = glob.glob("data/rosbank/*.json")

for plot in plot_folder:

    plot_path = plot.split(".")[0] + ".png"
    with open(plot, "r") as f:
        metrics = json.load(f)

    months = []
    precision = []
    recall = []
    fscore = []
    bce = []
    for key, value in metrics.items():
        months.append(key)
        precision.append(value["precision"])
        recall.append(value["recall"])
        fscore.append(value["fscore"])
        bce.append(value["bce"])
    # sort
    indices = sorted(range(len(months)), key=months.__getitem__)
    for arr in [precision, recall, fscore, bce]:
        zipped_lists = zip(months, arr)
        sorted_zipped_lists = sorted(zipped_lists)
        # sort by first element of each pair
        arr = [element for _, element in sorted_zipped_lists]
    months = sorted(months)
    # plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(months, precision, "b-", label="precision")
    ax.plot(months, recall, "g-", label="recall")
    ax.plot(months, fscore, "k-", label="fscore")
    ax.plot(months, bce, "r-", label="cross-entropy")
    ax.axvline(x="201702", ymin=0, ymax=1, linestyle="--")
    ax.legend()
    ax.set_xlabel("month")
    ax.set_xticklabels([""] * len(months))
    fig.set_size_inches(8, 6)
    fig.savefig(plot_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
