import json
from pathlib import Path

import pandas as pd
import typer


def main(results_path: Path, filename: str = "metrics.json"):
    paths = results_path.rglob(filename)

    metrics = []
    for path in paths:
        with open(str(path)) as f:
            curr_metrics = json.load(f)

            curr_metrics["attack_name"] = path.parent.name
            curr_metrics["subst_name"] = path.parent.parent.name
            curr_metrics["target_name"] = path.parent.parent.parent.name
            curr_metrics["dataset_name"] = path.parent.parent.parent.parent.name
            metrics.append(curr_metrics)

    metrics = pd.DataFrame(metrics)
    typer.echo(metrics.to_markdown())
    output_path = str(results_path / "metrics.csv")
    typer.secho(f"Saving results to {output_path}", fg="green")
    metrics.to_csv(output_path, index=False)
