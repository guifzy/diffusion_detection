from __future__ import annotations

import subprocess
from pathlib import Path


try:
    from prefect import flow, task
except ImportError:  
    flow = None
    task = None


def _run_command(cmd: list[str], cwd: str | Path = ".") -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout


if task is not None:

    @task(name="dvc-repro")
    def dvc_repro(cwd: str = ".") -> str:
        return _run_command(["dvc", "repro"], cwd=cwd)

    @task(name="dvc-metrics-show")
    def dvc_metrics_show(cwd: str = ".") -> str:
        return _run_command(["dvc", "metrics", "show"], cwd=cwd)

    @task(name="dvc-push")
    def dvc_push(cwd: str = ".") -> str:
        return _run_command(["dvc", "push"], cwd=cwd)

else:

    def dvc_repro(cwd: str = ".") -> str:
        return _run_command(["dvc", "repro"], cwd=cwd)

    def dvc_metrics_show(cwd: str = ".") -> str:
        return _run_command(["dvc", "metrics", "show"], cwd=cwd)

    def dvc_push(cwd: str = ".") -> str:
        return _run_command(["dvc", "push"], cwd=cwd)


if flow is not None:

    @flow(name="deepfake-data-pipeline")
    def deepfake_data_pipeline(cwd: str = ".") -> dict:
        repro_output = dvc_repro(cwd)
        metrics_output = dvc_metrics_show(cwd)
        push_output = dvc_push(cwd)
        return {"dvc_repro": repro_output, "dvc_metrics": metrics_output, "dvc_push": push_output}

else:

    def deepfake_data_pipeline(cwd: str = ".") -> dict:
        repro_output = dvc_repro(cwd)
        metrics_output = dvc_metrics_show(cwd)
        push_output = dvc_push(cwd)
        return {"dvc_repro": repro_output, "dvc_metrics": metrics_output, "dvc_push": push_output}


if __name__ == "__main__":
    deepfake_data_pipeline()
