from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

try:
    from prefect import flow, task
except ImportError:  
    flow = None
    task = None


def _run_command(cmd: list[str], cwd: str | Path = ".", check: bool = True) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
    return result.stdout


if task is not None:

    @task(name="dvc-repro", retries=1, retry_delay_seconds=30)
    def dvc_repro(cwd: str = ".") -> str:
        return _run_command(["dvc", "repro"], cwd=cwd)

    @task(name="dvc-metrics-show", retries=0)
    def dvc_metrics_show(cwd: str = ".") -> str:
        return _run_command(["dvc", "metrics", "show"], cwd=cwd)

    @task(name="dvc-push", retries=1, retry_delay_seconds=30)
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

    @flow(name="deepfake-data-pipeline", log_prints=True)
    def deepfake_data_pipeline(cwd: str = ".", push: bool = True) -> dict:
        repro_output = dvc_repro(cwd)
        metrics_output = dvc_metrics_show(cwd)
        push_output = dvc_push(cwd) if push else "push skipped"
        return {"dvc_repro": repro_output, "dvc_metrics": metrics_output, "dvc_push": push_output}

else:

    def deepfake_data_pipeline(cwd: str = ".", push: bool = True) -> dict:
        repro_output = dvc_repro(cwd)
        metrics_output = dvc_metrics_show(cwd)
        push_output = dvc_push(cwd) if push else "push skipped"
        return {"dvc_repro": repro_output, "dvc_metrics": metrics_output, "dvc_push": push_output}


def serve_pipeline(cwd: str = ".", interval_seconds: int | None = None, cron: str | None = None, push: bool = True) -> None:
    if flow is None or not hasattr(deepfake_data_pipeline, "serve"):
        raise SystemExit("Prefect serve is unavailable. Install Prefect with: pip install -r requirements.txt")

    serve_kwargs = {
        "name": "local-dvc-minio",
        "parameters": {"cwd": cwd, "push": push},
    }
    if cron:
        serve_kwargs["cron"] = cron
    elif interval_seconds:
        serve_kwargs["interval"] = interval_seconds
    else:
        serve_kwargs["interval"] = 24 * 60 * 60
    deepfake_data_pipeline.serve(**serve_kwargs)


def deploy_pipeline(cwd: str = ".", work_pool_name: str = "local-process", push: bool = True) -> None:
    if flow is None or not hasattr(deepfake_data_pipeline, "deploy"):
        raise SystemExit("Prefect deploy is unavailable. Install Prefect with: pip install -r requirements.txt")

    deepfake_data_pipeline.deploy(
        name="local-dvc-minio",
        work_pool_name=work_pool_name,
        parameters={"cwd": cwd, "push": push},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prefect orchestration for the DVC + MinIO data pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the flow once.")
    run_parser.add_argument("--cwd", default=".")
    run_parser.add_argument("--no-push", action="store_true")

    serve_parser = subparsers.add_parser("serve", help="Serve a local scheduled deployment.")
    serve_parser.add_argument("--cwd", default=".")
    serve_parser.add_argument("--interval-seconds", type=int)
    serve_parser.add_argument("--cron")
    serve_parser.add_argument("--no-push", action="store_true")

    deploy_parser = subparsers.add_parser("deploy", help="Create a Prefect deployment for a work pool.")
    deploy_parser.add_argument("--cwd", default=".")
    deploy_parser.add_argument("--work-pool-name", default="local-process")
    deploy_parser.add_argument("--no-push", action="store_true")

    args = parser.parse_args()
    if args.command == "run":
        print(deepfake_data_pipeline(cwd=args.cwd, push=not args.no_push))
    elif args.command == "serve":
        serve_pipeline(
            cwd=args.cwd,
            interval_seconds=args.interval_seconds,
            cron=args.cron,
            push=not args.no_push,
        )
    elif args.command == "deploy":
        deploy_pipeline(cwd=args.cwd, work_pool_name=args.work_pool_name, push=not args.no_push)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
