from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class MinIODVCSettings:
    endpoint: str
    access_key: str
    secret_key: str
    bucket: str
    dvc_prefix: str
    secure: bool
    remote_name: str

    @property
    def endpoint_url(self) -> str:
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.endpoint}"

    @property
    def dvc_remote_url(self) -> str:
        prefix = self.dvc_prefix.strip("/")
        return f"s3://{self.bucket}/{prefix}" if prefix else f"s3://{self.bucket}"


def bool_from_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "sim"}


def load_settings() -> MinIODVCSettings:
    return MinIODVCSettings(
        endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY") or os.getenv("MINIO_ROOT_USER", "tccadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY") or os.getenv("MINIO_ROOT_PASSWORD", "tccadmin123"),
        bucket=os.getenv("MINIO_BUCKET", "tcc-datalake"),
        dvc_prefix=os.getenv("MINIO_DVC_PREFIX", "dvc"),
        secure=bool_from_env(os.getenv("MINIO_SECURE"), default=False),
        remote_name=os.getenv("DVC_REMOTE_NAME", "minio"),
    )


def run(cmd: list[str], cwd: Path = PROJECT_ROOT, check: bool = True) -> subprocess.CompletedProcess:
    if shutil.which(cmd[0]) is None:
        raise SystemExit(f"Command not found: {cmd[0]}. Run: pip install -r requirements.txt")
    return subprocess.run(cmd, cwd=cwd, check=check, text=True, capture_output=True)


def print_result(result: subprocess.CompletedProcess) -> None:
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())


def ensure_dvc_repo() -> None:
    if (PROJECT_ROOT / ".dvc").exists():
        return
    cmd = ["dvc", "init"] if (PROJECT_ROOT / ".git").exists() else ["dvc", "init", "--no-scm"]
    result = run(cmd)
    print_result(result)


def ensure_minio_bucket(settings: MinIODVCSettings) -> None:
    try:
        from minio import Minio
        from minio.error import S3Error
    except ImportError as exc:
        raise SystemExit("MinIO client is not installed. Run: pip install -r requirements.txt") from exc

    client = Minio(
        settings.endpoint,
        access_key=settings.access_key,
        secret_key=settings.secret_key,
        secure=settings.secure,
    )
    try:
        exists = client.bucket_exists(settings.bucket)
    except S3Error as exc:
        raise SystemExit(f"Could not connect to MinIO at {settings.endpoint_url}: {exc}") from exc

    if exists:
        print(f"Bucket already exists: {settings.bucket}")
        return
    client.make_bucket(settings.bucket)
    print(f"Created bucket: {settings.bucket}")


def configure_dvc_remote(settings: MinIODVCSettings) -> None:
    ensure_dvc_repo()
    existing = run(["dvc", "remote", "list"], check=False)
    if settings.remote_name not in existing.stdout:
        print_result(run(["dvc", "remote", "add", "-d", settings.remote_name, settings.dvc_remote_url]))
    else:
        print_result(run(["dvc", "remote", "modify", settings.remote_name, "url", settings.dvc_remote_url]))

    print_result(run(["dvc", "remote", "modify", settings.remote_name, "endpointurl", settings.endpoint_url]))
    print_result(run(["dvc", "remote", "modify", "--local", settings.remote_name, "access_key_id", settings.access_key]))
    print_result(run(["dvc", "remote", "modify", "--local", settings.remote_name, "secret_access_key", settings.secret_key]))
    print_result(run(["dvc", "remote", "default", settings.remote_name]))


def check_dvc_remote() -> None:
    print_result(run(["dvc", "doctor"], check=False))
    print_result(run(["dvc", "remote", "list"], check=False))


def dvc_repro_push() -> None:
    print_result(run(["dvc", "repro"]))
    print_result(run(["dvc", "push"]))


def dvc_pull() -> None:
    print_result(run(["dvc", "pull"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local MinIO + DVC remote for the data lake pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Create bucket and configure DVC remote.")
    subparsers.add_parser("bucket", help="Create/check the MinIO bucket only.")
    subparsers.add_parser("remote", help="Configure DVC remote only.")
    subparsers.add_parser("check", help="Show DVC/remote diagnostics.")
    subparsers.add_parser("repro-push", help="Run dvc repro and dvc push.")
    subparsers.add_parser("pull", help="Run dvc pull.")
    args = parser.parse_args()

    settings = load_settings()
    if args.command == "init":
        ensure_minio_bucket(settings)
        configure_dvc_remote(settings)
        check_dvc_remote()
    elif args.command == "bucket":
        ensure_minio_bucket(settings)
    elif args.command == "remote":
        configure_dvc_remote(settings)
    elif args.command == "check":
        check_dvc_remote()
    elif args.command == "repro-push":
        dvc_repro_push()
    elif args.command == "pull":
        dvc_pull()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
