from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import pandas as pd

from src.shared.contracts.schemas import CONTRACTS
from src.shared.core.io_utils import read_dataframe


@dataclass(frozen=True)
class ContractValidationResult:
    contract_name: str
    path: str
    exists: bool
    row_count: int
    missing_columns: tuple[str, ...]
    invalid_values: dict[str, tuple[str, ...]]
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_dataframe_contract(
    df: "pd.DataFrame",
    contract_name: str,
    path: str | Path = "",
) -> ContractValidationResult:
    contract = CONTRACTS[contract_name]
    missing_columns = tuple(column for column in contract.required_columns if column not in df.columns)
    invalid_values: dict[str, tuple[str, ...]] = {}

    for column, accepted in (contract.accepted_values or {}).items():
        if column not in df.columns:
            continue
        values = df[column].fillna("").astype(str).str.strip()
        invalid = sorted(set(values) - set(accepted))
        if invalid:
            invalid_values[column] = tuple(invalid)

    status = "passed" if not missing_columns and not invalid_values else "failed"
    return ContractValidationResult(
        contract_name=contract_name,
        path=str(path),
        exists=True,
        row_count=int(len(df)),
        missing_columns=missing_columns,
        invalid_values=invalid_values,
        status=status,
    )


def validate_table_contract(path: str | Path, contract_name: str) -> ContractValidationResult:
    path = Path(path)
    if not path.exists() and not path.with_suffix(".csv").exists() and not path.with_suffix(".parquet").exists():
        return ContractValidationResult(
            contract_name=contract_name,
            path=str(path),
            exists=False,
            row_count=0,
            missing_columns=CONTRACTS[contract_name].required_columns,
            invalid_values={},
            status="missing",
        )
    df = read_dataframe(path)
    return validate_dataframe_contract(df, contract_name, path)


def summarize_validation_results(results: Iterable[ContractValidationResult]) -> dict:
    items = [result.to_dict() for result in results]
    return {
        "status": "passed" if all(item["status"] == "passed" for item in items) else "failed",
        "validated_assets": len(items),
        "failed_assets": sum(1 for item in items if item["status"] != "passed"),
        "results": items,
    }
