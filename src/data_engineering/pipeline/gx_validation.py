from __future__ import annotations

from pathlib import Path
from typing import Any

from src.shared.contracts.schemas import CONTRACTS


def _result_to_dict(result: Any) -> dict:
    if hasattr(result, "to_json_dict"):
        return result.to_json_dict()
    if hasattr(result, "dict"):
        return result.dict()
    return {
        "success": bool(getattr(result, "success", False)),
        "expectation_type": getattr(getattr(result, "expectation_config", None), "type", ""),
    }


def _expectations_for_contract(gx, contract_name: str, columns: list[str]) -> list[Any]:
    contract = CONTRACTS[contract_name]
    expectations = [
        gx.expectations.ExpectTableColumnsToContainSet(column_set=list(contract.required_columns)),
    ]

    for column, accepted_values in (contract.accepted_values or {}).items():
        if column in columns:
            expectations.append(
                gx.expectations.ExpectColumnValuesToBeInSet(
                    column=column,
                    value_set=list(accepted_values),
                )
            )

    if contract_name in {"video_features", "gold_training_dataset"} and "missing_feature_ratio" in columns:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="missing_feature_ratio",
                min_value=0.0,
                max_value=1.0,
            )
        )

    if contract_name == "gold_training_dataset" and "is_trainable" in columns:
        expectations.append(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column="is_trainable",
                value_set=[True, False],
            )
        )

    return expectations


def validate_dataframe_with_gx(df, contract_name: str, asset_name: str) -> dict:
    try:
        import great_expectations as gx
    except ImportError:
        return {
            "contract_name": contract_name,
            "asset_name": asset_name,
            "status": "unavailable",
            "success": False,
            "error_message": "great_expectations is not installed",
            "results": [],
        }

    try:
        context = gx.get_context(mode="ephemeral")
    except TypeError:
        context = gx.get_context()

    try:
        data_source = context.data_sources.add_pandas(f"pandas_{asset_name}")
        data_asset = data_source.add_dataframe_asset(name=asset_name)
        batch_definition = data_asset.add_batch_definition_whole_dataframe("whole_dataframe")
        batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

        results = []
        for expectation in _expectations_for_contract(gx, contract_name, list(df.columns)):
            result = batch.validate(expectation)
            results.append(_result_to_dict(result))

        success = all(bool(item.get("success")) for item in results)
        return {
            "contract_name": contract_name,
            "asset_name": asset_name,
            "status": "passed" if success else "failed",
            "success": success,
            "evaluated_expectations": len(results),
            "failed_expectations": sum(1 for item in results if not item.get("success")),
            "results": results,
        }
    except Exception as exc:
        return {
            "contract_name": contract_name,
            "asset_name": asset_name,
            "status": "error",
            "success": False,
            "error_message": str(exc),
            "results": [],
        }


def validate_tables_with_gx(tables: dict[str, tuple[object, str | Path]]) -> dict:
    results = []
    for contract_name, (df, asset_path) in tables.items():
        if getattr(df, "empty", True):
            results.append(
                {
                    "contract_name": contract_name,
                    "asset_name": str(asset_path),
                    "status": "missing",
                    "success": False,
                    "error_message": "empty dataframe",
                    "results": [],
                }
            )
            continue
        results.append(validate_dataframe_with_gx(df, contract_name, str(asset_path).replace("/", "_")))

    return {
        "status": "passed" if results and all(item.get("success") for item in results) else "failed",
        "validated_assets": len(results),
        "failed_assets": sum(1 for item in results if not item.get("success")),
        "results": results,
    }
