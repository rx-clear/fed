from __future__ import annotations

import csv
import math
import os
from typing import Dict, Optional


class ExperimentRecorder:

    # Keep the original columns first so existing result files remain easy to
    # consume.  Protocol columns are optional and are populated by the coded
    # experiment when available.
    FIELDNAMES = [
        "round",
        "train_acc",
        "train_loss",
        "test_acc",
        "test_loss",
        "dmcfe_fedavg_max_error",
        "dmcfe_fedavg_mean_error",
        "dmcfe_fedrep_max_error",
        "dmcfe_fedrep_mean_error",
        "dropout_rate",
        "online_clients",
        "online_client_ids",
        "recovery_threshold",
        "coded_communication_rounds",
        "secagg_baseline_rounds",
        "recovery_error",
        "dmcfe_weight_signatures_verified",
        "run_id",
        "method",
        "seed",
        "config_hash",
        "matrix_id",
        "backend",
        "backend_secure",
        "participation_rate",
        "protocol_dropout_rate",
        "coordinates",
        "chunks",
        "chunk_size",
        "scale",
        "clip_bound",
        "clipped_fraction",
        "encode_seconds",
        "aggregate_seconds",
        "total_aggregation_seconds",
        "serialized_upload_bytes",
        "serialized_key_bytes",
        "recovery_bytes",
        "peak_memory_bytes",
    ]

    def __init__(
        self,
        output_path: str,
    ):
        self.output_path = output_path

        output_dir = os.path.dirname(
            output_path
        )

        if output_dir:
            os.makedirs(
                output_dir,
                exist_ok=True,
            )

        # If a caller resumes an older experiment, preserve its header rather
        # than writing a wider row into a narrow CSV.  New files get all
        # protocol columns above.
        self.fieldnames = list(self.FIELDNAMES)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            with open(output_path, "r", newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), [])
            if header:
                self.fieldnames = header
        else:
            with open(
                output_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.DictWriter(
                    f,
                    fieldnames=self.fieldnames,
                )

                writer.writeheader()

    def append(
        self,
        round_idx: int,
        train_acc: Optional[float],
        train_loss: Optional[float],
        test_acc: Optional[float],
        test_loss: Optional[float],
        dmcfe_fedavg_max_error: Optional[float] = None,
        dmcfe_fedavg_mean_error: Optional[float] = None,
        dmcfe_fedrep_max_error: Optional[float] = None,
        dmcfe_fedrep_mean_error: Optional[float] = None,
        dropout_rate: Optional[float] = None,
        online_clients: Optional[int] = None,
        online_client_ids: Optional[str] = None,
        recovery_threshold: Optional[int] = None,
        coded_communication_rounds: Optional[int] = None,
        secagg_baseline_rounds: Optional[int] = None,
        recovery_error: Optional[float] = None,
        dmcfe_weight_signatures_verified: Optional[int] = None,
        run_id: Optional[str] = None,
        method: Optional[str] = None,
        seed: Optional[int] = None,
        config_hash: Optional[str] = None,
        matrix_id: Optional[str] = None,
        backend: Optional[str] = None,
        backend_secure: Optional[bool] = None,
        participation_rate: Optional[float] = None,
        protocol_dropout_rate: Optional[float] = None,
        coordinates: Optional[int] = None,
        chunks: Optional[int] = None,
        chunk_size: Optional[int] = None,
        scale: Optional[int] = None,
        clip_bound: Optional[float] = None,
        clipped_fraction: Optional[float] = None,
        encode_seconds: Optional[float] = None,
        aggregate_seconds: Optional[float] = None,
        total_aggregation_seconds: Optional[float] = None,
        serialized_upload_bytes: Optional[int] = None,
        serialized_key_bytes: Optional[int] = None,
        recovery_bytes: Optional[int] = None,
        peak_memory_bytes: Optional[int] = None,
    ) -> Dict[str, object]:

        def _finite_or_blank(value: Optional[float], name: str):
            if value is None:
                return ""
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite when provided")
            return value

        def _optional_nonnegative_int(value: Optional[int], name: str):
            if value is None:
                return ""
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            return int(value)

        if int(round_idx) != round_idx or int(round_idx) < 0:
            raise ValueError("round_idx must be a non-negative integer")

        row = {
            "round": int(round_idx),
            "train_acc": _finite_or_blank(train_acc, "train_acc"),
            "train_loss": _finite_or_blank(train_loss, "train_loss"),
            "test_acc": _finite_or_blank(test_acc, "test_acc"),
            "test_loss": _finite_or_blank(test_loss, "test_loss"),
            "dmcfe_fedavg_max_error": _finite_or_blank(
                dmcfe_fedavg_max_error, "dmcfe_fedavg_max_error"
            ),
            "dmcfe_fedavg_mean_error": _finite_or_blank(
                dmcfe_fedavg_mean_error, "dmcfe_fedavg_mean_error"
            ),
            "dmcfe_fedrep_max_error": _finite_or_blank(
                dmcfe_fedrep_max_error, "dmcfe_fedrep_max_error"
            ),
            "dmcfe_fedrep_mean_error": _finite_or_blank(
                dmcfe_fedrep_mean_error, "dmcfe_fedrep_mean_error"
            ),
            "dropout_rate": _finite_or_blank(dropout_rate, "dropout_rate"),
            "online_clients": _optional_nonnegative_int(online_clients, "online_clients"),
            "online_client_ids": "" if online_client_ids is None else str(online_client_ids),
            "recovery_threshold": _optional_nonnegative_int(
                recovery_threshold, "recovery_threshold"
            ),
            "coded_communication_rounds": _optional_nonnegative_int(
                coded_communication_rounds, "coded_communication_rounds"
            ),
            "secagg_baseline_rounds": _optional_nonnegative_int(
                secagg_baseline_rounds, "secagg_baseline_rounds"
            ),
            "recovery_error": _finite_or_blank(recovery_error, "recovery_error"),
            "dmcfe_weight_signatures_verified": _optional_nonnegative_int(
                dmcfe_weight_signatures_verified,
                "dmcfe_weight_signatures_verified",
            ),
            "run_id": "" if run_id is None else str(run_id),
            "method": "" if method is None else str(method),
            "seed": "" if seed is None else int(seed),
            "config_hash": "" if config_hash is None else str(config_hash),
            "matrix_id": "" if matrix_id is None else str(matrix_id),
            "backend": "" if backend is None else str(backend),
            "backend_secure": "" if backend_secure is None else bool(backend_secure),
            "participation_rate": _finite_or_blank(participation_rate, "participation_rate"),
            "protocol_dropout_rate": _finite_or_blank(protocol_dropout_rate, "protocol_dropout_rate"),
            "coordinates": _optional_nonnegative_int(coordinates, "coordinates"),
            "chunks": _optional_nonnegative_int(chunks, "chunks"),
            "chunk_size": _optional_nonnegative_int(chunk_size, "chunk_size"),
            "scale": _optional_nonnegative_int(scale, "scale"),
            "clip_bound": _finite_or_blank(clip_bound, "clip_bound"),
            "clipped_fraction": _finite_or_blank(clipped_fraction, "clipped_fraction"),
            "encode_seconds": _finite_or_blank(encode_seconds, "encode_seconds"),
            "aggregate_seconds": _finite_or_blank(aggregate_seconds, "aggregate_seconds"),
            "total_aggregation_seconds": _finite_or_blank(total_aggregation_seconds, "total_aggregation_seconds"),
            "serialized_upload_bytes": _optional_nonnegative_int(serialized_upload_bytes, "serialized_upload_bytes"),
            "serialized_key_bytes": _optional_nonnegative_int(serialized_key_bytes, "serialized_key_bytes"),
            "recovery_bytes": _optional_nonnegative_int(recovery_bytes, "recovery_bytes"),
            "peak_memory_bytes": _optional_nonnegative_int(peak_memory_bytes, "peak_memory_bytes"),
        }

        with open(
            self.output_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )

            writer.writerow(row)
        return row
