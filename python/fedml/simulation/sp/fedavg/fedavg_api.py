import copy
import logging
import random

import numpy as np
import torch
try:
    import wandb
except ImportError:  # W&B is optional for local/reproducible experiments.
    wandb = None

from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
from .client import Client

from .quantization import quantize, dequantize
from .dmcfe.scalar_smoke import run_scalar_mcfe_smoke
from .dmcfe.no_dropout_smoke import (
    run_dmcfe_no_dropout_smoke
)
from .dmcfe.secret_sharing_smoke import (
    run_secret_sharing_smoke,
)
from .dmcfe.symmetric_encryption_smoke import (
    run_symmetric_encryption_smoke,
)
from .dmcfe.round1_keysharing_smoke import (
    run_round1_keysharing_smoke,
)
from .dmcfe.dropout_smoke import (
    run_dmcfe_dropout_smoke,
)
from .dmcfe.dropout_e2e_smoke import (
    run_dmcfe_dropout_e2e_smoke,
)
from .dmcfe.vector_dropout_smoke import (
    run_dmcfe_vector_dropout_smoke,
)
from .dmcfe.real_delta_smoke import (
    run_real_quantized_delta_smoke,
)
from .dmcfe.weight_signatures import (
    WeightSigner,
    verify_weight_proposals,
)
from .experiment_recorder import (
    ExperimentRecorder,
)
from .vectorize import (
    flatten_delta,
    unflatten_delta,
)
from .privacy_baselines import BatchCryptPrototype, PairwiseMaskingAggregator
from .optimized_dmcfe_ip import OptimizedDMCFEIPAggregator, ToyMCFEBackend

class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        # Validate optional telemetry once, before any client work starts.
        if bool(getattr(args, "enable_wandb", False)) and wandb is None:
            raise RuntimeError(
                "enable_wandb is true, but the optional 'wandb' package is not installed"
            )
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        logging.info("model = {}".format(model))
        self.model_trainer = create_model_trainer(model, args)
        self.model = model
        # ============================================================
        # Experiment recorder
        # ============================================================

        enable_dmcfe = bool(
            getattr(
                self.args,
                "enable_dmcfe",
                False
            )
        )
        logging.info(
            "DMCFE aggregation enabled: %s",
            enable_dmcfe
        )
        run_dmcfe_smoke_tests = bool(
            getattr(
                self.args,
                "run_dmcfe_smoke_tests",
                False
            )
        )

        logging.info(
            "DMCFE smoke tests enabled: %s",
            run_dmcfe_smoke_tests
        )
        result_file = self._get_result_file(enable_dmcfe)


        self.experiment_recorder = (
            ExperimentRecorder(
                output_path=result_file
            )
        )

        # 当前轮的 DMCFE/FedAvg 对比误差
        self.current_dmcfe_max_error = None
        self.current_dmcfe_mean_error = None
        self.current_online_clients = None
        self.current_online_client_ids = None
        self.current_recovery_threshold = None
        self.current_dropout_rate = None
        self.current_participation_rate = None
        self.current_aggregation_diagnostics = None
        self.current_protocol_recovery_active = False
        self.current_dmcfe_weight_signatures_verified = None
        self._current_dmcfe_signed_weights = None
        self._dmcfe_weight_signers = {}
        self._optimized_dmcfe_aggregator = None
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            self.model_trainer,
        )

    def _wandb_enabled(self):
        """Return whether W&B logging is requested and available."""
        enabled = bool(getattr(self.args, "enable_wandb", False))
        if enabled and wandb is None:
            raise RuntimeError(
                "enable_wandb is true, but the optional 'wandb' package is not installed"
            )
        return enabled

    def _get_result_file(self, enable_dmcfe):
        """Return the metrics path; personalized algorithms may override it."""
        configured_path = getattr(self.args, "experiment_result_file", None)
        if configured_path:
            return str(configured_path)
        if enable_dmcfe:
            return "results/dmcfe_fedavg.csv"
        return "results/fedavg_baseline.csv"

    def _get_algorithm_name(self):
        return "FedAvg"

    def _get_dmcfe_metric_kwargs(self):
        if self._get_algorithm_name() == "FedRep":
            return {
                "dmcfe_fedavg_max_error": None,
                "dmcfe_fedavg_mean_error": None,
                "dmcfe_fedrep_max_error": self.current_dmcfe_max_error,
                "dmcfe_fedrep_mean_error": self.current_dmcfe_mean_error,
            }
        return {
            "dmcfe_fedavg_max_error": self.current_dmcfe_max_error,
            "dmcfe_fedavg_mean_error": self.current_dmcfe_mean_error,
            "dmcfe_fedrep_max_error": None,
            "dmcfe_fedrep_mean_error": None,
        }

    def _record_metrics(self, round_idx, train_acc=None, train_loss=None,
                        test_acc=None, test_loss=None):
        """Persist one round with provenance and available aggregation diagnostics."""
        diagnostics = self.current_aggregation_diagnostics
        enabled = bool(getattr(self.args, "enable_dmcfe", False))
        privacy = getattr(self.args, "privacy_aggregation", None)
        method = getattr(self.args, "experiment_method", None)
        if not method:
            method = self._get_algorithm_name()
            if enabled:
                method += "+FixedPointReference"
            elif privacy:
                method += "+" + str(privacy)
        backend = (
            diagnostics.backend if diagnostics is not None else
            ("smoke-reference" if enabled else str(privacy or "plaintext"))
        )
        self.experiment_recorder.append(
            round_idx=round_idx,
            train_acc=train_acc,
            train_loss=train_loss,
            test_acc=test_acc,
            test_loss=test_loss,
            run_id=getattr(self.args, "experiment_run_id", None),
            method=method,
            seed=getattr(self.args, "random_seed", None),
            config_hash=getattr(self.args, "experiment_config_hash", None),
            matrix_id=getattr(self.args, "experiment_matrix_id", None),
            backend=backend,
            backend_secure=diagnostics.backend_secure if diagnostics else False,
            dropout_rate=self.current_dropout_rate,
            participation_rate=self.current_participation_rate,
            protocol_dropout_rate=self.current_dropout_rate,
            online_clients=self.current_online_clients,
            online_client_ids=self.current_online_client_ids,
            recovery_threshold=self.current_recovery_threshold,
            coded_communication_rounds=1 if self.current_protocol_recovery_active else None,
            recovery_error=(
                self.current_dmcfe_max_error
                if self.current_protocol_recovery_active else None
            ),
            dmcfe_weight_signatures_verified=self.current_dmcfe_weight_signatures_verified,
            coordinates=diagnostics.coordinates if diagnostics else None,
            chunks=diagnostics.chunks if diagnostics else None,
            chunk_size=diagnostics.chunk_size if diagnostics else None,
            scale=diagnostics.scale if diagnostics else None,
            clip_bound=diagnostics.clip_bound if diagnostics else None,
            clipped_fraction=diagnostics.clipped_fraction if diagnostics else None,
            encode_seconds=diagnostics.encode_seconds if diagnostics else None,
            aggregate_seconds=diagnostics.aggregate_seconds if diagnostics else None,
            total_aggregation_seconds=(
                diagnostics.total_aggregation_seconds if diagnostics else None
            ),
            **self._get_dmcfe_metric_kwargs(),
        )

    def _get_aggregated_param_order(self, model_params):
        """Return state-dict keys that participate in server aggregation."""
        return list(model_params.keys())

    def _verify_dmcfe_weights(self, round_idx, participant_ids):
        """Authenticate the weights used by DMCFE functional key derivation."""
        if not bool(getattr(self.args, "dmcfe_enable_weight_signatures", True)):
            self.current_dmcfe_weight_signatures_verified = None
            self._current_dmcfe_signed_weights = None
            return None
        participant_ids = [int(client_id) for client_id in participant_ids]
        for client_id in participant_ids:
            self._dmcfe_weight_signers.setdefault(client_id, WeightSigner())
        proposals = [
            self._dmcfe_weight_signers[client_id].propose(
                round_idx,
                participant_ids,
                client_id,
                int(self.train_data_local_num_dict[client_id]),
            )
            for client_id in participant_ids
        ]
        public_keys = {
            client_id: signer.public_key_bytes
            for client_id, signer in self._dmcfe_weight_signers.items()
        }
        weights = verify_weight_proposals(proposals, public_keys, round_idx, participant_ids)
        self._current_dmcfe_signed_weights = {
            int(client_id): int(weight)
            for client_id, weight in zip(participant_ids, weights)
        }
        self.current_dmcfe_weight_signatures_verified = len(proposals)
        return proposals

    def _get_dmcfe_dropout_ids(self, name):
        value = getattr(self.args, name, None)
        if value is None or value == "":
            return []
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("["):
                import json
                value = json.loads(value)
            else:
                value = [item for item in value.split(",") if item.strip()]
        return [int(item) for item in value]

    def _aggregate_dmcfe_optimized(self, local_deltas, key_order, round_idx):
        """Aggregate the complete shared state through the optimized backend."""
        if self._optimized_dmcfe_aggregator is None:
            backend_name = str(
                getattr(self.args, "dmcfe_backend", "chunked_reference")
            ).strip().lower()
            if backend_name in {"toy_mcfe", "mcfe"}:
                backend = ToyMCFEBackend(
                    modulus_bits=int(getattr(self.args, "dmcfe_modulus_bits", 256))
                )
            elif backend_name in {"chunked_reference", "reference"}:
                backend = None
            else:
                raise ValueError(
                    "dmcfe_backend must be 'chunked_reference' or 'toy_mcfe'"
                )
            self._optimized_dmcfe_aggregator = OptimizedDMCFEIPAggregator(
                scale=int(getattr(self.args, "dmcfe_scale", 1 << 14)),
                clip_bound=float(getattr(self.args, "dmcfe_clip_bound", 8.0)),
                chunk_size=int(getattr(self.args, "dmcfe_chunk_size", 1 << 16)),
                backend=backend,
            )
        return self._optimized_dmcfe_aggregator.aggregate_deltas(
            local_deltas,
            key_order=key_order,
            round_idx=round_idx,
        )

    def _prepare_client_for_evaluation(self, client, client_idx):
        """Hook for algorithms that need client-specific evaluation state."""

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = {
            key: value.detach().clone()
            for key, value in self.model_trainer.get_model_params().items()
        }
        # ============================================================
        # DMCFE experiment switch
        # ============================================================

        enable_dmcfe = bool(
            getattr(
                self.args,
                "enable_dmcfe",
                False
            )
        )

        logging.info(
            "DMCFE aggregation enabled: %s",
            enable_dmcfe
        )
        run_dmcfe_smoke_tests = bool(
            getattr(
                self.args,
                "run_dmcfe_smoke_tests",
                False
            )
        )

        dmcfe_aggregation_mode = str(
            getattr(self.args, "dmcfe_aggregation_mode", "smoke")
        ).strip().lower()
        if dmcfe_aggregation_mode not in {"smoke", "optimized"}:
            raise ValueError(
                "dmcfe_aggregation_mode must be either 'smoke' or 'optimized'"
            )
        use_optimized_dmcfe = (
            enable_dmcfe and dmcfe_aggregation_mode == "optimized"
        )
        if use_optimized_dmcfe and (
            self._get_dmcfe_dropout_ids("dmcfe_dropout_before_encryption_ids")
            or self._get_dmcfe_dropout_ids("dmcfe_dropout_before_recovery_ids")
        ):
            raise ValueError(
                "optimized fixed-point aggregation does not implement protocol dropout recovery"
            )

        logging.info(
            "DMCFE smoke tests enabled: %s",
            run_dmcfe_smoke_tests
        )
        logging.info(
            "DMCFE aggregation mode: %s",
            dmcfe_aggregation_mode
        )
        # ============================================================
        # Canonical model parameter order
        # ============================================================

        model_param_order = self._get_aggregated_param_order(w_global)
        if not model_param_order:
            raise ValueError("At least one model parameter must be aggregated")

        logging.info(
            "Canonical model parameter order: %s",
            model_param_order
        )
        mlops.log_training_status(
            mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING
        )
        mlops.log_aggregation_status(
            mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING
        )
        mlops.log_round_info(self.args.comm_round, -1)
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))
            self.current_dmcfe_max_error = None
            self.current_dmcfe_mean_error = None
            self.current_online_clients = None
            self.current_online_client_ids = None
            self.current_recovery_threshold = None
            self.current_dropout_rate = None
            self.current_participation_rate = None
            self.current_aggregation_diagnostics = None
            self.current_protocol_recovery_active = False
            self.current_dmcfe_weight_signatures_verified = None
            self._current_dmcfe_signed_weights = None
            if (
                enable_dmcfe
                and
                run_dmcfe_smoke_tests
                and
                self.args.rank == 0
                and round_idx == 0
                and getattr(
                    self.args,
                    "enable_dmcfe_debug",
                    True
                )
            ):
                logging.info(
                    "Running DMCFE secret sharing smoke test..."
                )

                run_secret_sharing_smoke()

                logging.info(
                    "Running DMCFE symmetric encryption smoke test..."
                )

                run_symmetric_encryption_smoke()
                logging.info(
                    "Running DMCFE Round-1 KeySharing smoke test..."
                )

                run_round1_keysharing_smoke()
                logging.info(
                    "Running DMCFE dropout Round-3 smoke test..."
                )
                run_dmcfe_dropout_smoke()
                logging.info(
                    "Running DMCFE dropout E2E scalar smoke test..."
                )

                run_dmcfe_dropout_e2e_smoke()
                logging.info(
                    "Running DMCFE 4-D vector dropout regression test..."
                )

                run_dmcfe_vector_dropout_smoke(
                    vector_dim=4
                )


                logging.info(
                    "Running DMCFE 32-D vector dropout smoke test..."
                )

                run_dmcfe_vector_dropout_smoke(
                    vector_dim=32
                )
            w_locals = []

            # 保存这一轮每个客户端的一个量化值
            # scalar regression
            mcfe_scalar_inputs = []
            # ============================================================
            # REAL FedML quantized vectors for DMCFE integration
            # ============================================================

            real_quantized_inputs = []
            # REAL FedAvg client sample counts
            real_sample_nums = []
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            self.current_online_clients = len(client_indexes)
            self.current_online_client_ids = ",".join(
                str(int(client_id)) for client_id in client_indexes
            )
            self.current_recovery_threshold = None
            self.current_participation_rate = (
                len(client_indexes) / float(self.args.client_num_in_total)
            )
            logging.info("client_indexes = " + str(client_indexes))
            if enable_dmcfe:
                self._verify_dmcfe_weights(round_idx, client_indexes)

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # train on new dataset
                mlops.event(
                    "train",
                    event_started=True,
                    event_value="{}_{}".format(str(round_idx), str(idx)),
                )

                # ============================================================
                # 1. 在客户端训练之前，保存本轮全局模型的真正独立副本
                # ============================================================

                global_before = {
                    key: value.detach().clone()
                    for key, value in w_global.items()
                }


                # ============================================================
                # 2. 客户端基于同一个 global model 做本地训练
                # ============================================================

                w_local = client.train(
                    copy.deepcopy(global_before)
                )


                # ============================================================
                # 3. 客户端训练完成后，也立即复制一份本地模型
                # 防止后续共享 trainer 时引用继续发生变化
                # ============================================================

                w_local_copy = {
                    key: value.detach().clone()
                    for key, value in w_local.items()
                }


                # ============================================================
                # 4. 计算模型更新量：
                # delta_w = W_local - W_global
                # ============================================================

                delta_w = {}

                for key in model_param_order:

                    delta_w[key] = (
                        w_local_copy[key]
                        -
                        global_before[key]
                    )


                # ============================================================
                # 5. 检查 delta 是否真的非 0
                # ============================================================
                # ============================================================
                # 6. Tensor 字典 -> 一维向量
                # ============================================================

                delta_vector = flatten_delta(
                    delta_w,
                    key_order=model_param_order
                )

                logging.info(
                    "client %s delta dim=%d norm=%.6f nonzero=%d",
                    client_idx,
                    delta_vector.numel(),
                    torch.norm(delta_vector).item(),
                    torch.count_nonzero(delta_vector).item()
                )

                # ============================================================
                # Fixed-point quantization
                # precision = 14
                # ============================================================
                if enable_dmcfe:

                    quantized_vector = quantize(
                        delta_vector,
                        precision=14
                    )
                    # ============================================================
                    # Save REAL quantized local update
                    #
                    # Important:
                    # clone + cpu prevents later trainer/model reuse from
                    # changing the stored vector.
                    # ============================================================

                    real_quantized_inputs.append(
                        quantized_vector
                        .detach()
                        .cpu()
                        .clone()
                    )
                    logging.info(
                        "client %s quantized vector saved, dim=%d",
                        client_idx,
                        quantized_vector.numel()
                    )
                    # ============================================================
                    # MCFE scalar smoke test input
                    #
                    # 当前只取一个参数
                    # ============================================================

                    scalar_value = int(
                        quantized_vector[0].item()
                    )


                    mcfe_scalar_inputs.append(
                        scalar_value
                    )

                    restored_vector = dequantize(
                        quantized_vector,
                        precision=14
                    )

                    original_vector = (
                        delta_vector
                        .detach()
                        .cpu()
                        .to(torch.float64)
                    )


                    quant_error = torch.abs(
                        original_vector
                        -
                        restored_vector.cpu()
                    )




                    logging.info(
                        "client %s quantization error max=%.8f mean=%.8f",
                        client_idx,
                        quant_error.max().item(),
                        quant_error.mean().item()
                    )

                mlops.event(
                    "train",
                    event_started=False,
                    event_value="{}_{}".format(str(round_idx), str(idx)),
                )
                # self.logging.info("local weights = " + str(w))

                logging.info(
                    "client %s delta norm=%f",
                    client_idx,
                    torch.norm(delta_vector).item()
                )

                logging.info(
                    "client %s delta nonzero=%d",
                    client_idx,
                    torch.count_nonzero(delta_vector).item()
                )
                sample_num = int(
                    client.get_sample_number()
                )

                if self._current_dmcfe_signed_weights is not None:
                    signed_sample_num = self._current_dmcfe_signed_weights.get(int(client_idx))
                    if signed_sample_num != sample_num:
                        raise ValueError(
                            "DMCFE signed sample weight does not match client update"
                        )

                # FedAvg always needs this
                w_locals.append((sample_num, copy.deepcopy(delta_w)))
                if enable_dmcfe:
                    real_sample_nums.append(
                        sample_num
                    )
            # ============================================================
            # 三个客户端全部完成后
            # 做一次 MCFE scalar correctness test
            # ============================================================
            # ============================================================
            # REAL FedML quantized delta -> DMCFE
            #
            # First checkpoint:
            # only communication round 0
            # only first 32 real coordinates
            # ============================================================

            if enable_dmcfe and not use_optimized_dmcfe:
                logging.info(
                        "Running REAL FedML quantized-delta "
                        "7850-D DMCFE test for round %d...",
                        round_idx
                )
                logging.info(
                        "All clients finished local training"
                )
                logging.info(
                        "real_quantized_inputs=%d",
                        len(real_quantized_inputs)
                )

                logging.info(
                        "real_sample_nums=%s",
                        real_sample_nums
                )
                logging.info(
                        "client_indexes=%d",
                        len(client_indexes)
                )

                requested_validation_dim = int(
                    getattr(self.args, "dmcfe_validation_dim", 7850)
                )
                if requested_validation_dim < 1:
                    raise ValueError("dmcfe_validation_dim must be positive")
                validation_dim = min(
                    requested_validation_dim,
                    int(real_quantized_inputs[0].numel()),
                )
                if validation_dim != requested_validation_dim:
                    logging.warning(
                        "dmcfe_validation_dim=%d exceeds model dimension=%d; using %d",
                        requested_validation_dim,
                        int(real_quantized_inputs[0].numel()),
                        validation_dim,
                    )
                logging.info(
                    "DMCFE equivalence will be checked on %d/%d coordinates",
                    validation_dim,
                    int(real_quantized_inputs[0].numel()),
                )

                dmcfe_result = run_real_quantized_delta_smoke(
                        client_vectors=real_quantized_inputs,
                        client_sample_nums=real_sample_nums,
                        sampled_client_ids=[
                            int(client_id)
                            for client_id in client_indexes
                        ],
                        round_idx=round_idx,
                        vector_dim=validation_dim,
                        dropout_before_encryption_ids=self._get_dmcfe_dropout_ids(
                            "dmcfe_dropout_before_encryption_ids"
                        ),
                        dropout_before_recovery_ids=self._get_dmcfe_dropout_ids(
                            "dmcfe_dropout_before_recovery_ids"
                        ),
                        recovery_threshold=getattr(
                            self.args, "dmcfe_recovery_threshold", None
                        ),
                )
                self.current_online_clients = dmcfe_result["active_client_count"]
                self.current_online_client_ids = ",".join(
                    str(int(client_id)) for client_id in dmcfe_result["active_client_ids"]
                )
                self.current_recovery_threshold = dmcfe_result["recovery_threshold"]
                self.current_protocol_recovery_active = True
                self.current_dropout_rate = 1.0 - (
                    self.current_online_clients / float(len(client_indexes))
                )
                dmcfe_quantized_avg = (
                        dmcfe_result["normalized_vector"]
                )
                    # ============================================================
                    # Dequantization
                    #
                    # quantize:
                    #       q = round(x * 2^precision)
                    #
                    # dequantize:
                    #       x = q / 2^precision
                    #
                    # ============================================================

                precision = 14


                dmcfe_float_delta_vector = (
                        dmcfe_quantized_avg
                        /
                        (2 ** precision)
                )
                logging.info(
                        "DMCFE dequantized vector preview:"
                )

                logging.info(
                        dmcfe_float_delta_vector[:10]
                )

            if (enable_dmcfe
                and not use_optimized_dmcfe
                and
                run_dmcfe_smoke_tests
            ):
                run_scalar_mcfe_smoke(
                    client_values=mcfe_scalar_inputs,
                    round_idx=round_idx
                )
                run_dmcfe_no_dropout_smoke(
                    client_values=mcfe_scalar_inputs,
                    round_idx=round_idx
                )
            # update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))

            # ============================================================
            # Baseline FedAvg aggregation
            # ============================================================

            delta_global = self._aggregate(
                w_locals
            )
            #单元测试
            plain_probe = flatten_delta(
                delta_global,
                key_order=model_param_order
            )

            plain_restored = unflatten_delta(
                plain_probe,
                reference_dict=delta_global,
                key_order=model_param_order,
            )

            plain_probe_again = flatten_delta(
                plain_restored,
                key_order=model_param_order
            )

            plain_roundtrip_error = torch.abs(
                plain_probe.to(torch.float64)
                -
                plain_probe_again.to(torch.float64)
            )

            logging.info(
                "Plain flatten/unflatten sanity "
                "max error = %.10f",
                plain_roundtrip_error.max().item()
            )

            assert (
                plain_roundtrip_error.max().item()
                <
                1e-7
            )
            # ============================================================
            # Compare FedAvg and DMCFE aggregation
            # ============================================================

                # ============================================================
            # Restore DMCFE aggregated vector to model parameter structure
            # ============================================================
            if enable_dmcfe and use_optimized_dmcfe:
                    optimized_result = self._aggregate_dmcfe_optimized(
                        w_locals,
                        model_param_order,
                        round_idx,
                    )
                    dmcfe_delta_global = optimized_result.deltas
                    diagnostics = optimized_result.diagnostics
                    self.current_aggregation_diagnostics = diagnostics
                    self.current_dmcfe_max_error = diagnostics.max_abs_error
                    self.current_dmcfe_mean_error = diagnostics.mean_abs_error
                    logging.info(
                        "Optimized DMCFE-%s round=%d backend=%s coordinates=%d "
                        "chunks=%d clipped=%.6f max_error=%.8f",
                        self._get_algorithm_name(),
                        round_idx,
                        diagnostics.backend,
                        diagnostics.coordinates,
                        diagnostics.chunks,
                        diagnostics.clipped_fraction,
                        diagnostics.max_abs_error,
                    )
                    for k in model_param_order:
                        w_global[k] = w_global[k] + dmcfe_delta_global[k]
            elif enable_dmcfe:
                    fedavg_vector = flatten_delta(
                            delta_global,
                            key_order=model_param_order
                    )
                    fedavg_vector_cpu = (
                        fedavg_vector.detach().cpu().to(torch.float64)
                    )
                    dmcfe_checked_vector = dmcfe_float_delta_vector.detach().cpu().to(
                        torch.float64
                    )
                    if dmcfe_checked_vector.numel() != validation_dim:
                        raise ValueError(
                            "DMCFE validation output has an unexpected dimension"
                        )
                    # The real smoke test may intentionally cover only a
                    # prefix for large models.  Keep the untouched suffix from
                    # the regular FedAvg delta so model reconstruction remains
                    # shape-correct, while equivalence is reported only for
                    # coordinates actually checked by DMCFE.
                    error = torch.abs(
                        fedavg_vector_cpu[:validation_dim] - dmcfe_checked_vector
                    )
                    dmcfe_full_vector = fedavg_vector_cpu.clone()
                    dmcfe_full_vector[:validation_dim] = dmcfe_checked_vector
                    # ========================================================
                    # DMCFE mode
                    # ========================================================
                    dmcfe_delta_global = (
                        unflatten_delta(
                            flat_vector=dmcfe_full_vector,
                            reference_dict=(
                                delta_global
                            ),
                            key_order=model_param_order,
                        )
                    )
                    dmcfe_roundtrip_vector = (
                        flatten_delta(
                            dmcfe_delta_global,
                            key_order=model_param_order
                        )
                        .detach()
                        .cpu()
                        .to(torch.float64)
                    )


                    roundtrip_error = torch.abs(
                        dmcfe_roundtrip_vector
                        - dmcfe_full_vector
                    )


                    logging.info(
                        "DMCFE unflatten/flatten "
                        "roundtrip max error = %.10f",
                        roundtrip_error.max().item()
                    )
                    assert (
                        roundtrip_error.max().item()
                        <
                        1e-6
                    ), (
                        "DMCFE vector mapping failed: "
                        f"roundtrip_error="
                        f"{roundtrip_error.max().item()}"
                    )
                    self.current_dmcfe_max_error = (
                        error.max().item()
                    )

                    self.current_dmcfe_mean_error = (
                        error.mean().item()
                    )
                    logging.info(
                        "DMCFE-%s equivalence max error = %.10f",
                        self._get_algorithm_name(),
                        self.current_dmcfe_max_error
                    )

                    logging.info(
                        "DMCFE-%s equivalence mean error = %.10f",
                        self._get_algorithm_name(),
                        self.current_dmcfe_mean_error
                    )
                    # 保存flatten版本，用于和DMCFE比较

                    #fedavg_reference_vector = flatten_delta(
                    #    delta_global
                    #)
                    # ============================================================
                    # REAL DMCFE-driven global model update
                    # ============================================================

                    for k in model_param_order:

                        w_global[k] = (
                            w_global[k]
                            +
                            dmcfe_delta_global[k]
                        )
            else:

                # ========================================================
                # ORIGINAL FedAvg baseline
                # ========================================================

                logging.info(
                    "Using plaintext %s aggregation for round %d",
                    self._get_algorithm_name(),
                    round_idx
                )


                for k in model_param_order:

                    w_global[k] = (
                        w_global[k]
                        +
                        delta_global[k]
                    )

            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)
            else:
                self._record_metrics(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(
            range(test_data_num), min(num_samples, test_data_num)
        )
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size
        )
        self.val_global = sample_testset

    def _aggregate(self, w_locals):

        if not w_locals:
            raise ValueError("w_locals must contain at least one client update")

        privacy_mode = str(
            getattr(self.args, "privacy_aggregation", "") or ""
        ).strip().lower()
        if privacy_mode == "batchcrypt":
            return BatchCryptPrototype(
                precision=int(getattr(self.args, "batchcrypt_precision", 14)),
                modulus_bits=int(getattr(self.args, "batchcrypt_modulus_bits", 256)),
            ).aggregate(w_locals)
        if privacy_mode in {"masking", "pairwise_masking"}:
            return PairwiseMaskingAggregator(
                seed=getattr(self.args, "masking_seed", None)
            ).aggregate(w_locals)

        if not hasattr(w_locals[0][1], "keys"):
            raise TypeError("client updates must be mappings of parameter tensors")
        expected_keys = tuple(w_locals[0][1].keys())
        if not expected_keys:
            raise ValueError("client updates must contain at least one parameter")
        training_num = 0

        for idx in range(len(w_locals)):

            sample_num, local_delta = w_locals[idx]
            if not hasattr(local_delta, "keys"):
                raise TypeError("client updates must be mappings of parameter tensors")
            if isinstance(sample_num, bool) or int(sample_num) != sample_num:
                raise ValueError("sample counts must be integers")
            sample_num = int(sample_num)
            if sample_num <= 0:
                raise ValueError("sample counts must be positive")
            if set(local_delta.keys()) != set(expected_keys):
                raise ValueError(
                    "client updates must have identical parameter keys"
                )
            training_num += sample_num
        if training_num <= 0:
            raise ValueError("total sample count must be positive")


        global_delta = {}


        # 初始化
        for k, value in w_locals[0][1].items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"client update '{k}' must be a torch.Tensor")
            global_delta[k] = (
                value.detach().clone()
                if not torch.is_floating_point(value)
                else value.detach().clone().mul_(0)
            )


        # 加权求和
        for sample_num, local_delta in w_locals:

            weight = sample_num / training_num


            for k in expected_keys:
                value = local_delta[k]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"client update '{k}' must be a torch.Tensor")
                if value.shape != global_delta[k].shape:
                    raise ValueError(f"client update '{k}' has inconsistent shape")
                if torch.is_floating_point(value):
                    global_delta[k].add_(value, alpha=weight)
                elif not torch.equal(value, w_locals[0][1][k]):
                    raise ValueError(
                        f"integer client update '{k}' differs across clients"
                    )


        return global_delta

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            temp_w = []
            for _, local_w in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                client_idx,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            self._prepare_client_for_evaluation(client, client_idx)
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(
                copy.deepcopy(train_local_metrics["test_total"])
            )
            train_metrics["num_correct"].append(
                copy.deepcopy(train_local_metrics["test_correct"])
            )
            train_metrics["losses"].append(
                copy.deepcopy(train_local_metrics["test_loss"])
            )

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(
                copy.deepcopy(test_local_metrics["test_total"])
            )
            test_metrics["num_correct"].append(
                copy.deepcopy(test_local_metrics["test_correct"])
            )
            test_metrics["losses"].append(
                copy.deepcopy(test_local_metrics["test_loss"])
            )

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(
            train_metrics["num_samples"]
        )
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self._wandb_enabled():
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)
        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self._wandb_enabled():
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        mlops.log({"Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)
        # ============================================================
        # Save per-round experiment metrics
        # ============================================================

        self._record_metrics(
            round_idx=round_idx,
            train_acc=train_acc,
            train_loss=train_loss,
            test_acc=test_acc,
            test_loss=test_loss,
        )

    def _local_test_on_validation_set(self, round_idx):

        logging.info(
            "################local_test_on_validation_set : {}".format(round_idx)
        )

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        self._prepare_client_for_evaluation(client, 0)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self._wandb_enabled():
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self._wandb_enabled():
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception(
                "Unknown format to log metrics for dataset {}!" % self.args.dataset
            )

        logging.info(stats)
        # ============================================================
        # Save per-round experiment metrics
        # ============================================================

        self._record_metrics(
            round_idx=round_idx,
            train_acc=None,
            train_loss=None,
            test_acc=test_acc,
            test_loss=test_loss,
        )
