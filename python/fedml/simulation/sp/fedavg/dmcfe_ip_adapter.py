from __future__ import annotations

import logging
import math
from collections import OrderedDict
from typing import Mapping, Sequence, Tuple

import torch


logger = logging.getLogger(__name__)

StateDict = Mapping[str, torch.Tensor]
LocalModel = Tuple[int, StateDict]


class ReferenceInnerProductBackend:
    """
    DMCFE-IP 的明文参考后端。

    作用：
    1. 验证定点量化和加权聚合结果；
    2. 确定 FedML 接入位置；
    3. 为后续真实 DMCFE-IP 后端提供统一接口。

    注意：该后端没有密码学安全性，不能用于安全实验结论。
    """

    name = "reference"
    is_secure = False

    def weighted_sum(
        self,
        client_vectors: Sequence[torch.Tensor],
        client_weights: Sequence[int],
        label: str,
    ) -> torch.Tensor:
        if not client_vectors:
            raise ValueError("client_vectors 不能为空")

        if len(client_vectors) != len(client_weights):
            raise ValueError(
                "client_vectors 与 client_weights 数量不一致"
            )

        result = torch.zeros_like(
            client_vectors[0],
            dtype=torch.int64,
            device="cpu",
        )

        for vector, weight in zip(client_vectors, client_weights):
            if vector.dtype != torch.int64:
                raise TypeError(
                    f"DMCFE-IP 输入必须是 int64，实际为 {vector.dtype}"
                )

            if vector.shape != result.shape:
                raise ValueError("客户端模型向量形状不一致")

            if (
                isinstance(weight, bool)
                or not math.isfinite(float(weight))
                or int(weight) != weight
            ):
                raise ValueError("DMCFE-IP weights must be finite integers")

            result.add_(vector * int(weight))

        logger.info(
            "[DMCFE-IP][reference] label=%s, clients=%d, elements=%d",
            label,
            len(client_vectors),
            result.numel(),
        )

        return result


class DMCFEIPAggregator:
    """
    FedAvg 与 DMCFE-IP 之间的聚合适配器。

    当前计算：

        Δw_i = w_i - w_global
        q_i  = round(clip(Δw_i) * scale)

        q_sum = Σ n_i q_i

        w_new = w_global + q_sum / (scale * Σn_i)

    其中：
        n_i 为客户端本地样本数量。
    """

    def __init__(
        self,
        scale: int = 10000,
        clip_bound: float = 8.0,
        backend=None,
    ) -> None:
        if (
            isinstance(scale, bool)
            or not isinstance(scale, int)
            or scale <= 0
        ):
            raise ValueError("scale 必须大于 0")

        if not math.isfinite(float(clip_bound)) or clip_bound <= 0:
            raise ValueError("clip_bound 必须大于 0")

        self.scale = int(scale)
        self.clip_bound = float(clip_bound)
        self.backend = backend or ReferenceInnerProductBackend()

        if not self.backend.is_secure:
            logger.warning(
                "[DMCFE-IP] 当前使用 reference 后端，"
                "仅验证聚合逻辑，不提供密码学保护。"
            )

    def _quantize(
        self,
        tensor: torch.Tensor,
        parameter_name: str,
    ) -> torch.Tensor:
        value = tensor.detach().to(
            device="cpu",
            dtype=torch.float64,
        )
        if not torch.isfinite(value).all():
            raise ValueError(
                f"parameter {parameter_name} contains NaN or infinity"
            )

        clipped_mask = value.abs() > self.clip_bound

        if clipped_mask.any():
            clipped_ratio = (
                clipped_mask.sum().item() / clipped_mask.numel()
            )

            logger.warning(
                "[DMCFE-IP] 参数 %s 发生截断，比例 %.6f",
                parameter_name,
                clipped_ratio,
            )

        value = torch.clamp(
            value,
            min=-self.clip_bound,
            max=self.clip_bound,
        )

        quantized = torch.round(value * self.scale)

        if quantized.numel() > 0:
            max_value = float(quantized.abs().max().item())

            if max_value >= float(1 << 63):
                raise OverflowError(
                    f"参数 {parameter_name} 量化后超过 int64 范围"
                )

        return quantized.to(torch.int64)

    def aggregate(
        self,
        global_params: StateDict,
        local_models: Sequence[LocalModel],
        round_idx: int,
    ) -> OrderedDict:
        if not local_models:
            raise ValueError("没有收到客户端模型")

        if not global_params:
            raise ValueError("global_params must not be empty")

        sample_numbers = []
        for sample_num, _ in local_models:
            if (
                isinstance(sample_num, bool)
                or int(sample_num) != sample_num
                or int(sample_num) <= 0
            ):
                raise ValueError("client sample numbers must be positive integers")
            sample_numbers.append(int(sample_num))

        if any(number <= 0 for number in sample_numbers):
            raise ValueError("客户端样本数量必须大于 0")

        total_samples = sum(sample_numbers)

        expected_keys = set(global_params.keys())
        for client_id, (_, local_params) in enumerate(local_models):
            if set(local_params.keys()) != expected_keys:
                raise ValueError(
                    f"client {client_id} parameter keys do not match global model"
                )

        output = OrderedDict()

        for parameter_name, global_tensor in global_params.items():
            # BatchNorm 的 num_batches_tracked 等整数参数
            # 不适合进行定点加密加权，暂时采用第一个客户端结果。
            if not torch.is_floating_point(global_tensor):
                reference = local_models[0][1][parameter_name].detach().cpu()
                if any(
                    not torch.equal(
                        local_params[parameter_name].detach().cpu(), reference
                    )
                    for _, local_params in local_models[1:]
                ):
                    raise ValueError(
                        f"integer parameter {parameter_name} differs across clients"
                    )
                output[parameter_name] = (
                    reference
                    .detach()
                    .clone()
                    .to(global_tensor.device)
                )
                continue

            global_cpu = global_tensor.detach().to(
                device="cpu",
                dtype=torch.float64,
            )

            quantized_updates = []

            for _, local_params in local_models:
                local_tensor = local_params[parameter_name].detach().to(
                    device="cpu",
                    dtype=torch.float64,
                )

                if local_tensor.shape != global_cpu.shape:
                    raise ValueError(
                        f"参数 {parameter_name} 的客户端形状不一致："
                        f"{local_tensor.shape} != {global_cpu.shape}"
                    )

                # 对更新量加密，而不是直接加密完整模型参数。
                update = local_tensor - global_cpu

                quantized_updates.append(
                    self._quantize(update, parameter_name)
                )

            max_quantized = max(
                int(t.abs().max().item())
                if t.numel() > 0 else 0
                for t in quantized_updates
            )

            int64_limit = torch.iinfo(torch.int64).max

            if (
                max_quantized > 0
                and max_quantized > int64_limit // total_samples
            ):
                raise OverflowError(
                    f"参数 {parameter_name} 加权求和可能导致 int64 溢出；"
                    "请降低 scale 或 clip_bound"
                )

            label = (
                f"fedml-run:{round_idx}:"
                f"{parameter_name}:"
                f"{len(local_models)}"
            )

            # 正式后端将在这里完成：
            # Encrypt -> PartialKeyGen -> KeyCombine -> Decrypt
            weighted_quantized_sum = self.backend.weighted_sum(
                client_vectors=quantized_updates,
                client_weights=sample_numbers,
                label=label,
            )

            average_update = (
                weighted_quantized_sum.to(torch.float64)
                / float(self.scale * total_samples)
            )

            aggregated_tensor = global_cpu + average_update

            output[parameter_name] = aggregated_tensor.to(
                device=global_tensor.device,
                dtype=global_tensor.dtype,
            )

        logger.info(
            "[DMCFE-IP] round=%d 聚合完成，clients=%d，samples=%d，"
            "backend=%s",
            round_idx,
            len(local_models),
            total_samples,
            self.backend.name,
        )

        return output
