import torch


def flatten_delta(
    delta_dict,
    key_order=None,
):
    """
    Flatten parameter tensors into one 1-D vector.

    key_order MUST be shared with unflatten_delta.
    """

    if key_order is None:
        key_order = list(
            delta_dict.keys()
        )

    flat_parts = []

    for key in key_order:

        if key not in delta_dict:
            raise KeyError(
                f"Missing key in delta_dict: {key}"
            )

        tensor = delta_dict[key]

        flat_parts.append(
            tensor.reshape(-1)
        )

    if not flat_parts:
        return torch.empty(0)

    return torch.cat(
        flat_parts,
        dim=0
    )


def unflatten_delta(
    flat_vector,
    reference_dict,
    key_order=None,
):
    """
    Restore 1-D vector into tensors with exactly
    the same layout as reference_dict.
    """

    if key_order is None:
        key_order = list(
            reference_dict.keys()
        )

    result = {}

    offset = 0

    for key in key_order:

        if key not in reference_dict:
            raise KeyError(
                f"Missing reference key: {key}"
            )

        ref = reference_dict[key]

        numel = ref.numel()

        end = offset + numel

        if end > flat_vector.numel():
            raise ValueError(
                f"Vector too short for {key}: "
                f"need {numel} values"
            )

        chunk = flat_vector[
            offset:end
        ]

        result[key] = (
            chunk
            .reshape(ref.shape)
            .to(
                device=ref.device,
                dtype=ref.dtype
            )
        )

        offset = end

    if offset != flat_vector.numel():
        raise ValueError(
            "Flat vector size mismatch: "
            f"used={offset}, "
            f"total={flat_vector.numel()}"
        )

    return result