"""Parameter partitioning utilities for FedRep."""

from __future__ import annotations

import fnmatch
import json
from typing import Iterable, List, Tuple

import torch


def _parse_selectors(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if not isinstance(parsed, list):
                raise ValueError("fedrep_personalized_layers must be a list")
            value = parsed
        else:
            value = stripped.split(",")
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "fedrep_personalized_layers must be a list or comma-separated string"
        )
    selectors = [str(item).strip() for item in value if str(item).strip()]
    return selectors


def _infer_last_parameterized_module(model) -> str:
    candidates = []
    for module_name, module in model.named_modules():
        if any(True for _ in module.parameters(recurse=False)):
            candidates.append(module_name)
    if not candidates:
        raise ValueError("FedRep requires a model with trainable parameters")
    return candidates[-1]


def _matches(key: str, selectors: Iterable[str]) -> bool:
    for selector in selectors:
        if selector == "":
            return True
        if any(character in selector for character in "*?["):
            if fnmatch.fnmatchcase(key, selector):
                return True
        elif key == selector or key.startswith(selector + "."):
            return True
    return False


def resolve_fedrep_parameter_keys(
    model,
    model_state,
    personalized_layers=None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Resolve shared/personalized state and trainable parameter names.

    Selectors are module prefixes, exact state keys, or shell-style patterns.
    With no selectors, the final module containing direct parameters is used as
    the personalized prediction head.
    """

    selectors = _parse_selectors(personalized_layers)
    if not selectors:
        selectors = [_infer_last_parameterized_module(model)]

    state_keys = list(model_state.keys())
    personalized_state_keys = [
        key for key in state_keys if _matches(key, selectors)
    ]
    shared_state_keys = [
        key
        for key in state_keys
        if key not in personalized_state_keys
        and torch.is_floating_point(model_state[key])
    ]

    parameter_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    personalized_parameter_names = [
        name for name in parameter_names if _matches(name, selectors)
    ]
    shared_parameter_names = [
        name for name in parameter_names if name not in personalized_parameter_names
    ]

    unmatched = [
        selector
        for selector in selectors
        if not any(_matches(key, [selector]) for key in state_keys)
    ]
    if unmatched:
        raise ValueError(
            "FedRep personalized layer selector(s) matched no model state: "
            + ", ".join(unmatched)
        )
    if not personalized_parameter_names:
        raise ValueError("FedRep requires at least one trainable personalized parameter")
    if not shared_parameter_names or not shared_state_keys:
        raise ValueError(
            "FedRep requires at least one shared representation parameter; "
            "use a model with separate representation and prediction-head layers"
        )

    return (
        shared_state_keys,
        personalized_state_keys,
        shared_parameter_names,
        personalized_parameter_names,
    )
