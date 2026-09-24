"""Fixed-point signed plaintext packing from the DMCFE-IP construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class PackedLayout:
    value_bits: int
    padding_bits: int
    slots: int

    @property
    def slot_bits(self) -> int:
        return self.value_bits + self.padding_bits

    @property
    def offset(self) -> int:
        return 1 << (self.value_bits - 1)

    @classmethod
    def for_clients(cls, value_bits: int, number_of_clients: int, modulus_bits: int) -> "PackedLayout":
        if value_bits < 2 or number_of_clients < 1 or modulus_bits < 2:
            raise ValueError("value_bits >= 2, number_of_clients >= 1 and modulus_bits >= 2 are required")
        # One sign/carry guard bit plus enough carry bits for a client sum.
        padding_bits = max(1, math.ceil(math.log2(number_of_clients + 1)) + 1)
        slots = (int(modulus_bits) - 1) // (value_bits + padding_bits)
        if slots < 1:
            raise ValueError("modulus_bits is too small for one packed value")
        return cls(int(value_bits), int(padding_bits), int(slots))


def pack_signed(values: Sequence[int], layout: PackedLayout) -> int:
    if len(values) > layout.slots:
        raise ValueError("too many values for the packed layout")
    # Encode unused slots as signed zero as well.  This keeps a packed
    # plaintext decodable without carrying a separate length field and makes
    # sums of differently padded vectors well-defined.
    result = sum(layout.offset << (index * layout.slot_bits) for index in range(layout.slots))
    slot_mask = (1 << layout.slot_bits) - 1
    for index, value in enumerate(values):
        value = int(value)
        lower = -(1 << (layout.value_bits - 1))
        upper = (1 << (layout.value_bits - 1)) - 1
        if not lower <= value <= upper:
            raise ValueError(f"value {value} does not fit in {layout.value_bits} signed bits")
        shift = index * layout.slot_bits
        result &= ~(slot_mask << shift)
        result |= (value + layout.offset) << shift
    return result


def unpack_signed(packed: int, layout: PackedLayout, count: Optional[int] = None) -> List[int]:
    if int(packed) < 0:
        raise ValueError("packed plaintext must be non-negative")
    count = layout.slots if count is None else int(count)
    if count < 0 or count > layout.slots:
        raise ValueError("count exceeds the packed layout")
    mask = (1 << layout.slot_bits) - 1
    return [((int(packed) >> (i * layout.slot_bits)) & mask) - layout.offset for i in range(count)]


def unpack_aggregate(packed: int, layout: PackedLayout, number_of_clients: int, count: Optional[int] = None) -> List[int]:
    """Decode a sum of packed values, removing the per-client non-negative offset."""
    if number_of_clients < 1:
        raise ValueError("number_of_clients must be positive")
    encoded = unpack_signed(packed, layout, count)
    return [value - (number_of_clients - 1) * layout.offset for value in encoded]


def quantize_fixed_point(values: Sequence[float], precision: int = 14) -> List[int]:
    if precision < 0:
        raise ValueError("precision must be non-negative")
    scale = 1 << int(precision)
    return [int(round(float(value) * scale)) for value in values]
