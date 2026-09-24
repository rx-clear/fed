"""Round-bound weight proposals for the DMCFE aggregation protocol.

The paper authenticates each client's aggregation weight before the server
derives the functional decryption key.  This module keeps that idea separate
from the training code and binds a signature to the round and participant set,
which also prevents replaying a valid proposal in another round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def _payload(round_idx: int, participant_ids: Sequence[int], client_id: int, weight: int) -> bytes:
    ids = tuple(sorted(int(value) for value in participant_ids))
    if len(ids) != len(set(ids)):
        raise ValueError("participant_ids must be unique")
    if int(round_idx) < 0 or int(round_idx) != round_idx:
        raise ValueError("round_idx must be a non-negative integer")
    if int(client_id) not in ids:
        raise ValueError("client_id must belong to participant_ids")
    # Length-prefixed decimal encoding is deterministic and unambiguous.
    fields = [b"DMCFE-IP/weight-proposal/v1", str(int(round_idx)).encode()]
    fields.append(str(len(ids)).encode())
    fields.extend(str(value).encode() for value in ids)
    fields.extend((str(int(client_id)).encode(), str(int(weight)).encode()))
    return b"|".join(fields)


@dataclass(frozen=True)
class WeightProposal:
    round_idx: int
    participant_ids: Tuple[int, ...]
    client_id: int
    weight: int
    signature: bytes


class WeightSigner:
    """Ed25519 signer used for one client's weight proposals."""

    def __init__(self, private_key: Optional[Ed25519PrivateKey] = None):
        self._private_key = private_key or Ed25519PrivateKey.generate()

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def propose(
        self,
        round_idx: int,
        participant_ids: Sequence[int],
        client_id: int,
        weight: int,
    ) -> WeightProposal:
        weight = int(weight)
        if weight <= 0:
            raise ValueError("DMCFE weights must be positive integers")
        ids = tuple(sorted(int(value) for value in participant_ids))
        payload = _payload(round_idx, ids, client_id, weight)
        return WeightProposal(
            int(round_idx), ids, int(client_id), weight,
            self._private_key.sign(payload),
        )


def verify_weight_proposals(
    proposals: Sequence[WeightProposal],
    public_keys: Mapping[int, Union[bytes, Ed25519PublicKey]],
    round_idx: int,
    participant_ids: Sequence[int],
) -> Tuple[int, ...]:
    """Verify a complete set of proposals and return weights in ``participant_ids`` order."""
    ids = tuple(sorted(int(value) for value in participant_ids))
    if len(ids) != len(set(ids)):
        raise ValueError("participant_ids must be unique")
    if len(proposals) != len(ids):
        raise ValueError("one signed weight proposal is required per participant")
    by_client = {int(proposal.client_id): proposal for proposal in proposals}
    if set(by_client) != set(ids):
        raise ValueError("signed proposals do not match participant_ids")
    for client_id in ids:
        proposal = by_client[client_id]
        if proposal.round_idx != int(round_idx):
            raise ValueError("signed weight proposal belongs to another round")
        if proposal.participant_ids != ids:
            raise ValueError("signed participant set mismatch")
        key = public_keys.get(client_id)
        if key is None:
            raise ValueError(f"missing public key for client {client_id}")
        public_key = key if isinstance(key, Ed25519PublicKey) else Ed25519PublicKey.from_public_bytes(bytes(key))
        try:
            public_key.verify(proposal.signature, _payload(
                proposal.round_idx, proposal.participant_ids,
                proposal.client_id, proposal.weight,
            ))
        except Exception as exc:
            raise ValueError(f"invalid weight signature for client {client_id}") from exc
        if proposal.weight <= 0:
            raise ValueError("DMCFE weights must be positive integers")
    return tuple(by_client[client_id].weight for client_id in participant_ids)
