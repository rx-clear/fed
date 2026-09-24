from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ============================================================
# DMCFE Round-1 Share Packet
# ============================================================
#
# Paper:
#
#   i || j || [[sk_i^(1)]]_j || [[eta_i]]_j
#
# We need an unambiguous binary encoding in real code.
#
# Format:
#
#   version        : uint8
#   sender_id      : uint64
#   receiver_id    : uint64
#   sk_share_len   : uint16
#   eta_share_len  : uint16
#   sk_share       : variable bytes
#   eta_share      : variable bytes
#
# ">" means fixed big-endian encoding.
# ============================================================

_PROTOCOL_VERSION = 1

_HEADER = struct.Struct(
    ">BQQHH"
)


def _int_to_bytes(
    value: int,
) -> bytes:
    """
    Encode a non-negative integer using the minimum
    number of big-endian bytes.
    """

    value = int(value)

    if value < 0:
        raise ValueError(
            "share values must be non-negative"
        )

    length = max(
        1,
        (value.bit_length() + 7) // 8,
    )

    return value.to_bytes(
        length,
        byteorder="big",
        signed=False,
    )


@dataclass(frozen=True)
class DMCFESharePayload:
    """
    Plaintext carried inside C_{i,j}.

    Paper mapping:

        sender_id
            ->
        i

        receiver_id
            ->
        j

        sk_share
            ->
        [[sk_i^(1)]]_j

        eta_share
            ->
        [[eta_i]]_j
    """

    sender_id: int

    receiver_id: int

    sk_share: int

    eta_share: int

    # ========================================================
    # Serialize
    # ========================================================

    def pack(
        self,
    ) -> bytes:

        sender_id = int(
            self.sender_id
        )

        receiver_id = int(
            self.receiver_id
        )

        # ----------------------------------------------------
        # Client IDs stored as uint64
        # ----------------------------------------------------

        if not (
            0
            <= sender_id
            < (1 << 64)
        ):
            raise ValueError(
                "sender_id must fit in uint64"
            )

        if not (
            0
            <= receiver_id
            < (1 << 64)
        ):
            raise ValueError(
                "receiver_id must fit in uint64"
            )

        # ----------------------------------------------------
        # Convert Shamir shares into bytes
        # ----------------------------------------------------

        sk_bytes = _int_to_bytes(
            self.sk_share
        )

        eta_bytes = _int_to_bytes(
            self.eta_share
        )

        # uint16 lengths
        if len(sk_bytes) > 0xFFFF:

            raise ValueError(
                "sk_share encoding is too large"
            )

        if len(eta_bytes) > 0xFFFF:

            raise ValueError(
                "eta_share encoding is too large"
            )

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        header = _HEADER.pack(
            _PROTOCOL_VERSION,
            sender_id,
            receiver_id,
            len(sk_bytes),
            len(eta_bytes),
        )

        # ----------------------------------------------------
        # Paper:
        #
        # i || j || share_sk || share_eta
        #
        # plus explicit lengths required for parsing.
        # ----------------------------------------------------

        return (
            header
            + sk_bytes
            + eta_bytes
        )

    # ========================================================
    # Deserialize
    # ========================================================

    @classmethod
    def unpack(
        cls,
        data: bytes,
    ) -> "DMCFESharePayload":

        if not isinstance(
            data,
            (bytes, bytearray),
        ):
            raise TypeError(
                "data must be bytes-like"
            )

        data = bytes(data)

        # ----------------------------------------------------
        # Header must exist
        # ----------------------------------------------------

        if len(data) < _HEADER.size:

            raise ValueError(
                "share payload is too short"
            )

        (
            version,
            sender_id,
            receiver_id,
            sk_len,
            eta_len,
        ) = _HEADER.unpack(
            data[: _HEADER.size]
        )

        # ----------------------------------------------------
        # Protocol-version check
        # ----------------------------------------------------

        if version != _PROTOCOL_VERSION:

            raise ValueError(
                "unsupported share payload "
                f"version: {version}"
            )

        expected_len = (
            _HEADER.size
            + sk_len
            + eta_len
        )

        if len(data) != expected_len:

            raise ValueError(
                "invalid share payload length: "
                f"expected {expected_len}, "
                f"got {len(data)}"
            )

        # ----------------------------------------------------
        # Decode sk share
        # ----------------------------------------------------

        offset = _HEADER.size

        sk_bytes = data[
            offset:
            offset + sk_len
        ]

        offset += sk_len

        # ----------------------------------------------------
        # Decode eta share
        # ----------------------------------------------------

        eta_bytes = data[
            offset:
            offset + eta_len
        ]

        sk_share = int.from_bytes(
            sk_bytes,
            byteorder="big",
            signed=False,
        )

        eta_share = int.from_bytes(
            eta_bytes,
            byteorder="big",
            signed=False,
        )

        return cls(
            sender_id=sender_id,
            receiver_id=receiver_id,
            sk_share=sk_share,
            eta_share=eta_share,
        )


# ============================================================
# Symmetric Encryption
# ============================================================

class SymmetricEncryption:
    """
    Symmetric encryption primitive used by DMCFE-IP.

    Paper mapping:

        SE.Enc(K, x)
            ->
        encrypt(key, plaintext)

        SE.Dec(K, ct)
            ->
        decrypt(key, ciphertext)

    Concrete implementation:

        AES-GCM
    """

    # AES-GCM recommended nonce size
    NONCE_SIZE = 12

    # AES supports:
    #
    # 128 bits
    # 192 bits
    # 256 bits

    VALID_AES_KEY_SIZES = (
        16,
        24,
        32,
    )

    # ========================================================
    # Key validation
    # ========================================================

    @staticmethod
    def _validate_key(
        key: bytes,
    ) -> bytes:

        if not isinstance(
            key,
            (bytes, bytearray),
        ):
            raise TypeError(
                "key must be bytes-like"
            )

        key = bytes(key)

        if len(key) not in (
            SymmetricEncryption
            .VALID_AES_KEY_SIZES
        ):

            raise ValueError(
                "AES-GCM key must be "
                "16, 24, or 32 bytes "
                f"(got {len(key)} bytes)"
            )

        return key

    # ========================================================
    # SE.Enc
    # ========================================================

    def encrypt(
        self,
        key: bytes,
        plaintext: bytes,
        associated_data: Optional[
            bytes
        ] = None,
    ) -> bytes:

        key = self._validate_key(
            key
        )

        if not isinstance(
            plaintext,
            (bytes, bytearray),
        ):
            raise TypeError(
                "plaintext must be bytes-like"
            )

        # ----------------------------------------------------
        # Fresh nonce for every encryption.
        #
        # nonce does NOT need to be secret.
        # ----------------------------------------------------

        nonce = os.urandom(
            self.NONCE_SIZE
        )

        aesgcm = AESGCM(
            key
        )

        # cryptography returns:
        #
        # ciphertext || authentication_tag
        #
        ciphertext = aesgcm.encrypt(
            nonce,
            bytes(plaintext),
            associated_data,
        )

        # ----------------------------------------------------
        # Our transport packet:
        #
        # nonce || ciphertext || tag
        #
        # Server can forward this entire byte string.
        # ----------------------------------------------------

        return (
            nonce
            + ciphertext
        )

    # ========================================================
    # SE.Dec
    # ========================================================

    def decrypt(
        self,
        key: bytes,
        packet: bytes,
        associated_data: Optional[
            bytes
        ] = None,
    ) -> bytes:

        key = self._validate_key(
            key
        )

        if not isinstance(
            packet,
            (bytes, bytearray),
        ):
            raise TypeError(
                "packet must be bytes-like"
            )

        packet = bytes(
            packet
        )

        if len(packet) <= self.NONCE_SIZE:

            raise ValueError(
                "encrypted packet is too short"
            )

        nonce = packet[
            : self.NONCE_SIZE
        ]

        ciphertext = packet[
            self.NONCE_SIZE :
        ]

        aesgcm = AESGCM(
            key
        )

        # If:
        #
        # - key is wrong
        # - nonce is wrong
        # - ciphertext is modified
        #
        # AESGCM.decrypt() raises InvalidTag.
        #
        return aesgcm.decrypt(
            nonce,
            ciphertext,
            associated_data,
        )

    # ========================================================
    # DMCFE Round-1 helper
    # ========================================================

    def encrypt_share_packet(
        self,
        key: bytes,
        sender_id: int,
        receiver_id: int,
        sk_share: int,
        eta_share: int,
    ) -> bytes:
        """
        Implements:

        C_{i,j} =
            SE.Enc(
                k^(2)_{i,j},
                i || j ||
                [[sk_i^(1)]]_j ||
                [[eta_i]]_j
            )
        """

        payload = DMCFESharePayload(
            sender_id=sender_id,
            receiver_id=receiver_id,
            sk_share=sk_share,
            eta_share=eta_share,
        )

        return self.encrypt(
            key=key,
            plaintext=payload.pack(),
        )

    # ========================================================
    # DMCFE Round-3 helper
    # ========================================================

    def decrypt_share_packet(
        self,
        key: bytes,
        packet: bytes,
    ) -> DMCFESharePayload:
        """
        Client j decrypts C_{i,j}.
        """

        plaintext = self.decrypt(
            key=key,
            packet=packet,
        )

        return DMCFESharePayload.unpack(
            plaintext
        )