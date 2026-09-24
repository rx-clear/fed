from __future__ import annotations

import hashlib
import logging
import secrets

from cryptography.hazmat.primitives import (
    hashes,
    serialization,
)

from cryptography.hazmat.primitives.asymmetric import (
    x25519,
)

from cryptography.hazmat.primitives.kdf.hkdf import (
    HKDF,
)

from .secret_sharing import (
    ShamirSecretSharing,
)

from .symmetric_encryption import (
    SymmetricEncryption,
)


# ============================================================
# DMCFE Dropout / Round-3 Scalar Smoke
# ============================================================
#
# Test scenario:
#
# U1 = {0,1,2,3,4}
#
# Round 1:
#   everyone online
#
# U2 = {0,1,2,3,4}
#
# Round 2:
#   client 4 drops before submitting dk / ciphertext
#
# U3 = {0,1,2,3}
#
# Round 3:
#   client 3 drops before sending recovery material
#
# U4 = {0,1,2}
#
#
# Therefore:
#
# U2 - U3 = {4}
#
#   -> recover sk_4^(1)
#   -> remove unmatched pairwise masks
#
#
# U3 - U4 = {3}
#
#   -> recover eta_3
#   -> remove PRG(eta_3)
#
#
# Finally verify:
#
# recovered dk_y'
#
#       ==
#
# sum(s_i for i in U3)
#
# because this scalar smoke uses y_i = 1.
# ============================================================


K1_INFO = b"DMCFE-IP/k1/v1"
K2_INFO = b"DMCFE-IP/k2/v1"

PRG_MODULUS = 1_000_003


# ============================================================
# Key Agreement
# ============================================================

def _derive_pairwise_key(
    private_key: x25519.X25519PrivateKey,
    peer_public_key: x25519.X25519PublicKey,
    info: bytes,
) -> bytes:

    shared_secret = private_key.exchange(
        peer_public_key
    )

    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info,
    ).derive(
        shared_secret
    )


def _private_key_to_raw(
    private_key: x25519.X25519PrivateKey,
) -> bytes:

    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=(
            serialization.NoEncryption()
        ),
    )


# ============================================================
# Correctness-oriented scalar PRG
# ============================================================

def _seed_to_bytes(
    seed,
) -> bytes:

    if isinstance(
        seed,
        (bytes, bytearray),
    ):
        return bytes(seed)

    if isinstance(seed, int):

        if seed < 0:
            raise ValueError(
                "PRG integer seed must be non-negative"
            )

        length = max(
            1,
            (seed.bit_length() + 7) // 8,
        )

        return seed.to_bytes(
            length,
            byteorder="big",
            signed=False,
        )

    raise TypeError(
        "unsupported PRG seed type"
    )


def _prg_scalar(
    seed,
    domain: bytes,
) -> int:
    """
    Deterministic scalar PRG for protocol correctness testing.

    This is intentionally separated from the production PRG
    implementation.

    Output is centered around zero so logs remain readable.
    """

    seed_bytes = _seed_to_bytes(
        seed
    )

    digest = hashlib.sha256(
        domain
        + b"|"
        + seed_bytes
    ).digest()

    value = (
        int.from_bytes(
            digest,
            byteorder="big",
            signed=False,
        )
        % PRG_MODULUS
    )

    if value > (
        PRG_MODULUS // 2
    ):
        value -= PRG_MODULUS

    return value


def _pair_prg(
    key: bytes,
) -> int:

    return _prg_scalar(
        key,
        b"DMCFE/pairwise-mask/v1",
    )


def _eta_prg(
    eta: int,
) -> int:

    return _prg_scalar(
        eta,
        b"DMCFE/eta-mask/v1",
    )


# ============================================================
# Main smoke test
# ============================================================

def run_dmcfe_dropout_smoke():

    logging.info("")
    logging.info(
        "========== DMCFE Dropout Round-3 Smoke =========="
    )

    # ========================================================
    # Client sets
    # ========================================================

    U1 = {
        0,
        1,
        2,
        3,
        4,
    }

    U2 = {
        0,
        1,
        2,
        3,
        4,
    }

    # client 4 drops before Round-2 submission
    U3 = {
        0,
        1,
        2,
        3,
    }

    # client 3 drops before Round-3 response
    U4 = {
        0,
        1,
        2,
    }

    threshold = 3

    # Lepcat / scalar correctness test:
    #
    # y_i = 1
    # common mask multiplier = 1

    common_y = 1

    # --------------------------------------------------------
    # Protocol-set validation
    # --------------------------------------------------------

    assert U2.issubset(U1)

    assert U3.issubset(U2)

    assert U4.issubset(U3)

    if len(U4) < threshold:

        raise RuntimeError(
            "Round 3 must abort: "
            "|U4| < threshold"
        )

    early_dropout = (
        U2 - U3
    )

    late_dropout = (
        U3 - U4
    )

    # Expected:
    #
    # early_dropout = {4}
    # late_dropout  = {3}

    # ========================================================
    # Initialize primitives
    # ========================================================

    ss = ShamirSecretSharing()

    se = SymmetricEncryption()

    # ========================================================
    # ROUND 0
    #
    # Generate two X25519 keypairs per client.
    # Generate MCFE scalar s_i.
    # Generate eta_i.
    # ========================================================

    client_state = {}

    for i in sorted(U1):

        sk1 = (
            x25519
            .X25519PrivateKey
            .generate()
        )

        sk2 = (
            x25519
            .X25519PrivateKey
            .generate()
        )

        client_state[i] = {

            "sk1": sk1,

            "pk1": sk1.public_key(),

            "sk2": sk2,

            "pk2": sk2.public_key(),

            # small scalar MCFE secret for easy debugging
            "s": (
                secrets.randbelow(101)
                - 50
            ),

            "eta": secrets.randbits(
                128
            ),

            "k1": {},

            "k2": {},
        }

    # ========================================================
    # ROUND 1
    #
    # Pairwise key agreement.
    # ========================================================

    for i in sorted(U1):

        for j in sorted(U1):

            if i == j:
                continue

            client_state[i]["k1"][j] = (
                _derive_pairwise_key(
                    private_key=(
                        client_state[i]["sk1"]
                    ),
                    peer_public_key=(
                        client_state[j]["pk1"]
                    ),
                    info=K1_INFO,
                )
            )

            client_state[i]["k2"][j] = (
                _derive_pairwise_key(
                    private_key=(
                        client_state[i]["sk2"]
                    ),
                    peer_public_key=(
                        client_state[j]["pk2"]
                    ),
                    info=K2_INFO,
                )
            )

    # ========================================================
    # Verify pairwise KA
    # ========================================================

    pairwise_ka_correct = True

    for i in sorted(U1):

        for j in sorted(U1):

            if i >= j:
                continue

            if (
                client_state[i]["k1"][j]
                !=
                client_state[j]["k1"][i]
            ):
                pairwise_ka_correct = False

            if (
                client_state[i]["k2"][j]
                !=
                client_state[j]["k2"][i]
            ):
                pairwise_ka_correct = False

    # ========================================================
    # ROUND 1
    #
    # Shamir share:
    #
    # sk_i^(1)
    # eta_i
    #
    # Encrypt:
    #
    # C_i,j = SE.Enc(k2_i,j, ...)
    #
    # Server only forwards packets.
    # ========================================================

    server_packets = {}

    for i in sorted(U1):

        recipients = [
            j
            for j in sorted(U1)
            if j != i
        ]

        sk1_raw = _private_key_to_raw(
            client_state[i]["sk1"]
        )

        sk_shares = ss.share_bytes(
            threshold=threshold,
            participant_ids=recipients,
            secret=sk1_raw,
        )

        eta_shares = ss.share(
            threshold=threshold,
            participant_ids=recipients,
            secret=client_state[i]["eta"],
        )

        for j in recipients:

            packet = (
                se.encrypt_share_packet(
                    key=(
                        client_state[i]
                        ["k2"][j]
                    ),
                    sender_id=i,
                    receiver_id=j,
                    sk_share=(
                        sk_shares[j]
                    ),
                    eta_share=(
                        eta_shares[j]
                    ),
                )
            )

            server_packets[
                (i, j)
            ] = packet

    # ========================================================
    # Server forwards encrypted packets.
    #
    # client_inbox[j][i] = C_i,j
    # ========================================================

    client_inbox = {
        j: {}
        for j in U1
    }

    for (
        sender,
        receiver,
    ), packet in server_packets.items():

        client_inbox[
            receiver
        ][sender] = packet

    # ========================================================
    # ROUND 2
    #
    # Clients in U3 actually submit:
    #
    # dk_i,y
    #
    # Client 4 is already gone.
    #
    #
    # dk_i,y =
    #
    # s_i
    # +
    # PRG(eta_i)
    # +
    # pairwise_mask_i
    #
    #
    # pairwise masks are computed over U2,
    # NOT U3.
    #
    # This is exactly why client 4 leaves an unmatched mask.
    # ========================================================

    partial_dk = {}

    client_pair_masks = {}

    for i in sorted(U3):

        pair_mask = 0

        for j in sorted(U2):

            if i == j:
                continue

            mask = _pair_prg(
                client_state[i]
                ["k1"][j]
            )

            if i < j:

                pair_mask += mask

            else:

                pair_mask -= mask

        client_pair_masks[i] = (
            pair_mask
        )

        partial_dk[i] = (
            client_state[i]["s"]
            +
            _eta_prg(
                client_state[i]["eta"]
            )
            +
            pair_mask * common_y
        )

    # ========================================================
    # Important diagnostic:
    #
    # Pairwise masks INSIDE U3 cancel.
    #
    # Masks involving client 4 do NOT cancel because
    # client 4 submitted no dk.
    # ========================================================

    total_pair_mask_u3 = sum(
        client_pair_masks.values()
    )

    # ========================================================
    # ROUND 3
    #
    # U4 clients respond.
    #
    # Each responder decrypts the C_j,i received
    # during Round 1.
    #
    # It sends:
    #
    # 1. own eta_i
    #
    # 2. eta shares needed for U3 - U4
    #
    # 3. sk1 shares needed for U2 - U3
    # ========================================================

    round3_messages = {}

    round3_packet_decryption_correct = True

    for i in sorted(U4):

        decrypted_from_peers = {}

        for j in sorted(
            U2 - {i}
        ):

            packet = (
                client_inbox[i][j]
            )

            recovered = (
                se.decrypt_share_packet(
                    key=(
                        client_state[i]
                        ["k2"][j]
                    ),
                    packet=packet,
                )
            )

            if (
                recovered.sender_id != j
                or
                recovered.receiver_id != i
            ):

                round3_packet_decryption_correct = (
                    False
                )

            decrypted_from_peers[
                j
            ] = recovered

        # ----------------------------------------------------
        # Build Round-3 response
        # ----------------------------------------------------

        eta_shares_for_server = {}

        for dropped_id in sorted(
            late_dropout
        ):

            eta_shares_for_server[
                dropped_id
            ] = (
                decrypted_from_peers[
                    dropped_id
                ].eta_share
            )

        sk_shares_for_server = {}

        for dropped_id in sorted(
            early_dropout
        ):

            sk_shares_for_server[
                dropped_id
            ] = (
                decrypted_from_peers[
                    dropped_id
                ].sk_share
            )

        round3_messages[i] = {

            # alive Round-3 client directly reveals eta_i
            "eta_self": (
                client_state[i]["eta"]
            ),

            "eta_shares": (
                eta_shares_for_server
            ),

            "sk_shares": (
                sk_shares_for_server
            ),
        }

    # ========================================================
    # SERVER ROUND 3
    #
    # Recover sk_i^(1)
    #
    # for i in U2 - U3
    # ========================================================

    recovered_sk1 = {}

    sk_recovery_correct = True

    for dropped_id in sorted(
        early_dropout
    ):

        available_shares = {
            responder: (
                round3_messages[
                    responder
                ]["sk_shares"][
                    dropped_id
                ]
            )
            for responder in U4
        }

        raw = ss.combine_bytes(
            threshold=threshold,
            shares=available_shares,
            length=32,
        )

        original_raw = (
            _private_key_to_raw(
                client_state[
                    dropped_id
                ]["sk1"]
            )
        )

        if raw != original_raw:

            sk_recovery_correct = (
                False
            )

        recovered_sk1[
            dropped_id
        ] = (
            x25519
            .X25519PrivateKey
            .from_private_bytes(
                raw
            )
        )

    # ========================================================
    # Recover eta_i
    #
    # for i in U3 - U4
    # ========================================================

    recovered_eta = {}

    eta_recovery_correct = True

    for dropped_id in sorted(
        late_dropout
    ):

        available_shares = {
            responder: (
                round3_messages[
                    responder
                ]["eta_shares"][
                    dropped_id
                ]
            )
            for responder in U4
        }

        eta_value = ss.combine(
            threshold=threshold,
            shares=available_shares,
        )

        recovered_eta[
            dropped_id
        ] = eta_value

        if (
            eta_value
            !=
            client_state[
                dropped_id
            ]["eta"]
        ):

            eta_recovery_correct = (
                False
            )

    # ========================================================
    # Server now obtains eta_i for ALL i in U3.
    #
    # U4:
    #     eta_i sent directly.
    #
    # U3-U4:
    #     reconstructed using Shamir.
    # ========================================================

    server_eta = {}

    for i in sorted(U4):

        server_eta[i] = (
            round3_messages[i]
            ["eta_self"]
        )

    for i in sorted(
        late_dropout
    ):

        server_eta[i] = (
            recovered_eta[i]
        )

    eta_set_correct = (
        set(server_eta.keys())
        ==
        set(U3)
    )

    # ========================================================
    # Recover residual pairwise mask caused by:
    #
    # U2 - U3
    #
    # Here:
    #
    # client 4
    #
    # Server has recovered sk_4^(1).
    #
    # Therefore server can recreate:
    #
    # k_4,i^(1)
    #
    # using public key pk_i^(1).
    # ========================================================

    residual_pair_mask = 0

    recovered_pair_keys_correct = True

    for dropped_id in sorted(
        early_dropout
    ):

        dropped_sk1 = (
            recovered_sk1[
                dropped_id
            ]
        )

        for i in sorted(U3):

            reconstructed_key = (
                _derive_pairwise_key(
                    private_key=(
                        dropped_sk1
                    ),
                    peer_public_key=(
                        client_state[i]
                        ["pk1"]
                    ),
                    info=K1_INFO,
                )
            )

            reference_key = (
                client_state[i]
                ["k1"][
                    dropped_id
                ]
            )

            if (
                reconstructed_key
                !=
                reference_key
            ):

                recovered_pair_keys_correct = (
                    False
                )

            mask = _pair_prg(
                reconstructed_key
            )

            # Fig. 4 sign convention:
            #
            # + PRG(k_i,j) if i < j
            #
            # - PRG(k_i,j) if i > j

            if i < dropped_id:

                residual_pair_mask += (
                    mask
                )

            else:

                residual_pair_mask -= (
                    mask
                )

    # ========================================================
    # Diagnostic:
    #
    # Total uncancelled pairwise contribution observed in U3
    #
    # should exactly equal the residual contribution
    # reconstructed using dropped clients' sk1.
    # ========================================================

    residual_mask_correct = (
        total_pair_mask_u3
        ==
        residual_pair_mask
    )

    # ========================================================
    # Compute dk_y'
    #
    # Paper:
    #
    # dk_y'
    #
    # =
    #
    # sum_{i in U3} dk_i,y
    #
    # -
    #
    # sum_{i in U3} PRG(eta_i)
    #
    # -
    #
    # residual pairwise masks
    # ========================================================

    sum_partial_dk = sum(
        partial_dk.values()
    )

    eta_mask_sum = sum(
        _eta_prg(
            server_eta[i]
        )
        for i in U3
    )

    recovered_dk_y_prime = (
        sum_partial_dk
        -
        eta_mask_sum
        -
        residual_pair_mask
        * common_y
    )

    # ========================================================
    # Reference result
    #
    # y_i = 1 for surviving Round-2 submitters U3
    #
    # therefore:
    #
    # dk_y' = sum s_i
    # ========================================================

    reference_dk_y_prime = sum(
        client_state[i]["s"]
        for i in U3
    )

    dk_correct = (
        recovered_dk_y_prime
        ==
        reference_dk_y_prime
    )

    # ========================================================
    # Additional useful diagnostic:
    #
    # If we only remove eta masks but do NOT repair
    # client-4's residual pairwise masks, result should
    # normally be wrong.
    # ========================================================

    naive_dk_without_dropout_fix = (
        sum_partial_dk
        -
        eta_mask_sum
    )

    dropout_fix_was_needed = (
        naive_dk_without_dropout_fix
        != reference_dk_y_prime
    )

    # ========================================================
    # Overall correctness
    # ========================================================

    overall_correct = all(
        [
            pairwise_ka_correct,
            round3_packet_decryption_correct,
            sk_recovery_correct,
            eta_recovery_correct,
            eta_set_correct,
            recovered_pair_keys_correct,
            residual_mask_correct,
            dk_correct,
        ]
    )

    # ========================================================
    # Logging
    # ========================================================

    logging.info(
        "U1: %s",
        sorted(U1),
    )

    logging.info(
        "U2: %s",
        sorted(U2),
    )

    logging.info(
        "U3: %s",
        sorted(U3),
    )

    logging.info(
        "U4: %s",
        sorted(U4),
    )

    logging.info(
        "threshold: %s",
        threshold,
    )

    logging.info(
        "U2 - U3 (early dropout): %s",
        sorted(early_dropout),
    )

    logging.info(
        "U3 - U4 (late dropout): %s",
        sorted(late_dropout),
    )

    logging.info(
        "pairwise KA correctness: %s",
        pairwise_ka_correct,
    )

    logging.info(
        "Round-3 packet decryption correctness: %s",
        round3_packet_decryption_correct,
    )

    logging.info(
        "sk^(1) reconstruction correctness: %s",
        sk_recovery_correct,
    )

    logging.info(
        "eta reconstruction correctness: %s",
        eta_recovery_correct,
    )

    logging.info(
        "server eta set correctness: %s",
        eta_set_correct,
    )

    logging.info(
        "recovered dropout pair keys correctness: %s",
        recovered_pair_keys_correct,
    )

    logging.info(
        "total pair mask in U3: %s",
        total_pair_mask_u3,
    )

    logging.info(
        "reconstructed residual pair mask: %s",
        residual_pair_mask,
    )

    logging.info(
        "residual mask correctness: %s",
        residual_mask_correct,
    )

    logging.info(
        "sum partial dk: %s",
        sum_partial_dk,
    )

    logging.info(
        "eta mask sum: %s",
        eta_mask_sum,
    )

    logging.info(
        "naive dk without dropout repair: %s",
        naive_dk_without_dropout_fix,
    )

    logging.info(
        "recovered dk_y': %s",
        recovered_dk_y_prime,
    )

    logging.info(
        "reference dk_y': %s",
        reference_dk_y_prime,
    )

    logging.info(
        "dropout correction was needed: %s",
        dropout_fix_was_needed,
    )

    logging.info(
        "DMCFE dropout dk correctness: %s",
        dk_correct,
    )

    logging.info(
        "DMCFE dropout Round-3 correctness: %s",
        overall_correct,
    )

    logging.info(
        "=================================================="
    )

    logging.info("")

    # ========================================================
    # Assertions
    # ========================================================

    assert pairwise_ka_correct

    assert round3_packet_decryption_correct

    assert sk_recovery_correct

    assert eta_recovery_correct

    assert eta_set_correct

    assert recovered_pair_keys_correct

    assert residual_mask_correct

    assert dk_correct

    assert overall_correct

    return True