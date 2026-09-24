from __future__ import annotations

import logging
import secrets
from .mcfe import MCFE

from cryptography.hazmat.primitives.asymmetric import (
    x25519,
)

from .secret_sharing import (
    ShamirSecretSharing,
)

from .symmetric_encryption import (
    SymmetricEncryption,
)

# Reuse the helpers that were already verified in dropout_smoke.py
from .dropout_smoke import (
    K1_INFO,
    K2_INFO,
    _derive_pairwise_key,
    _private_key_to_raw,
    _pair_prg,
    _eta_prg,
)


# ============================================================
# DMCFE Dropout End-to-End Scalar Smoke
# ============================================================
#
# Goal:
#
#   Round 0
#       ↓
#   Round 1 KeySharing
#       ↓
#   Round 2 MCFE ciphertext + partial dk
#       ↓
#   Early dropout
#       ↓
#   Round 3 secret reconstruction
#       ↓
#   recover dk_y'
#       ↓
#   MCFE decrypt
#       ↓
#
#   decrypted result
#       ==
#   plaintext sum over U3
#
#
# Fixed test:
#
# client 0 ->  10
# client 1 ->  20
# client 2 ->  -5
# client 3 -> 100
# client 4 ->  50
#
# U3 = {0,1,2,3}
#
# Therefore expected output:
#
# 10 + 20 - 5 + 100 = 125
#
# Client 4's value 50 MUST NOT appear.
# ============================================================


# ============================================================
# Correctness-oriented DCR parameters
# ============================================================
#
# Small parameters are deliberate here.
#
# This is a smoke test, NOT the final security parameter.
#
# Later:
#     replace with mcfe.py Setup / 2048-bit N.
# ============================================================



# ============================================================
# Minimal DCR-MCFE scalar encryption
# ============================================================




# ============================================================
# Main test
# ============================================================

def run_dmcfe_dropout_e2e_smoke():

    logging.info("")
    logging.info(
        "========== DMCFE Dropout End-to-End Scalar Smoke =========="
    )

    # ========================================================
    # 1. Protocol client sets
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

    # client 4 drops BEFORE Round-2 submission
    U3 = {
        0,
        1,
        2,
        3,
    }

    # client 3 drops AFTER Round-2 submission,
    # before Round-3 recovery response
    U4 = {
        0,
        1,
        2,
    }

    threshold = 3

    assert U2.issubset(U1)
    assert U3.issubset(U2)
    assert U4.issubset(U3)

    if len(U4) < threshold:

        raise RuntimeError(
            "|U4| < threshold"
        )

    early_dropout = (
        U2 - U3
    )

    late_dropout = (
        U3 - U4
    )

    # ========================================================
    # 2. Fixed scalar plaintexts
    # ========================================================

    client_values = {

        0: 10,

        1: 20,

        2: -5,

        3: 100,

        # client 4 will drop before ciphertext upload
        4: 50,
    }

    expected_plaintext = sum(
        client_values[i]
        for i in U3
    )

    # Must be:
    #
    # 10 + 20 - 5 + 100 = 125

    assert (
        expected_plaintext
        == 125
    )

    # ========================================================
    # 3. Weight vector
    #
    # Initial:
    #
    # y = [1,1,1,1,1]
    #
    # Final:
    #
    # client 4 early-dropped
    #
    # y' = [1,1,1,1,0]
    # ========================================================

    initial_y = {
        i: 1
        for i in U2
    }

    final_y = dict(
        initial_y
    )

    for i in early_dropout:

        final_y[i] = 0

    expected_final_y = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 0,
    }

    final_weight_correct = (
        final_y
        ==
        expected_final_y
    )

    # ========================================================
    # 4. Initialize crypto
    # ========================================================

    # --------------------------------------------------------
    # REAL MCFE implementation
    #
    # MCFE.__init__() already calls setup().
    #
    # IMPORTANT:
    # All clients in this smoke test MUST share the same
    # MCFE public parameters N and N^2.
    # --------------------------------------------------------

    mcfe = MCFE(
        modulus_bits=256
    )

    ss = ShamirSecretSharing()

    se = SymmetricEncryption()

    label = (
        b"DMCFE-IP/"
        b"dropout-e2e/"
        b"round-0"
    )

    # ========================================================
    # 5. ROUND 0
    #
    # Generate:
    #
    # sk_i^(1), pk_i^(1)
    # sk_i^(2), pk_i^(2)
    # MCFE secret s_i
    # eta_i
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

            "pk1": (
                sk1.public_key()
            ),

            "sk2": sk2,

            "pk2": (
                sk2.public_key()
            ),

            # MCFE secret.
            #
            # Positive for simple exponent handling
            # in this bridge smoke.

            "s": mcfe.keygen(),

            "eta": (
                secrets.randbits(
                    128
                )
            ),

            "k1": {},

            "k2": {},
        }

    # ========================================================
    # 6. ROUND 1 — Pairwise KA
    # ========================================================

    for i in sorted(U1):

        for j in sorted(U1):

            if i == j:
                continue

            client_state[
                i
            ]["k1"][j] = (
                _derive_pairwise_key(
                    private_key=(
                        client_state[i][
                            "sk1"
                        ]
                    ),
                    peer_public_key=(
                        client_state[j][
                            "pk1"
                        ]
                    ),
                    info=K1_INFO,
                )
            )

            client_state[
                i
            ]["k2"][j] = (
                _derive_pairwise_key(
                    private_key=(
                        client_state[i][
                            "sk2"
                        ]
                    ),
                    peer_public_key=(
                        client_state[j][
                            "pk2"
                        ]
                    ),
                    info=K2_INFO,
                )
            )

    # ========================================================
    # 7. ROUND 1 — Secret Sharing + C_i,j
    # ========================================================

    server_share_packets = {}

    for i in sorted(U1):

        recipients = [
            j
            for j in sorted(U1)
            if j != i
        ]

        sk1_raw = (
            _private_key_to_raw(
                client_state[i][
                    "sk1"
                ]
            )
        )

        sk_shares = (
            ss.share_bytes(
                threshold=threshold,
                participant_ids=recipients,
                secret=sk1_raw,
            )
        )

        eta_shares = (
            ss.share(
                threshold=threshold,
                participant_ids=recipients,
                secret=(
                    client_state[i][
                        "eta"
                    ]
                ),
            )
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

            server_share_packets[
                (i, j)
            ] = packet

    # ========================================================
    # 8. Server forwards Round-1 encrypted shares
    # ========================================================

    client_inbox = {
        j: {}
        for j in U1
    }

    for (
        sender,
        receiver,
    ), packet in (
        server_share_packets.items()
    ):

        client_inbox[
            receiver
        ][sender] = packet

    # ========================================================
    # 9. ROUND 2
    #
    # Only U3 submits:
    #
    # ciphertext_i
    # partial_dk_i
    #
    # Client 4 does NOT submit either.
    # ========================================================

    ciphertexts = {}

    partial_dk = {}

    pair_masks = {}

    for i in sorted(U3):

        # ----------------------------------------------------
        # REAL MCFE ciphertext layer
        # ----------------------------------------------------

        ciphertexts[i] = (
            mcfe.encrypt(
                secret_key=(
                    client_state[i]["s"]
                ),
                message=(
                    client_values[i]
                ),
                label=label,
            )
        )

        # ----------------------------------------------------
        # DMCFE decentralized partial key
        #
        # dk_i,y =
        #
        # s_i
        # +
        # PRG(eta_i)
        # +
        # pairwise_mask_i
        #
        # Pairwise masks use U2.
        #
        # NOT U3.
        # ----------------------------------------------------

        pair_mask = 0

        for j in sorted(U2):

            if i == j:
                continue

            mask = _pair_prg(
                client_state[i][
                    "k1"
                ][j]
            )

            if i < j:

                pair_mask += mask

            else:

                pair_mask -= mask

        pair_masks[i] = (
            pair_mask
        )

        partial_dk[i] = (
            client_state[i]["s"]
            +
            _eta_prg(
                client_state[i][
                    "eta"
                ]
            )
            +
            pair_mask
        )

    ciphertext_submitters_correct = (
        set(ciphertexts.keys())
        ==
        U3
    )

    partial_dk_submitters_correct = (
        set(partial_dk.keys())
        ==
        U3
    )

    # ========================================================
    # 10. ROUND 3
    #
    # Only U4 responds.
    #
    # Each U4 client decrypts Round-1 packets.
    # ========================================================

    round3_messages = {}

    packet_decryption_correct = True

    for i in sorted(U4):

        decrypted_from_peers = {}

        for j in sorted(
            U2 - {i}
        ):

            recovered = (
                se.decrypt_share_packet(
                    key=(
                        client_state[i][
                            "k2"
                        ][j]
                    ),
                    packet=(
                        client_inbox[i][j]
                    ),
                )
            )

            if (
                recovered.sender_id != j
                or
                recovered.receiver_id != i
            ):

                packet_decryption_correct = (
                    False
                )

            decrypted_from_peers[
                j
            ] = recovered

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

            # U4 clients directly reveal eta_i.

            "eta_self": (
                client_state[i][
                    "eta"
                ]
            ),

            "eta_shares": (
                eta_shares_for_server
            ),

            "sk_shares": (
                sk_shares_for_server
            ),
        }

    # ========================================================
    # 11. Server reconstructs early-dropout sk1
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

        recovered_raw = (
            ss.combine_bytes(
                threshold=threshold,
                shares=available_shares,
                length=32,
            )
        )

        reference_raw = (
            _private_key_to_raw(
                client_state[
                    dropped_id
                ]["sk1"]
            )
        )

        if (
            recovered_raw
            !=
            reference_raw
        ):

            sk_recovery_correct = (
                False
            )

        recovered_sk1[
            dropped_id
        ] = (
            x25519
            .X25519PrivateKey
            .from_private_bytes(
                recovered_raw
            )
        )

    # ========================================================
    # 12. Server reconstructs eta for U3 - U4
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

        recovered_eta[
            dropped_id
        ] = (
            ss.combine(
                threshold=threshold,
                shares=available_shares,
            )
        )

        if (
            recovered_eta[
                dropped_id
            ]
            !=
            client_state[
                dropped_id
            ]["eta"]
        ):

            eta_recovery_correct = (
                False
            )

    # ========================================================
    # 13. Server now has eta_i for every i in U3
    # ========================================================

    server_eta = {}

    for i in U4:

        server_eta[i] = (
            round3_messages[i][
                "eta_self"
            ]
        )

    for i in late_dropout:

        server_eta[i] = (
            recovered_eta[i]
        )

    server_eta_correct = (
        set(server_eta.keys())
        ==
        U3
    )

    # ========================================================
    # 14. Recover unmatched pairwise masks
    #
    # caused by U2 - U3 = {4}
    # ========================================================

    residual_pair_mask = 0

    recovered_pair_keys_correct = (
        True
    )

    for dropped_id in sorted(
        early_dropout
    ):

        recovered_private = (
            recovered_sk1[
                dropped_id
            ]
        )

        for i in sorted(U3):

            recovered_key = (
                _derive_pairwise_key(
                    private_key=(
                        recovered_private
                    ),
                    peer_public_key=(
                        client_state[i][
                            "pk1"
                        ]
                    ),
                    info=K1_INFO,
                )
            )

            reference_key = (
                client_state[i][
                    "k1"
                ][dropped_id]
            )

            if (
                recovered_key
                !=
                reference_key
            ):

                recovered_pair_keys_correct = (
                    False
                )

            mask = _pair_prg(
                recovered_key
            )

            if i < dropped_id:

                residual_pair_mask += (
                    mask
                )

            else:

                residual_pair_mask -= (
                    mask
                )

    actual_pair_mask_sum = sum(
        pair_masks.values()
    )

    residual_mask_correct = (
        residual_pair_mask
        ==
        actual_pair_mask_sum
    )

    # ========================================================
    # 15. Recover dk_y'
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
    )

    surviving_client_ids = sorted(
        U3
    )
    surviving_secret_keys = [
        client_state[i]["s"]
        for i in surviving_client_ids
    ]
    
    ciphertext_list = [
        ciphertexts[i]
        for i in surviving_client_ids
    ]
    

    surviving_weights = [
        final_y[i]
        for i in surviving_client_ids
    ]

    reference_dk_y_prime = (
        mcfe.derive_key(
            secret_keys=surviving_secret_keys,
            weights=surviving_weights,
        )
    )

    dk_correct = (
        recovered_dk_y_prime
        ==
        reference_dk_y_prime
    )

    # ========================================================
    # 16. END-TO-END MCFE DECRYPT
    #
    # IMPORTANT:
    #
    # use ciphertexts from U3
    #
    # client 3 remains included,
    # even though client 3 is absent from U4.
    #
    # client 4 is excluded.
    # ========================================================

    ciphertext_list = [
        ciphertexts[i]
        for i in sorted(U3)
    ]

    decrypted_plaintext = (
        mcfe.decrypt(
            ciphertexts=(
                ciphertext_list
            ),
            weights=(
                surviving_weights
            ),
            derived_key=(
                recovered_dk_y_prime
            ),
            label=label,
        )
    )

    e2e_correct = (
        decrypted_plaintext
        ==
        expected_plaintext
    )
    logging.info(
        "MCFE backend: dmcfe.mcfe.MCFE"
    )

    logging.info(
        "MCFE modulus actual bits: %s",
        mcfe.N.bit_length(),
    )

    logging.info(
        "official MCFE API integration correctness: %s",
        e2e_correct,
    )
    # ========================================================
    # 17. Negative diagnostic
    #
    # Try decrypting with dk WITHOUT residual-mask repair.
    #
    # This should normally fail.
    # ========================================================

    naive_dk = (
        sum_partial_dk
        -
        eta_mask_sum
    )

    try:

        naive_decrypted = (
            mcfe.decrypt(
                ciphertexts=(
                    ciphertext_list
                ),
                weights=(
                    surviving_weights
                ),
                derived_key=(
                    naive_dk
                ),
                label=label,
            )
        )

    except Exception:

        naive_decrypted = None

    dropout_fix_needed = (
        naive_decrypted
        !=
        expected_plaintext
    )

    # ========================================================
    # 18. Overall correctness
    # ========================================================

    overall_correct = all(
        [
            final_weight_correct,
            ciphertext_submitters_correct,
            partial_dk_submitters_correct,
            packet_decryption_correct,
            sk_recovery_correct,
            eta_recovery_correct,
            server_eta_correct,
            recovered_pair_keys_correct,
            residual_mask_correct,
            dk_correct,
            e2e_correct,
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
        "early dropout U2-U3: %s",
        sorted(early_dropout),
    )

    logging.info(
        "late dropout U3-U4: %s",
        sorted(late_dropout),
    )

    logging.info(
        "client plaintexts: %s",
        client_values,
    )

    logging.info(
        "initial y: %s",
        initial_y,
    )

    logging.info(
        "final y': %s",
        final_y,
    )

    logging.info(
        "final weight correctness: %s",
        final_weight_correct,
    )

    logging.info(
        "ciphertext submitters: %s",
        sorted(ciphertexts.keys()),
    )

    logging.info(
        "ciphertext submitter set correctness: %s",
        ciphertext_submitters_correct,
    )

    logging.info(
        "partial-dk submitter set correctness: %s",
        partial_dk_submitters_correct,
    )

    logging.info(
        "Round-3 packet decryption correctness: %s",
        packet_decryption_correct,
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
        server_eta_correct,
    )

    logging.info(
        "recovered dropout pair keys correctness: %s",
        recovered_pair_keys_correct,
    )

    logging.info(
        "actual residual pair mask: %s",
        actual_pair_mask_sum,
    )

    logging.info(
        "reconstructed residual pair mask: %s",
        residual_pair_mask,
    )

    logging.info(
        "residual pair mask correctness: %s",
        residual_mask_correct,
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
        "dk_y' correctness: %s",
        dk_correct,
    )

    logging.info(
        "expected surviving plaintext sum: %s",
        expected_plaintext,
    )

    logging.info(
        "MCFE dropout decrypted result: %s",
        decrypted_plaintext,
    )

    logging.info(
        "naive decrypted result without dropout fix: %s",
        naive_decrypted,
    )

    logging.info(
        "dropout correction was required: %s",
        dropout_fix_needed,
    )

    logging.info(
        "DMCFE dropout E2E scalar correctness: %s",
        e2e_correct,
    )

    logging.info(
        "DMCFE dropout E2E overall correctness: %s",
        overall_correct,
    )

    logging.info(
        "========================================================="
    )

    logging.info("")

    # ========================================================
    # Assertions
    # ========================================================

    assert final_weight_correct

    assert ciphertext_submitters_correct

    assert partial_dk_submitters_correct

    assert packet_decryption_correct

    assert sk_recovery_correct

    assert eta_recovery_correct

    assert server_eta_correct

    assert recovered_pair_keys_correct

    assert residual_mask_correct

    assert dk_correct

    assert e2e_correct

    assert overall_correct

    return True