from __future__ import annotations

import logging
import secrets

from cryptography.hazmat.primitives.asymmetric import (
    x25519,
)

from .mcfe import MCFE

from .secret_sharing import (
    ShamirSecretSharing,
)

from .symmetric_encryption import (
    SymmetricEncryption,
)

from .dropout_smoke import (
    K1_INFO,
    K2_INFO,
    _derive_pairwise_key,
    _private_key_to_raw,
    _pair_prg,
    _eta_prg,
)


# ============================================================
# DMCFE 4-Dimensional Vector Dropout Smoke
# ============================================================
#
# Goal:
#
# scalar
#     ↓
# 4-dimensional vector
#
#
# Client vectors:
#
# client 0:
#     [10, -2,  3,  7]
#
# client 1:
#     [20,  5, -8,  1]
#
# client 2:
#     [-5,  4,  9, -3]
#
# client 3:
#     [100, -6, 2, 10]
#
# client 4:
#     [50, 20, 30, 40]
#
#
# Dropout:
#
# U1 = {0,1,2,3,4}
# U2 = {0,1,2,3,4}
#
# client 4 early dropout
#
# U3 = {0,1,2,3}
#
# client 3 late dropout
#
# U4 = {0,1,2}
#
#
# Expected final vector:
#
# client0 + client1 + client2 + client3
#
# =
#
# [125, 1, 6, 15]
#
#
# client 4 MUST be excluded.
# ============================================================


def _sum_vectors(
    vectors: list[list[int]],
) -> list[int]:

    if not vectors:
        raise ValueError(
            "vectors must not be empty"
        )

    dim = len(vectors[0])

    if dim == 0:
        raise ValueError(
            "vector dimension must be > 0"
        )

    for vector in vectors:

        if len(vector) != dim:

            raise ValueError(
                "all vectors must have "
                "the same dimension"
            )

    result = [
        0
        for _ in range(dim)
    ]

    for vector in vectors:

        for coordinate in range(dim):

            result[coordinate] += (
                int(
                    vector[
                        coordinate
                    ]
                )
            )

    return result

def _build_test_vectors(
    vector_dim: int,
) -> dict[int, list[int]]:
    """
    Build deterministic test vectors.

    The first 4 coordinates preserve our original
    4-D regression test exactly.

    Coordinates >= 4 are generated deterministically
    so the same function can test 32-D, 128-D, etc.
    """

    if vector_dim < 1:
        raise ValueError(
            "vector_dim must be >= 1"
        )

    # ========================================================
    # Preserve original 4-D regression values
    # ========================================================

    base_vectors = {

        0: [
            10,
            -2,
            3,
            7,
        ],

        1: [
            20,
            5,
            -8,
            1,
        ],

        2: [
            -5,
            4,
            9,
            -3,
        ],

        3: [
            100,
            -6,
            2,
            10,
        ],

        # Early dropout client
        4: [
            50,
            20,
            30,
            40,
        ],
    }

    # ========================================================
    # Case 1:
    # vector_dim <= 4
    # ========================================================

    if vector_dim <= 4:

        return {
            client_id: values[
                :vector_dim
            ]
            for client_id, values
            in base_vectors.items()
        }

    # ========================================================
    # Case 2:
    # extend beyond coordinate 3
    # ========================================================

    vectors = {
        client_id: list(values)
        for client_id, values
        in base_vectors.items()
    }

    for coordinate in range(
        4,
        vector_dim,
    ):

        # Mix positive / negative / zero values.

        vectors[0].append(
            (
                coordinate % 9
            )
            - 4
        )

        vectors[1].append(
            2
            * (
                (
                    coordinate % 7
                )
                - 3
            )
        )

        vectors[2].append(
            -(
                (
                    coordinate % 5
                )
                - 2
            )
        )

        vectors[3].append(
            (
                (
                    3 * coordinate
                    + 1
                )
                % 13
            )
            - 6
        )

        # Early dropout client's coordinates
        # are deliberately large enough that accidental
        # inclusion is easy to detect.

        vectors[4].append(
            50
            + coordinate
        )

    return vectors

def run_dmcfe_vector_dropout_smoke(
    vector_dim: int = 4,
):
    if vector_dim < 1:
        raise ValueError(
            "vector_dim must be >= 1"
        )
    logging.info("")

    logging.info(
        "========== DMCFE %s-D Vector Dropout Smoke ==========",
        vector_dim,
    )

    # ========================================================
    # 1. Protocol sets
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

    # Client 4 disappears before Round-2 submission.
    U3 = {
        0,
        1,
        2,
        3,
    }

    # Client 3 submitted ciphertexts,
    # but disappears before Round-3 recovery.
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
            "Round 3 abort: "
            "|U4| < threshold"
        )

    early_dropout = (
        U2 - U3
    )

    late_dropout = (
        U3 - U4
    )

    # ========================================================
    # 2. Client vectors
    # ========================================================

    client_vectors = (
        _build_test_vectors(
            vector_dim=vector_dim
        )
    )

    for client_id in U1:

        if len(
            client_vectors[
                client_id
            ]
        ) != vector_dim:

            raise RuntimeError(
                "client vectors have "
                "inconsistent dimensions"
            )

    # ========================================================
    # Expected surviving plaintext
    # ========================================================

    expected_vector = (
        _sum_vectors(
            [
                client_vectors[i]
                for i in sorted(U3)
            ]
        )
    )

    # ========================================================
    # Preserve original first-4-coordinate regression
    # ========================================================

    expected_first_four = [
        125,
        1,
        6,
        15,
    ]

    if vector_dim >= 4:

        first_four_correct = (
            expected_vector[:4]
            ==
            expected_first_four
        )

    else:

        first_four_correct = (
            expected_vector
            ==
            expected_first_four[
                :vector_dim
            ]
        )

    assert first_four_correct

    # ========================================================
    # 3. Final weight vector
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
    # 4. Crypto initialization
    # ========================================================

    # One public MCFE setup shared by all clients.
    mcfe = MCFE(
        modulus_bits=256
    )

    ss = (
        ShamirSecretSharing()
    )

    se = (
        SymmetricEncryption()
    )

    # --------------------------------------------------------
    # Each vector coordinate uses a different label.
    #
    # Same s_i
    # Same y
    # Same derived dk_y'
    #
    # but different H(label).
    # --------------------------------------------------------

    labels = [
        (
            f"DMCFE-IP/"
            f"vector-dropout/"
            f"coord-{coordinate}"
        ).encode(
            "utf-8"
        )
        for coordinate
        in range(vector_dim)
    ]

    distinct_labels_correct = (
        len(
            set(labels)
        )
        ==
        vector_dim
    )

    # ========================================================
    # 5. ROUND 0
    #
    # Generate:
    #
    # MCFE s_i
    #
    # X25519:
    #
    # sk_i^(1), pk_i^(1)
    # sk_i^(2), pk_i^(2)
    #
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

            # REAL MCFE KeyGen
            "s": (
                mcfe.keygen()
            ),

            "eta": (
                secrets.randbits(
                    128
                )
            ),

            "k1": {},

            "k2": {},
        }

    # ========================================================
    # 6. ROUND 1
    #
    # Pairwise KA
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
                        client_state[
                            i
                        ]["sk1"]
                    ),
                    peer_public_key=(
                        client_state[
                            j
                        ]["pk1"]
                    ),
                    info=K1_INFO,
                )
            )

            client_state[
                i
            ]["k2"][j] = (
                _derive_pairwise_key(
                    private_key=(
                        client_state[
                            i
                        ]["sk2"]
                    ),
                    peer_public_key=(
                        client_state[
                            j
                        ]["pk2"]
                    ),
                    info=K2_INFO,
                )
            )

    # ========================================================
    # Verify KA symmetry
    # ========================================================

    pairwise_ka_correct = True

    for i in sorted(U1):

        for j in sorted(U1):

            if i >= j:
                continue

            if (
                client_state[
                    i
                ]["k1"][j]
                !=
                client_state[
                    j
                ]["k1"][i]
            ):

                pairwise_ka_correct = (
                    False
                )

            if (
                client_state[
                    i
                ]["k2"][j]
                !=
                client_state[
                    j
                ]["k2"][i]
            ):

                pairwise_ka_correct = (
                    False
                )

    # ========================================================
    # 7. ROUND 1
    #
    # Shamir:
    #
    # sk_i^(1)
    # eta_i
    #
    # AES-GCM:
    #
    # C_i,j
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
    # Server forwards C_i,j
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
    # 8. ROUND 2
    #
    # Vector ciphertexts
    #
    # ciphertexts[i][coordinate]
    #
    # Client 4 is absent.
    # ========================================================

    ciphertexts = {}

    partial_dk = {}

    pair_masks = {}

    for i in sorted(U3):

        ciphertexts[i] = []

        # ----------------------------------------------------
        # REAL MCFE vector encryption
        #
        # One ciphertext per coordinate.
        # ----------------------------------------------------

        for coordinate in range(
            vector_dim
        ):

            ciphertext = (
                mcfe.encrypt(
                    secret_key=(
                        client_state[i][
                            "s"
                        ]
                    ),
                    message=(
                        client_vectors[i][
                            coordinate
                        ]
                    ),
                    label=(
                        labels[
                            coordinate
                        ]
                    ),
                )
            )

            ciphertexts[
                i
            ].append(
                ciphertext
            )

        # ----------------------------------------------------
        # DMCFE partial key
        #
        # IMPORTANT:
        #
        # This is generated ONCE per client,
        # not once per vector coordinate.
        #
        # dk_i,y depends on y,
        # not on the plaintext coordinate.
        # ----------------------------------------------------

        pair_mask = 0

        for j in sorted(U2):

            if i == j:
                continue

            mask = (
                _pair_prg(
                    client_state[i][
                        "k1"
                    ][j]
                )
            )

            if i < j:

                pair_mask += mask

            else:

                pair_mask -= mask

        pair_masks[i] = (
            pair_mask
        )

        partial_dk[i] = (
            client_state[i][
                "s"
            ]
            +
            _eta_prg(
                client_state[i][
                    "eta"
                ]
            )
            +
            pair_mask
        )

    # ========================================================
    # Ciphertext structure checks
    # ========================================================

    ciphertext_submitters_correct = (
        set(
            ciphertexts.keys()
        )
        ==
        U3
    )

    ciphertext_dimensions_correct = (
        all(
            len(
                ciphertexts[i]
            )
            ==
            vector_dim

            for i in U3
        )
    )

    total_ciphertext_count = sum(
        len(
            ciphertexts[i]
        )
        for i in U3
    )

    expected_ciphertext_count = (
        len(U3)
        *
        vector_dim
    )

    ciphertext_count_correct = (
        total_ciphertext_count
        ==
        expected_ciphertext_count
    )

    # ========================================================
    # 9. ROUND 3
    #
    # U4 responds.
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
                        client_state[i]
                        ["k2"][j]
                    ),
                    packet=(
                        client_inbox[i][j]
                    ),
                )
            )

            if (
                recovered.sender_id
                !=
                j
                or
                recovered.receiver_id
                !=
                i
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
    # 10. Recover sk^(1)
    #
    # for U2 - U3
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

            for responder
            in U4
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
    # 11. Recover eta
    #
    # for U3 - U4
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

            for responder
            in U4
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
    # Server eta set for all U3
    # ========================================================

    server_eta = {}

    for i in U4:

        server_eta[i] = (
            round3_messages[
                i
            ]["eta_self"]
        )

    for i in late_dropout:

        server_eta[i] = (
            recovered_eta[i]
        )

    server_eta_correct = (
        set(
            server_eta.keys()
        )
        ==
        U3
    )

    # ========================================================
    # 12. Recover unmatched pairwise mask
    #
    # caused by early dropout client 4.
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

            mask = (
                _pair_prg(
                    recovered_key
                )
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
    # 13. Recover dk_y'
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

    # ========================================================
    # REAL MCFE KeyDer reference
    # ========================================================

    surviving_client_ids = (
        sorted(U3)
    )

    surviving_secret_keys = [
        client_state[i]["s"]
        for i
        in surviving_client_ids
    ]

    surviving_weights = [
        final_y[i]
        for i
        in surviving_client_ids
    ]

    reference_dk_y_prime = (
        mcfe.derive_key(
            secret_keys=(
                surviving_secret_keys
            ),
            weights=(
                surviving_weights
            ),
        )
    )

    dk_correct = (
        recovered_dk_y_prime
        ==
        reference_dk_y_prime
    )

    # ========================================================
    # 14. REAL MCFE vector decryption
    #
    # Same recovered dk_y'
    #
    # Different ciphertext coordinate
    # Different label
    # ========================================================

    decrypted_vector = []

    coordinate_correctness = []

    for coordinate in range(
        vector_dim
    ):

        ciphertext_list = [

            ciphertexts[i][
                coordinate
            ]

            for i
            in surviving_client_ids
        ]

        decrypted_coordinate = (
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
                label=(
                    labels[
                        coordinate
                    ]
                ),
            )
        )

        decrypted_vector.append(
            decrypted_coordinate
        )

        coordinate_correctness.append(
            decrypted_coordinate
            ==
            expected_vector[
                coordinate
            ]
        )

    vector_correct = (
        decrypted_vector
        ==
        expected_vector
    )

    all_coordinates_correct = (
        all(
            coordinate_correctness
        )
    )
    correct_coordinate_count = sum(
        1
        for correct
        in coordinate_correctness
        if correct
    )

    coordinate_count_correct = (
        correct_coordinate_count
        ==
        vector_dim
    )
    # ========================================================
    # 15. Negative test
    #
    # Do not repair early-dropout mask.
    #
    # The same ciphertext vector should NOT
    # correctly decrypt.
    # ========================================================

    naive_dk = (
        sum_partial_dk
        -
        eta_mask_sum
    )

    naive_vector = []

    naive_failed = False

    for coordinate in range(
        vector_dim
    ):

        ciphertext_list = [

            ciphertexts[i][
                coordinate
            ]

            for i
            in surviving_client_ids
        ]

        try:

            naive_value = (
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
                    label=(
                        labels[
                            coordinate
                        ]
                    ),
                )
            )

            naive_vector.append(
                naive_value
            )

        except RuntimeError:

            naive_failed = True

            naive_vector = None

            break

    dropout_fix_required = (
        naive_vector
        !=
        expected_vector
    )

    # ========================================================
    # Overall
    # ========================================================

    overall_correct = all(
        [
            first_four_correct,
            final_weight_correct,
            distinct_labels_correct,
            pairwise_ka_correct,
            ciphertext_submitters_correct,
            ciphertext_dimensions_correct,
            ciphertext_count_correct,
            packet_decryption_correct,
            sk_recovery_correct,
            eta_recovery_correct,
            server_eta_correct,
            recovered_pair_keys_correct,
            residual_mask_correct,
            dk_correct,
            all_coordinates_correct,
            vector_correct,
            dropout_fix_required,
            
            coordinate_count_correct,
        ]
    )

    # ========================================================
    # Logging
    # ========================================================

    logging.info(
        "MCFE backend: dmcfe.mcfe.MCFE"
    )

    logging.info(
        "MCFE modulus bits: %s",
        mcfe.N.bit_length(),
    )

    logging.info(
        "vector dimension: %s",
        vector_dim,
    )

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
        "early dropout: %s",
        sorted(
            early_dropout
        ),
    )

    logging.info(
        "late dropout: %s",
        sorted(
            late_dropout
        ),
    )

    preview_size = min(
        8,
        vector_dim,
    )

    client_vector_preview = {
        client_id:
        client_vectors[
            client_id
        ][:preview_size]

        for client_id
        in sorted(U1)
    }

    logging.info(
        "client vector preview "
        "(first %s coords): %s",
        preview_size,
        client_vector_preview,
    )

    logging.info(
        "final y': %s",
        final_y,
    )

    logging.info(
        "distinct coordinate labels: %s",
        distinct_labels_correct,
    )

    logging.info(
        "ciphertext submitters: %s",
        sorted(
            ciphertexts.keys()
        ),
    )

    logging.info(
        "ciphertexts per surviving client: %s",
        vector_dim,
    )

    logging.info(
        "total ciphertext count: %s",
        total_ciphertext_count,
    )

    logging.info(
        "expected ciphertext count: %s",
        expected_ciphertext_count,
    )

    logging.info(
        "ciphertext count correctness: %s",
        ciphertext_count_correct,
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
        "expected vector preview: %s",
        expected_vector[
            :preview_size
        ],
    )

    logging.info(
        "decrypted vector preview: %s",
        decrypted_vector[
            :preview_size
        ],
    )
    logging.info(
        "correct coordinates: %s/%s",
        correct_coordinate_count,
        vector_dim,
    )
    
    logging.info(
        "coordinate correctness: %s",
        coordinate_correctness,
    )

    logging.info(
        "all coordinates correctness: %s",
        all_coordinates_correct,
    )

    logging.info(
        "naive vector without dropout fix: %s",
        naive_vector,
    )

    logging.info(
        "naive decrypt raised consistency failure: %s",
        naive_failed,
    )

    logging.info(
        "dropout correction required: %s",
        dropout_fix_required,
    )

    logging.info(
        "DMCFE %s-D vector correctness: %s",
        vector_dim,
        vector_correct,
    )

    logging.info(
        "DMCFE %s-D vector dropout overall correctness: %s",
        vector_dim,
        overall_correct,
    )

    logging.info(
        "======================================================"
    )

    logging.info("")

    # ========================================================
    # Assertions
    # ========================================================

    assert first_four_correct
    assert final_weight_correct
    assert distinct_labels_correct
    assert pairwise_ka_correct
    assert ciphertext_submitters_correct
    assert ciphertext_dimensions_correct
    assert ciphertext_count_correct
    assert packet_decryption_correct
    assert sk_recovery_correct
    assert eta_recovery_correct
    assert server_eta_correct
    assert recovered_pair_keys_correct
    assert residual_mask_correct
    assert dk_correct
    assert all_coordinates_correct
    assert vector_correct
    assert dropout_fix_required
    assert overall_correct
    assert coordinate_count_correct
    
    return True