from __future__ import annotations

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
# DMCFE Round-1 KeySharing integration smoke
# ============================================================
#
# Paper mapping:
#
# Round 0:
#
#   client i:
#
#       (pk_i^(1), sk_i^(1))
#       (pk_i^(2), sk_i^(2))
#
# Round 1:
#
#       k_i,j^(1)
#           = KA.agree(pk_j^(1), sk_i^(1))
#
#       k_i,j^(2)
#           = KA.agree(pk_j^(2), sk_i^(2))
#
#       [[sk_i^(1)]]_j
#           <- SS.Share(...)
#
#       [[eta_i]]_j
#           <- SS.Share(...)
#
#       C_i,j
#           =
#       SE.Enc(
#           k_i,j^(2),
#           i || j ||
#           [[sk_i^(1)]]_j ||
#           [[eta_i]]_j
#       )
#
# Server only forwards C_i,j.
#
# Client j decrypts it using k_j,i^(2).
# ============================================================


K1_INFO = b"DMCFE-IP/k1/v1"
K2_INFO = b"DMCFE-IP/k2/v1"


def _derive_pairwise_key(
    private_key: x25519.X25519PrivateKey,
    peer_public_key: x25519.X25519PublicKey,
    info: bytes,
) -> bytes:
    """
    X25519 + HKDF concrete instantiation of KA.agree().

    Returns:
        32-byte pairwise symmetric key.
    """

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
    """
    Serialize X25519 private key to 32 raw bytes.
    """

    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=(
            serialization.NoEncryption()
        ),
    )


def run_round1_keysharing_smoke():

    logging.info("")
    logging.info(
        "========== DMCFE Round-1 KeySharing Smoke =========="
    )

    # ========================================================
    # Configuration
    #
    # U1 = {0,1,2,3,4}
    #
    # threshold t = 3
    # ========================================================

    U1 = [
        0,
        1,
        2,
        3,
        4,
    ]

    threshold = 3

    n = len(U1)

    if n < threshold:

        raise ValueError(
            "|U1| must be >= threshold"
        )

    # ========================================================
    # Initialize primitives
    # ========================================================

    ss = ShamirSecretSharing()

    se = SymmetricEncryption()

    # ========================================================
    # ROUND 0
    #
    # Every client generates TWO independent
    # X25519 key pairs.
    #
    # client_state[i]:
    #
    #     sk1, pk1
    #     sk2, pk2
    #     eta
    #     k1
    #     k2
    # ========================================================

    client_state = {}

    for i in U1:

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

            # correctness-oriented eta
            "eta": secrets.randbits(
                128
            ),

            "k1": {},

            "k2": {},
        }

    # ========================================================
    # ROUND 1 — KA.agree
    #
    # Each client independently derives:
    #
    # k_i,j^(1)
    # k_i,j^(2)
    # ========================================================

    for i in U1:

        for j in U1:

            if i == j:
                continue

            # ------------------------------------------------
            # k^(1)
            # ------------------------------------------------

            client_state[i]["k1"][j] = (
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

            # ------------------------------------------------
            # k^(2)
            # ------------------------------------------------

            client_state[i]["k2"][j] = (
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
    # Verify pairwise agreement
    #
    # k_i,j == k_j,i
    # ========================================================

    k1_symmetric = True

    k2_symmetric = True

    k1_k2_separated = True

    for i in U1:

        for j in U1:

            if i >= j:
                continue

            if (
                client_state[i]["k1"][j]
                !=
                client_state[j]["k1"][i]
            ):

                k1_symmetric = False

            if (
                client_state[i]["k2"][j]
                !=
                client_state[j]["k2"][i]
            ):

                k2_symmetric = False

            # k1 and k2 must belong to separate
            # key-agreement domains.

            if (
                client_state[i]["k1"][j]
                ==
                client_state[i]["k2"][j]
            ):

                k1_k2_separated = False

    # ========================================================
    # ROUND 1 — Secret Sharing
    #
    # Reference shares are kept ONLY for smoke-test
    # correctness comparison.
    #
    # In the real protocol the server must not know
    # plaintext shares.
    # ========================================================

    reference_sk_shares = {}

    reference_eta_shares = {}

    # Server sees encrypted C_i,j only.

    server_packets = {}

    for i in U1:

        recipients = [
            j
            for j in U1
            if j != i
        ]

        # ----------------------------------------------------
        # Paper:
        #
        # SS.Share(
        #     t,
        #     U1 - {i},
        #     sk_i^(1)
        # )
        # ----------------------------------------------------

        sk1_raw = _private_key_to_raw(
            client_state[i]["sk1"]
        )

        if len(sk1_raw) != 32:

            raise RuntimeError(
                "unexpected X25519 "
                "private key length"
            )

        sk_shares = ss.share_bytes(
            threshold=threshold,
            participant_ids=recipients,
            secret=sk1_raw,
        )

        # ----------------------------------------------------
        # Paper:
        #
        # SS.Share(
        #     t,
        #     U1 - {i},
        #     eta_i
        # )
        # ----------------------------------------------------

        eta_shares = ss.share(
            threshold=threshold,
            participant_ids=recipients,
            secret=client_state[i][
                "eta"
            ],
        )

        reference_sk_shares[i] = (
            sk_shares
        )

        reference_eta_shares[i] = (
            eta_shares
        )

        # ====================================================
        # Build C_i,j
        # ====================================================

        for j in recipients:

            k2_ij = (
                client_state[i][
                    "k2"
                ][j]
            )

            packet = (
                se.encrypt_share_packet(
                    key=k2_ij,

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

            # -----------------------------------------------
            # Client i -> Server
            #
            # Server stores:
            #
            # (i, C_i,j)
            #
            # but never decrypts C_i,j.
            # -----------------------------------------------

            server_packets[
                (i, j)
            ] = packet

    # ========================================================
    # Server-side packet-count validation
    #
    # Every client sends n-1 packets.
    #
    # Total:
    #
    # n(n-1)
    # ========================================================

    expected_packet_count = (
        n * (n - 1)
    )

    packet_count_correct = (
        len(server_packets)
        ==
        expected_packet_count
    )

    # ========================================================
    # Server forwards packets to receivers.
    #
    # Client j receives:
    #
    # { C_i,j } for every i != j
    #
    # and decrypts using:
    #
    # k_j,i^(2)
    #
    # Since:
    #
    # k_i,j^(2)
    # =
    # k_j,i^(2)
    #
    # decryption must succeed.
    # ========================================================

    received_shares = {
        j: {}
        for j in U1
    }

    all_payloads_correct = True

    receiver_packet_counts_correct = True

    for j in U1:

        packets_for_j = [
            (i, packet)
            for (i, receiver), packet
            in server_packets.items()
            if receiver == j
        ]

        if len(packets_for_j) != (
            n - 1
        ):

            receiver_packet_counts_correct = (
                False
            )

        for (
            i,
            packet,
        ) in packets_for_j:

            # -----------------------------------------------
            # IMPORTANT:
            #
            # sender encrypted with:
            #
            #     k_i,j^(2)
            #
            # receiver decrypts with:
            #
            #     k_j,i^(2)
            #
            # -----------------------------------------------

            k2_ji = (
                client_state[j][
                    "k2"
                ][i]
            )

            recovered = (
                se.decrypt_share_packet(
                    key=k2_ji,
                    packet=packet,
                )
            )

            # =================================================
            # Paper Round-3 later requires checking:
            #
            # j' = j
            # i' = i
            #
            # We already validate identities here.
            # =================================================

            sender_correct = (
                recovered.sender_id
                == i
            )

            receiver_correct = (
                recovered.receiver_id
                == j
            )

            sk_share_correct = (
                recovered.sk_share
                ==
                reference_sk_shares[
                    i
                ][j]
            )

            eta_share_correct = (
                recovered.eta_share
                ==
                reference_eta_shares[
                    i
                ][j]
            )

            packet_correct = all(
                [
                    sender_correct,
                    receiver_correct,
                    sk_share_correct,
                    eta_share_correct,
                ]
            )

            if not packet_correct:

                all_payloads_correct = (
                    False
                )

            # -----------------------------------------------
            # Store plaintext shares LOCALLY at receiver j.
            #
            # These will be used in Round 3 later.
            # -----------------------------------------------

            received_shares[j][i] = {

                "sk_share": (
                    recovered.sk_share
                ),

                "eta_share": (
                    recovered.eta_share
                ),
            }

    # ========================================================
    # Every client should hold shares from n-1 peers.
    # ========================================================

    received_share_structure_correct = (
        all(
            len(
                received_shares[j]
            )
            == n - 1

            for j in U1
        )
    )

    # ========================================================
    # Overall Round-1 correctness
    # ========================================================

    round1_correct = all(
        [
            k1_symmetric,
            k2_symmetric,
            k1_k2_separated,
            packet_count_correct,
            receiver_packet_counts_correct,
            received_share_structure_correct,
            all_payloads_correct,
        ]
    )

    # ========================================================
    # Logging
    # ========================================================

    logging.info(
        "U1: %s",
        U1,
    )

    logging.info(
        "threshold: %s",
        threshold,
    )

    logging.info(
        "client count: %s",
        n,
    )

    logging.info(
        "k1 pairwise agreement correctness: %s",
        k1_symmetric,
    )

    logging.info(
        "k2 pairwise agreement correctness: %s",
        k2_symmetric,
    )

    logging.info(
        "k1 / k2 domain separation: %s",
        k1_k2_separated,
    )

    logging.info(
        "encrypted C_i,j packet count: %s",
        len(server_packets),
    )

    logging.info(
        "expected packet count: %s",
        expected_packet_count,
    )

    logging.info(
        "packet count correctness: %s",
        packet_count_correct,
    )

    logging.info(
        "receiver packet counts correctness: %s",
        receiver_packet_counts_correct,
    )

    logging.info(
        "received share structure correctness: %s",
        received_share_structure_correct,
    )

    logging.info(
        "all encrypted share payloads correctness: %s",
        all_payloads_correct,
    )

    logging.info(
        "DMCFE Round-1 KeySharing correctness: %s",
        round1_correct,
    )

    logging.info(
        "===================================================="
    )

    logging.info("")

    # ========================================================
    # Hard assertions
    # ========================================================

    assert k1_symmetric

    assert k2_symmetric

    assert k1_k2_separated

    assert packet_count_correct

    assert receiver_packet_counts_correct

    assert received_share_structure_correct

    assert all_payloads_correct

    assert round1_correct

    return True