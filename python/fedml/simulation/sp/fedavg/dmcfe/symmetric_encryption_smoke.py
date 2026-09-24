import logging
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import (
    AESGCM,
)

from .secret_sharing import (
    ShamirSecretSharing,
)

from .symmetric_encryption import (
    SymmetricEncryption,
)


def run_symmetric_encryption_smoke():

    logging.info("")
    logging.info(
        "========== DMCFE Symmetric Encryption Smoke =========="
    )

    # ========================================================
    # Simulated DMCFE Round 1
    # ========================================================

    dealer_id = 0

    receiver_id = 2

    participant_ids = [
        1,
        2,
        3,
        4,
    ]

    threshold = 3

    # ========================================================
    # Generate Shamir shares
    # ========================================================

    ss = ShamirSecretSharing()

    # --------------------------------------------------------
    # Simulated X25519 private key sk_i^(1)
    #
    # For this unit test we only need its raw 32-byte value.
    # --------------------------------------------------------

    sk_original_raw = secrets.token_bytes(
        32
    )

    sk_shares = ss.share_bytes(
        threshold=threshold,
        participant_ids=participant_ids,
        secret=sk_original_raw,
    )

    # --------------------------------------------------------
    # eta_i
    # --------------------------------------------------------

    eta_original = (
        9876543210123456789
    )

    eta_shares = ss.share(
        threshold=threshold,
        participant_ids=participant_ids,
        secret=eta_original,
    )

    sk_share_for_receiver = (
        sk_shares[receiver_id]
    )

    eta_share_for_receiver = (
        eta_shares[receiver_id]
    )

    # ========================================================
    # Simulated k^(2)_{i,j}
    #
    # For this unit test:
    #
    # generate a 256-bit AES key.
    #
    # In the integrated Round-1 implementation this will be
    # replaced with the real pairwise k^(2)_{i,j}.
    # ========================================================

    key_2_ij = AESGCM.generate_key(
        bit_length=256
    )

    se = SymmetricEncryption()

    # ========================================================
    # Client i:
    #
    # C_i,j =
    #
    # SE.Enc(
    #     k2_i,j,
    #     i || j || sk_share || eta_share
    # )
    # ========================================================

    encrypted_packet = (
        se.encrypt_share_packet(
            key=key_2_ij,
            sender_id=dealer_id,
            receiver_id=receiver_id,
            sk_share=(
                sk_share_for_receiver
            ),
            eta_share=(
                eta_share_for_receiver
            ),
        )
    )

    # ========================================================
    # Server
    #
    # Important:
    #
    # Server does NOT decrypt.
    #
    # It only stores / forwards C_i,j.
    # ========================================================

    server_forwarded_packet = (
        encrypted_packet
    )

    # ========================================================
    # Client j decrypts C_i,j
    # ========================================================

    recovered = (
        se.decrypt_share_packet(
            key=key_2_ij,
            packet=server_forwarded_packet,
        )
    )

    sender_correct = (
        recovered.sender_id
        == dealer_id
    )

    receiver_correct = (
        recovered.receiver_id
        == receiver_id
    )

    sk_share_correct = (
        recovered.sk_share
        == sk_share_for_receiver
    )

    eta_share_correct = (
        recovered.eta_share
        == eta_share_for_receiver
    )

    round_trip_correct = all(
        [
            sender_correct,
            receiver_correct,
            sk_share_correct,
            eta_share_correct,
        ]
    )

    # ========================================================
    # Test fresh nonce
    #
    # Encrypt same message again.
    #
    # Packet should differ because nonce is fresh.
    # ========================================================

    encrypted_packet_2 = (
        se.encrypt_share_packet(
            key=key_2_ij,
            sender_id=dealer_id,
            receiver_id=receiver_id,
            sk_share=(
                sk_share_for_receiver
            ),
            eta_share=(
                eta_share_for_receiver
            ),
        )
    )

    fresh_ciphertext = (
        encrypted_packet_2
        != encrypted_packet
    )

    # ========================================================
    # Tampering test
    #
    # Modify one ciphertext byte.
    #
    # AES-GCM authentication must reject it.
    # ========================================================

    tampered = bytearray(
        encrypted_packet
    )

    tampered[-1] ^= 0x01

    tampering_rejected = False

    try:

        se.decrypt_share_packet(
            key=key_2_ij,
            packet=bytes(tampered),
        )

    except InvalidTag:

        tampering_rejected = True

    # ========================================================
    # Wrong-key test
    # ========================================================

    wrong_key = AESGCM.generate_key(
        bit_length=256
    )

    wrong_key_rejected = False

    try:

        se.decrypt_share_packet(
            key=wrong_key,
            packet=encrypted_packet,
        )

    except InvalidTag:

        wrong_key_rejected = True

    # ========================================================
    # Logging
    # ========================================================

    logging.info(
        "dealer id: %s",
        dealer_id,
    )

    logging.info(
        "receiver id: %s",
        receiver_id,
    )

    logging.info(
        "encrypted packet bytes: %s",
        len(encrypted_packet),
    )

    logging.info(
        "sender id correctness: %s",
        sender_correct,
    )

    logging.info(
        "receiver id correctness: %s",
        receiver_correct,
    )

    logging.info(
        "sk share correctness: %s",
        sk_share_correct,
    )

    logging.info(
        "eta share correctness: %s",
        eta_share_correct,
    )

    logging.info(
        "SE round-trip correctness: %s",
        round_trip_correct,
    )

    logging.info(
        "fresh ciphertext / nonce: %s",
        fresh_ciphertext,
    )

    logging.info(
        "tampering rejected: %s",
        tampering_rejected,
    )

    logging.info(
        "wrong key rejected: %s",
        wrong_key_rejected,
    )

    logging.info(
        "======================================================="
    )

    logging.info("")

    # ========================================================
    # Hard assertions
    # ========================================================

    assert sender_correct

    assert receiver_correct

    assert sk_share_correct

    assert eta_share_correct

    assert round_trip_correct

    assert fresh_ciphertext

    assert tampering_rejected

    assert wrong_key_rejected

    return True