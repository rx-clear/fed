import logging

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from .secret_sharing import ShamirSecretSharing


def run_secret_sharing_smoke():

    logging.info("")
    logging.info(
        "========== Shamir Secret Sharing Smoke =========="
    )

    ss = ShamirSecretSharing()

    # ========================================================
    # Simulate:
    #
    # client 0 is the dealer
    #
    # U1 = {0, 1, 2, 3, 4}
    #
    # According to the paper:
    #
    # SS.Share(t, U1 - {0}, secret)
    #
    # Therefore recipients are:
    #
    # {1, 2, 3, 4}
    #
    # ========================================================

    dealer_id = 0

    participant_ids = [
        1,
        2,
        3,
        4,
    ]

    threshold = 3

    # ========================================================
    # Test 1:
    # Share eta_i
    # ========================================================

    eta_original = 12345678901234567890

    eta_shares = ss.share(
        threshold=threshold,
        participant_ids=participant_ids,
        secret=eta_original,
    )

    # Simulate only clients 1, 3, 4 surviving.
    #
    # 3 shares >= threshold

    eta_available_shares = {
        1: eta_shares[1],
        3: eta_shares[3],
        4: eta_shares[4],
    }

    eta_recovered = ss.combine(
        threshold=threshold,
        shares=eta_available_shares,
    )

    eta_correct = (
        eta_recovered
        == eta_original
    )

    # ========================================================
    # Test 2:
    # Less than threshold must fail
    # ========================================================

    insufficient_rejected = False

    try:

        ss.combine(
            threshold=threshold,
            shares={
                1: eta_shares[1],
                2: eta_shares[2],
            },
        )

    except ValueError:

        insufficient_rejected = True

    # ========================================================
    # Test 3:
    # Share X25519 private key sk_i^(1)
    # ========================================================

    sk_original = (
        x25519.X25519PrivateKey.generate()
    )

    # Official cryptography-compatible raw serialization.
    #
    # We deliberately do not use private_bytes_raw()
    # because private_bytes() works with older versions too.

    sk_original_raw = sk_original.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=(
            serialization.NoEncryption()
        ),
    )

    if len(sk_original_raw) != 32:
        raise RuntimeError(
            "unexpected X25519 private key length"
        )

    sk_shares = ss.share_bytes(
        threshold=threshold,
        participant_ids=participant_ids,
        secret=sk_original_raw,
    )

    # Different surviving subset:
    #
    # clients 1, 2, 4

    sk_available_shares = {
        1: sk_shares[1],
        2: sk_shares[2],
        4: sk_shares[4],
    }

    sk_recovered_raw = ss.combine_bytes(
        threshold=threshold,
        shares=sk_available_shares,
        length=32,
    )

    sk_raw_correct = (
        sk_recovered_raw
        == sk_original_raw
    )

    # ========================================================
    # Rebuild actual X25519 private key object
    # ========================================================

    sk_recovered = (
        x25519.X25519PrivateKey.from_private_bytes(
            sk_recovered_raw
        )
    )

    original_public_key = (
        sk_original
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )

    recovered_public_key = (
        sk_recovered
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )

    sk_object_correct = (
        original_public_key
        == recovered_public_key
    )

    # ========================================================
    # Logging
    # ========================================================

    logging.info(
        "dealer id: %s",
        dealer_id,
    )

    logging.info(
        "recipient clients: %s",
        participant_ids,
    )

    logging.info(
        "threshold: %s-of-%s",
        threshold,
        len(participant_ids),
    )

    logging.info(
        "eta reconstruction correctness: %s",
        eta_correct,
    )

    logging.info(
        "insufficient shares rejected: %s",
        insufficient_rejected,
    )

    logging.info(
        "X25519 raw key reconstruction correctness: %s",
        sk_raw_correct,
    )

    logging.info(
        "X25519 restored key object correctness: %s",
        sk_object_correct,
    )

    logging.info(
        "================================================="
    )

    logging.info("")

    # ========================================================
    # Hard correctness assertions
    # ========================================================

    assert eta_correct

    assert insufficient_rejected

    assert sk_raw_correct

    assert sk_object_correct

    return True