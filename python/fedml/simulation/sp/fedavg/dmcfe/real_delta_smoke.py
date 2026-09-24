from __future__ import annotations

import logging
import secrets
from typing import Sequence
import time
import torch

from cryptography.hazmat.primitives.asymmetric import (
    x25519,
)

from .mcfe import MCFE

from .dropout_smoke import (
    K1_INFO,
    _derive_pairwise_key,
    _private_key_to_raw,
    _prg_scalar,
    _eta_prg,
)
from .secret_sharing import ShamirSecretSharing
from .symmetric_encryption import SymmetricEncryption

def _pair_prg_dot_weights(
    key: bytes,
    weights: Sequence[int],
) -> int:
    """
    Correctness-oriented implementation of:

        PRG(k_i,j^(1)) · y

    Paper PRG output is a vector whose dimension
    equals the length of the client weight vector y.

    Both clients in the same pair must derive the
    EXACT SAME scalar inner product so that the
    + / - pairwise masks cancel.
    """

    result = 0

    for position, weight in enumerate(
        weights
    ):

        domain = (
            b"DMCFE/pairwise-mask/vector/v1/"
            +
            int(position).to_bytes(
                4,
                byteorder="big",
                signed=False,
            )
        )

        mask_component = (
            _prg_scalar(
                key,
                domain,
            )
        )

        result += (
            mask_component
            *
            int(weight)
        )

    return int(
        result
    )
def run_real_quantized_delta_smoke(
    client_vectors: Sequence[torch.Tensor],
    client_sample_nums: Sequence[int],
    round_idx: int,
    vector_dim: int = 32,
    sampled_client_ids=None,
    dropout_before_encryption_ids=None,
    dropout_before_recovery_ids=None,
    recovery_threshold=None,
):
    """
    Real FedML quantized-delta -> DMCFE correctness test.

    Current checkpoint:

        real FedML quantized vectors
            ->
        first vector_dim coordinates
            ->
        DMCFE no-dropout aggregation
            ->
        compare with plaintext integer sum

    Important:
        y_i = 1 in this checkpoint.

    We are NOT yet testing sample-weighted FedAvg.
    """
    total_start = time.perf_counter()
    logging.info("")

    logging.info(
        "========== Real FedML %s-D Quantized Delta DMCFE Smoke ==========",
        vector_dim,
    )

    # ========================================================
    # 1. Input validation
    # ========================================================

    if not client_vectors:
        raise ValueError(
            "client_vectors must not be empty"
        )

    if isinstance(vector_dim, bool) or not isinstance(vector_dim, int) or vector_dim < 1:
        raise ValueError(
            "vector_dim must be a positive integer"
        )

    client_vectors = [
        vector.detach().cpu().clone()
        for vector in client_vectors
    ]

    n = len(client_vectors)
    # ========================================================
    # REAL FedAvg aggregation weights
    #
    # y_i = number of local training samples
    # ========================================================

    if client_sample_nums is None:
        raise ValueError(
            "client_sample_nums is required "
            "for weighted FedAvg DMCFE"
        )

    if len(client_sample_nums) != n:
        raise ValueError(
            "client_sample_nums length mismatch"
        )

    if any(
        isinstance(sample_num, bool)
        or int(sample_num) != sample_num
        or int(sample_num) <= 0
        for sample_num in client_sample_nums
    ):
        raise ValueError(
            "all client sample numbers "
            "must be positive"
        )
    weights = [int(sample_num) for sample_num in client_sample_nums]

    full_dim = int(
        client_vectors[0].numel()
    )

    if vector_dim > full_dim:
        raise ValueError(
            f"vector_dim={vector_dim} "
            f"exceeds real vector length={full_dim}"
        )

    for index, vector in enumerate(
        client_vectors
    ):

        if vector.dtype != torch.int64:
            raise TypeError(
                f"client {index} vector must be int64, "
                f"got {vector.dtype}"
            )

        if vector.ndim != 1:
            raise ValueError(
                f"client {index} vector must be 1-D"
            )

        if vector.numel() != full_dim:
            raise ValueError(
                "real client vector lengths do not match"
            )

    if sampled_client_ids is None:

        sampled_client_ids = list(
            range(n)
        )

    sampled_client_ids = [
        int(client_id)
        for client_id
        in sampled_client_ids
    ]

    if len(sampled_client_ids) != n:
        raise ValueError(
            "sampled_client_ids length mismatch"
        )
    if len(set(sampled_client_ids)) != n:
        raise ValueError(
            "sampled_client_ids must be unique"
        )

    sampled_set = set(sampled_client_ids)
    early_ids = set(int(value) for value in (dropout_before_encryption_ids or ()))
    late_ids = set(int(value) for value in (dropout_before_recovery_ids or ()))
    if not early_ids.issubset(sampled_set) or not late_ids.issubset(sampled_set - early_ids):
        raise ValueError("dropout IDs must be disjoint subsets of sampled clients")
    u2 = set(range(n))
    early = {sampled_client_ids.index(value) for value in early_ids}
    late = {sampled_client_ids.index(value) for value in late_ids}
    u3 = u2 - early
    u4 = u3 - late
    threshold = int(recovery_threshold) if recovery_threshold is not None else max(1, (2 * n + 2) // 3)
    threshold = min(threshold, max(1, n - 1))
    if len(u4) < threshold:
        raise RuntimeError("DMCFE round aborted: recovery threshold is not met")
    total_sample_num = sum(weights[index] for index in u3)

    # ========================================================
    # 2. Take REAL first vector_dim coordinates
    # ========================================================

    real_vectors = [
        vector[:vector_dim].clone()
        for vector in client_vectors
    ]

    # ========================================================
    # REAL FedAvg weighted integer reference
    #
    # reference =
    #
    # sum_i n_i * Q(delta_i)
    #
    # IMPORTANT:
    # We are still in quantized integer domain.
    # Do NOT divide by total_sample_num yet.
    # ========================================================

    plaintext_reference = torch.zeros(
        vector_dim,
        dtype=torch.int64,
    )

    for index, (vector, weight) in enumerate(zip(real_vectors, weights)):
        if index not in u3:
            continue

        plaintext_reference += (
            vector
            *
            weight
        )

    # ========================================================
    # 3. One REAL MCFE public setup
    # ========================================================

    setup_start = time.perf_counter()

    mcfe = MCFE(
        modulus_bits=256
    )

    setup_seconds = (
        time.perf_counter()
        -
        setup_start
    )

    # ========================================================
    # 4. DMCFE client state
    #
    # This checkpoint only needs:
    #
    #   s_i
    #   eta_i
    #   sk_i^(1), pk_i^(1)
    #   k_i,j^(1)
    #
    # No Shamir/AES is needed because there is no dropout.
    # ========================================================
    key_setup_start = time.perf_counter()
    states = {}

    for slot_id in range(n):

        sk1 = (
            x25519
            .X25519PrivateKey
            .generate()
        )
        sk2 = x25519.X25519PrivateKey.generate()

        states[slot_id] = {

            "secret_key": (
                mcfe.keygen()
            ),

            "eta": (
                secrets.randbits(
                    128
                )
            ),

            "sk1": sk1,

            "pk1": (
                sk1.public_key()
            ),
            "sk2": sk2,
            "pk2": sk2.public_key(),

            "k1": {},
            "k2": {},
        }

    # ========================================================
    # 5. X25519 pairwise k^(1)
    # ========================================================

    pairwise_ka_correct = True

    for i in range(n):

        for j in range(
            i + 1,
            n,
        ):

            key_ij = (
                _derive_pairwise_key(
                    private_key=(
                        states[i]["sk1"]
                    ),
                    peer_public_key=(
                        states[j]["pk1"]
                    ),
                    info=K1_INFO,
                )
            )

            key_ji = (
                _derive_pairwise_key(
                    private_key=(
                        states[j]["sk1"]
                    ),
                    peer_public_key=(
                        states[i]["pk1"]
                    ),
                    info=K1_INFO,
                )
            )

            if key_ij != key_ji:

                pairwise_ka_correct = (
                    False
                )

            states[i]["k1"][j] = (
                key_ij
            )

            states[j]["k1"][i] = (
                key_ji
            )
            states[i]["k2"][j] = _derive_pairwise_key(
                states[i]["sk2"], states[j]["pk2"], b"DMCFE-IP/k2/v1"
            )
            states[j]["k2"][i] = _derive_pairwise_key(
                states[j]["sk2"], states[i]["pk2"], b"DMCFE-IP/k2/v1"
            )
    
    key_setup_seconds = (
        time.perf_counter()
        -
        key_setup_start
    )    
    # Round 1: Shamir shares of sk_i^(1) and eta_i.  The single-process
    # simulator stores the encrypted transport payloads as share maps; the
    # distributed transport uses the same share primitives with AES-GCM.
    ss = ShamirSecretSharing()
    recovery_shares = {}
    share_encryption = SymmetricEncryption()
    for dealer in sorted(u2):
        recipients = [client for client in sorted(u2) if client != dealer]
        if not recipients:
            continue
        sk_shares = ss.share_bytes(
            threshold=threshold,
            participant_ids=recipients,
            secret=_private_key_to_raw(states[dealer]["sk1"]),
        )
        eta_shares = ss.share(
            threshold=threshold,
            participant_ids=recipients,
            secret=states[dealer]["eta"],
        )
        recovery_shares[dealer] = {}
        for recipient in recipients:
            recovery_shares[dealer][recipient] = share_encryption.encrypt_share_packet(
                key=states[dealer]["k2"][recipient],
                sender_id=dealer,
                receiver_id=recipient,
                sk_share=sk_shares[recipient],
                eta_share=eta_shares[recipient],
            )
    # ========================================================
    # 6. DMCFE partial decryption keys
    #
    # y_i = 1
    #
    # dk_i =
    #
    # s_i
    # +
    # PRG(eta_i)
    # +
    # pairwise_mask_i
    # ========================================================
    partial_dk_start = (
        time.perf_counter()
    )
    partial_dks = []

    eta_masks = []

    pair_masks = []


    for i in sorted(u3):

        pair_mask_dot_y = 0

        for j in sorted(u2):
            if j == i:
                continue
            key = states[i]["k1"][j]

            # ====================================================
            # Paper:
            #
            # PRG(k_i,j^(1)) · y
            #
            # Same value for both clients i and j.
            # ====================================================

            mask_dot_y = (
                _pair_prg_dot_weights(
                    key=key,
                    weights=weights,
                )
            )

            if i < j:

                pair_mask_dot_y += (
                    mask_dot_y
                )

            else:

                pair_mask_dot_y -= (
                    mask_dot_y
                )


        eta_mask = (
            _eta_prg(
                states[i]["eta"]
            )
        )


        # ========================================================
        # REAL weighted DMCFE partial key:
        #
        # dk_i,y =
        #
        # s_i * y_i
        # +
        # pairwise_mask_i · y
        # +
        # PRG(eta_i)
        # ========================================================

        partial_dk = (
            states[i]["secret_key"]
            *
            weights[i]
            +
            pair_mask_dot_y
            +
            eta_mask
        )


        pair_masks.append(
            pair_mask_dot_y
        )

        eta_masks.append(
            eta_mask
        )

        partial_dks.append(partial_dk)
    partial_dk_seconds = (
        time.perf_counter()
        -
        partial_dk_start
    )
    # ========================================================
    # 7. Round 3 recovery.  Pairwise masks among U3 cancel; masks from an
    # early dropout remain and are recomputed from its Shamir-recovered sk1.
    # ========================================================

    total_pair_mask = sum(
        pair_masks
    )

    pair_masks_cancel = total_pair_mask == 0 if not early else True
    recovered_sk1 = {}
    for dropped in sorted(early):
        available = {}
        for responder in sorted(u4):
            if responder not in recovery_shares[dropped]:
                continue
            packet = share_encryption.decrypt_share_packet(
                key=states[responder]["k2"][dropped],
                packet=recovery_shares[dropped][responder],
            )
            available[responder] = packet.sk_share
        raw = ss.combine_bytes(threshold, available, 32)
        recovered_sk1[dropped] = x25519.X25519PrivateKey.from_private_bytes(raw)

    residual_pair_mask = 0
    for dropped in sorted(early):
        for active in sorted(u3):
            key = _derive_pairwise_key(
                recovered_sk1[dropped], states[active]["pk1"], K1_INFO
            )
            mask = _pair_prg_dot_weights(key, weights)
            residual_pair_mask += mask if active < dropped else -mask
    residual_mask_correct = total_pair_mask == residual_pair_mask

    recovered_eta = {active: states[active]["eta"] for active in sorted(u4)}
    for dropped in sorted(late):
        available = {}
        for responder in sorted(u4):
            if responder not in recovery_shares[dropped]:
                continue
            packet = share_encryption.decrypt_share_packet(
                key=states[responder]["k2"][dropped],
                packet=recovery_shares[dropped][responder],
            )
            available[responder] = packet.eta_share
        recovered_eta[dropped] = ss.combine(threshold, available)

    eta_mask_sum = sum(_eta_prg(recovered_eta[active]) for active in sorted(u3))

    # ========================================================
    # 8. Recover DMCFE dk_y
    # ========================================================

    recovered_dk = (
        sum(partial_dks) - eta_mask_sum - residual_pair_mask
    )

    secret_keys = [
        states[i]["secret_key"]
        for i in sorted(u3)
    ]

    

    reference_dk = (
        mcfe.derive_key(
            secret_keys=secret_keys,
            weights=[weights[i] for i in sorted(u3)],
        )
    )

    dk_correct = (
        recovered_dk
        ==
        reference_dk
    )

    # ========================================================
    # 9. Encrypt REAL quantized coordinates
    #
    # Same recovered dk
    # Different label per coordinate
    # ========================================================

    # ========================================================
    # 9. Encrypt REAL quantized coordinates
    # ========================================================

    ciphertexts_by_coordinate = []

    total_ciphertext_count = 0

    encryption_start = (
        time.perf_counter()
    )

    for coordinate in range(
        vector_dim
    ):

        label = (
            f"fedml-real-quantized/"
            f"round-{round_idx}/"
            f"coord-{coordinate}"
        ).encode(
            "utf-8"
        )

        coordinate_ciphertexts = []

        for i in sorted(u3):

            message = int(
                real_vectors[i][
                    coordinate
                ].item()
            )

            ciphertext = (
                mcfe.encrypt(
                    secret_key=(
                        states[i][
                            "secret_key"
                        ]
                    ),
                    message=message,
                    label=label,
                )
            )

            coordinate_ciphertexts.append(ciphertext)

            total_ciphertext_count += 1

        ciphertexts_by_coordinate.append(
            coordinate_ciphertexts
        )


    encryption_seconds = (
        time.perf_counter()
        -
        encryption_start
    )

    # ========================================================
    # 10. Decrypt REAL aggregated coordinates
    # ========================================================

    decrypted_values = []

    coordinate_correctness = []

    decryption_start = (
        time.perf_counter()
    )

    for coordinate in range(
        vector_dim
    ):

        label = (
            f"fedml-real-quantized/"
            f"round-{round_idx}/"
            f"coord-{coordinate}"
        ).encode(
            "utf-8"
        )

        decrypted = (
            mcfe.decrypt(
                ciphertexts=(
                    ciphertexts_by_coordinate[
                        coordinate
                    ]
                ),
                weights=[weights[i] for i in sorted(u3)],
                derived_key=(
                    recovered_dk
                ),
                label=label,
            )
        )

        expected = int(
            plaintext_reference[
                coordinate
            ].item()
        )

        decrypted_values.append(
            decrypted
        )

        coordinate_correctness.append(
            decrypted == expected
        )


    decryption_seconds = (
        time.perf_counter()
        -
        decryption_start
    )

    decrypted_vector = torch.tensor(
        decrypted_values,
        dtype=torch.int64,
    )
    normalized_quantized_average = (
        decrypted_vector
        .to(torch.float64)
        /
        float(total_sample_num)
    )

    vector_correct = torch.equal(
        decrypted_vector,
        plaintext_reference,
    )
    integer_error = (
        decrypted_vector.to(torch.int64)
        - plaintext_reference.to(torch.int64)
    ).abs()
    max_integer_error = int(integer_error.max().item()) if integer_error.numel() else 0
    mean_integer_error = float(integer_error.to(torch.float64).mean().item()) if integer_error.numel() else 0.0

    correct_coordinate_count = sum(
        1
        for result
        in coordinate_correctness
        if result
    )

    coordinate_count_correct = (
        correct_coordinate_count
        ==
        vector_dim
    )

    expected_ciphertext_count = len(u3) * vector_dim
    # ========================================================
    # Performance statistics
    # ========================================================

    encryptions_per_second = (
        total_ciphertext_count
        /
        encryption_seconds
        if encryption_seconds > 0
        else float("inf")
    )

    decryptions_per_second = (
        vector_dim
        /
        decryption_seconds
        if decryption_seconds > 0
        else float("inf")
    )

    average_encrypt_ms = (
        1000.0
        * encryption_seconds
        /
        total_ciphertext_count
    )

    average_decrypt_ms = (
        1000.0
        * decryption_seconds
        /
        vector_dim
    )

    # Fixed-width ciphertext size if serialized
    # according to modulus N^2.
    ciphertext_bytes = (
        mcfe.N2.bit_length()
        + 7
    ) // 8

    estimated_ciphertext_payload_bytes = (
        ciphertext_bytes
        *
        total_ciphertext_count
    )

    ciphertext_count_correct = (
        total_ciphertext_count
        ==
        expected_ciphertext_count
    )

    # ========================================================
    # 10. Logging
    # ========================================================

    preview_size = min(
        8,
        vector_dim,
    )
    logging.info(
        "FedAvg client sample weights: %s",
        {
            sampled_client_ids[i]:
            weights[i]

            for i in range(n)
        },
    )

    logging.info(
        "total sampled training examples: %s",
        total_sample_num,
    )
    logging.info(
        "round idx: %s",
        round_idx,
    )

    logging.info(
        "sampled FedML client ids: %s",
        sampled_client_ids,
    )

    logging.info(
        "real client count: %s",
        n,
    )

    logging.info(
        "full real quantized dimension: %s",
        full_dim,
    )

    logging.info(
        "DMCFE tested dimension: %s",
        vector_dim,
    )

    logging.info(
        "input dtype correctness: %s",
        all(
            vector.dtype == torch.int64
            for vector in client_vectors
        ),
    )

    logging.info(
        "real client previews: %s",
        {
            sampled_client_ids[i]:
            real_vectors[i][
                :preview_size
            ].tolist()

            for i in range(n)
        },
    )

    logging.info(
        "pairwise KA correctness: %s",
        pairwise_ka_correct,
    )

    logging.info(
        "pair mask sum: %s",
        total_pair_mask,
    )

    logging.info(
        "pair masks cancel: %s",
        pair_masks_cancel,
    )

    logging.info(
        "recovered dk: %s",
        recovered_dk,
    )

    logging.info(
        "reference dk: %s",
        reference_dk,
    )

    logging.info(
        "dk correctness: %s",
        dk_correct,
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
        "weighted plaintext reference preview: %s",
        plaintext_reference[
            :preview_size
        ].tolist(),
    )
    logging.info(
        "DMCFE weighted decrypted preview: %s",
        decrypted_vector[
            :preview_size
        ].tolist(),
    )
    logging.info(
        "DMCFE decrypted preview: %s",
        decrypted_vector[
            :preview_size
        ].tolist(),
    )
    logging.info(
        "normalized weighted quantized preview: %s",
        normalized_quantized_average[
            :preview_size
        ].tolist(),
    )
    logging.info(
        "correct coordinates: %s/%s",
        correct_coordinate_count,
        vector_dim,
    )
    logging.info(
        "MCFE setup time: %.6f s",
        setup_seconds,
    )

    logging.info(
        "key setup + pairwise KA time: %.6f s",
        key_setup_seconds,
    )

    logging.info(
        "partial dk generation time: %.6f s",
        partial_dk_seconds,
    )

    logging.info(
        "encryption time: %.6f s",
        encryption_seconds,
    )

    logging.info(
        "decryption time: %.6f s",
        decryption_seconds,
    )

    logging.info(
        "average encryption time: %.6f ms/ciphertext",
        average_encrypt_ms,
    )

    logging.info(
        "average decryption time: %.6f ms/coordinate",
        average_decrypt_ms,
    )

    logging.info(
        "encryption throughput: %.2f ciphertexts/s",
        encryptions_per_second,
    )

    logging.info(
        "decryption throughput: %.2f coordinates/s",
        decryptions_per_second,
    )

    logging.info(
        "estimated serialized ciphertext bytes each: %s",
        ciphertext_bytes,
    )

    logging.info(
        "estimated total ciphertext payload: %.3f MB",
        (
            estimated_ciphertext_payload_bytes
            /
            1024
            /
            1024
        ),
    )
    
    logging.info(
        "real quantized vector correctness: %s",
        vector_correct,
    )

    overall_correct = all(
        [
            pairwise_ka_correct,
            pair_masks_cancel,
            residual_mask_correct,
            dk_correct,
            ciphertext_count_correct,
            coordinate_count_correct,
            vector_correct,
        ]
    )
    total_seconds = (
        time.perf_counter()
        -
        total_start
    )
    logging.info(
        "real-delta DMCFE total smoke time: %.6f s",
        total_seconds,
    )
    logging.info(
        "REAL FedML quantized-delta "
        "DMCFE overall correctness: %s",
        overall_correct,
    )

    logging.info(
        "==============================================================="
    )

    logging.info("")

    # ========================================================
    # 11. Hard assertions
    # ========================================================

    assert pairwise_ka_correct
    assert pair_masks_cancel
    assert residual_mask_correct
    assert dk_correct
    assert ciphertext_count_correct
    assert coordinate_count_correct
    assert vector_correct
    assert overall_correct

    return {
        "correct": overall_correct,
        "decrypted_vector": decrypted_vector,
        "normalized_vector": normalized_quantized_average,
        "vector_dim": vector_dim,
        "full_dim": full_dim,
        "total_sample_num": total_sample_num,
        "active_client_count": len(u3),
        "active_client_ids": [sampled_client_ids[i] for i in sorted(u3)],
        "dropout_before_encryption_ids": sorted(early_ids),
        "dropout_before_recovery_ids": sorted(late_ids),
        "recovery_threshold": threshold,
        "max_integer_error": max_integer_error,
        "mean_integer_error": mean_integer_error,
        "ciphertext_count": total_ciphertext_count,
        "estimated_ciphertext_payload_bytes": estimated_ciphertext_payload_bytes,
        "timings": {
            "setup_seconds": setup_seconds,
            "key_setup_seconds": key_setup_seconds,
            "partial_dk_seconds": partial_dk_seconds,
            "encryption_seconds": encryption_seconds,
            "decryption_seconds": decryption_seconds,
            "total_seconds": total_seconds,
        },
    }
