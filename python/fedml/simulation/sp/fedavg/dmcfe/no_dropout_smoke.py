import secrets

from .mcfe import MCFE
from .prg import prg_to_signed_int
from .client_state import DMCFEClientState

from .key_agreement import (
    generate_keypair,
    derive_shared_key,
)

def run_dmcfe_no_dropout_smoke(
    client_values,
    round_idx
):

    print()
    print(
        "========== DMCFE No-Dropout Smoke =========="
    )


    # ============================================================
    # 1. Initialization
    # ============================================================

    mcfe = MCFE(
        modulus_bits=256
    )

    n = len(client_values)

    label = (
        f"dmcfe-round-{round_idx}-scalar-0"
    ).encode("utf-8")


    # ============================================================
    # 2. 每个Client独立生成自己的 s_i 和 eta_i
    # ============================================================

    states = []

    for client_id in range(n):

        private_1, public_1 = (
            generate_keypair()
        )

        private_2, public_2 = (
            generate_keypair()
        )


        state = DMCFEClientState(

            client_id=client_id,

            secret_key=mcfe.keygen(),

            eta=secrets.token_bytes(32),

            ka_private_1=private_1,

            ka_public_1=public_1,

            ka_private_2=private_2,

            ka_public_2=public_2,
        )

        states.append(state)


    # ============================================================
    # 3. 建立 pairwise shared keys
    #
    # k_01
    # k_02
    # k_12
    #
    # 当前只是模拟 Key Agreement。
    # 后续再换成真正 X25519 / KA.agree。
    # ============================================================

    # ============================================================
    # Real Key Agreement
    # ============================================================

    for i in range(n):

        for j in range(
            i + 1,
            n
        ):

            # ====================================================
            # k^(1)_i,j
            # ====================================================

            key_ij_1 = derive_shared_key(
                private_key=states[i].ka_private_1,
                peer_public_key=states[j].ka_public_1,
                info=b"dmcfe-pairwise-key-1",
            )


            key_ji_1 = derive_shared_key(
                private_key=states[j].ka_private_1,
                peer_public_key=states[i].ka_public_1,
                info=b"dmcfe-pairwise-key-1",
            )


            if key_ij_1 != key_ji_1:

                raise RuntimeError(
                    "KA key 1 mismatch"
                )


            states[i].pairwise_keys_1[j] = (
                key_ij_1
            )

            states[j].pairwise_keys_1[i] = (
                key_ji_1
            )


            # ====================================================
            # k^(2)_i,j
            # ====================================================

            key_ij_2 = derive_shared_key(
                private_key=states[i].ka_private_2,
                peer_public_key=states[j].ka_public_2,
                info=b"dmcfe-pairwise-key-2",
            )


            key_ji_2 = derive_shared_key(
                private_key=states[j].ka_private_2,
                peer_public_key=states[i].ka_public_2,
                info=b"dmcfe-pairwise-key-2",
            )


            if key_ij_2 != key_ji_2:

                raise RuntimeError(
                    "KA key 2 mismatch"
                )


            states[i].pairwise_keys_2[j] = (
                key_ij_2
            )

            states[j].pairwise_keys_2[i] = (
                key_ji_2
            )

    # ============================================================
    # 3.1 Key Agreement 验收
    # ============================================================

    print(
        "KA pairwise key 1 established:",
        True
    )

    print(
        "KA pairwise key 2 established:",
        True
    )

    print(
        "Client 0 key1 peers:",
        list(states[0].pairwise_keys_1.keys())
    )

    print(
        "Client 0 key2 peers:",
        list(states[0].pairwise_keys_2.keys())
    )
    # ============================================================
    # 4. Client Encryption + Partial DK
    # ============================================================

    ciphertexts = []

    partial_dks = []

    eta_masks = []

    pair_masks = []


    for state, message in zip(
        states,
        client_values
    ):

        client_id = state.client_id


        # --------------------------------------------------------
        # 4.1 Pairwise mask
        # --------------------------------------------------------

        pair_mask = 0


        for other_id, shared_key in (
            state.pairwise_keys_1.items()
        ):

            mask = prg_to_signed_int(
                shared_key,
                label
            )


            if client_id < other_id:

                pair_mask += mask

            else:

                pair_mask -= mask


        # --------------------------------------------------------
        # 4.2 eta mask
        # --------------------------------------------------------

        eta_mask = prg_to_signed_int(
            state.eta,
            label
        )


        # --------------------------------------------------------
        # 4.3 partial decryption key
        #
        # yi = 1
        # --------------------------------------------------------

        partial_dk = (
            state.secret_key
            +
            pair_mask
            +
            eta_mask
        )


        # --------------------------------------------------------
        # 4.4 Encrypt x_i
        # --------------------------------------------------------

        ciphertext = mcfe.encrypt(
            secret_key=state.secret_key,
            message=message,
            label=label
        )


        ciphertexts.append(
            ciphertext
        )

        partial_dks.append(
            partial_dk
        )

        pair_masks.append(
            pair_mask
        )

        eta_masks.append(
            eta_mask
        )


    # ============================================================
    # 5. 检查 pairwise masks 是否自动抵消
    # ============================================================

    pair_mask_sum = sum(
        pair_masks
    )


    print(
        "pair masks:",
        pair_masks
    )

    print(
        "pair mask sum:",
        pair_mask_sum
    )


    if pair_mask_sum != 0:

        raise RuntimeError(
            "Pairwise masks did not cancel"
        )


    # ============================================================
    # 6. Server聚合 partial DK
    #
    # Round 3:
    #
    # dk =
    # Σ partial_dk
    # -
    # Σ PRG(eta_i)
    #
    # 无掉线时 pairwise masks 已自动消失
    # ============================================================

    aggregated_partial_dk = sum(
        partial_dks
    )


    recovered_dk = (
        aggregated_partial_dk
        -
        sum(eta_masks)
    )


    # ============================================================
    # 7. Debug reference
    #
    # 这里只用于验证。
    # 正式Server不应该拿到各个 s_i。
    # ============================================================

    reference_dk = sum(
        state.secret_key
        for state in states
    )


    print(
        "recovered dk:",
        recovered_dk
    )

    print(
        "reference dk:",
        reference_dk
    )


    if recovered_dk != reference_dk:

        raise RuntimeError(
            "DMCFE partial key recovery failed"
        )


    # ============================================================
    # 8. Server MCFE.Dec
    # ============================================================

    weights = [
        1
        for _ in range(n)
    ]


    decrypted = mcfe.decrypt(
        ciphertexts=ciphertexts,
        weights=weights,
        derived_key=recovered_dk,
        label=label
    )


    plaintext_sum = sum(
        client_values
    )


    print(
        "client values:",
        client_values
    )

    print(
        "plaintext sum:",
        plaintext_sum
    )

    print(
        "DMCFE decrypted:",
        decrypted
    )


    success = (
        decrypted
        ==
        plaintext_sum
    )


    print(
        "DMCFE correctness:",
        success
    )


    if not success:

        raise RuntimeError(
            "DMCFE no-dropout smoke failed"
        )


    print(
        "============================================="
    )
    print()


    return decrypted