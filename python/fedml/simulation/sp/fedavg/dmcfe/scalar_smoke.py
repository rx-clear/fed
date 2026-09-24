from .mcfe import MCFE


def run_scalar_mcfe_smoke(
    client_values,
    round_idx
):

    print()
    print(
        "========== MCFE Scalar Smoke =========="
    )


    # ============================================================
    # 1. Setup
    # ============================================================

    mcfe = MCFE(
        modulus_bits=256
    )


    client_num = len(
        client_values
    )


    # ============================================================
    # 2. KeyGen
    # ============================================================

    secret_keys = [

        mcfe.keygen()

        for _ in range(
            client_num
        )
    ]


    # ============================================================
    # 3. 第一版只求和
    #
    # y = [1,1,1]
    # ============================================================

    weights = [
        1
        for _ in range(
            client_num
        )
    ]


    # ============================================================
    # 4. 每一轮使用不同 label
    # ============================================================

    label = (
        f"fedml-round-{round_idx}-scalar-0"
    ).encode(
        "utf-8"
    )


    # ============================================================
    # 5. Client Encryption
    # ============================================================

    ciphertexts = []


    for value, secret_key in zip(
        client_values,
        secret_keys
    ):

        ciphertext = mcfe.encrypt(
            secret_key=secret_key,
            message=value,
            label=label
        )


        ciphertexts.append(
            ciphertext
        )


    # ============================================================
    # 6. KeyDer
    # ============================================================

    derived_key = mcfe.derive_key(
        secret_keys,
        weights
    )


    # ============================================================
    # 7. Server Dec
    # ============================================================

    decrypted = mcfe.decrypt(
        ciphertexts=ciphertexts,
        weights=weights,
        derived_key=derived_key,
        label=label
    )


    # ============================================================
    # 8. plaintext reference
    # ============================================================

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
        "MCFE decrypted:",
        decrypted
    )


    success = (
        plaintext_sum
        ==
        decrypted
    )


    print(
        "MCFE correctness:",
        success
    )


    if not success:

        raise RuntimeError(
            "MCFE scalar smoke test failed"
        )


    print(
        "======================================="
    )
    print()


    return decrypted