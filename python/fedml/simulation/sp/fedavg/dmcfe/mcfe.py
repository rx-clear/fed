import hashlib
import math
import secrets
from dataclasses import dataclass

from sympy import randprime


@dataclass(frozen=True)
class EncryptionContext:
    """Precomputed label/secret-key term for one DMCFE encryption domain."""

    label: object
    secret_key: int
    mask_part: int


class MCFE:

    def __init__(self, modulus_bits=256):
        """
        仅用于正确性 smoke test。

        正式论文实验后续改成 2048-bit。
        """

        self.modulus_bits = modulus_bits

        self.N = None
        self.N2 = None

        self._encryption_cache = {}
        self._decryption_cache = {}

        self.setup()


    # ============================================================
    # Setup
    # ============================================================

    def setup(self):

        half_bits = self.modulus_bits // 2

        lower = 1 << (half_bits - 1)
        upper = 1 << half_bits

        p = int(
            randprime(lower, upper)
        )

        q = int(
            randprime(lower, upper)
        )

        while q == p:

            q = int(
                randprime(lower, upper)
            )


        self.N = p * q

        self.N2 = self.N * self.N


    # ============================================================
    # KeyGen
    # ============================================================

    def keygen(self, bound=1_000_000):
        """
        正确性测试阶段使用随机有符号整数。

        正式安全实现后续再替换为论文要求的密钥分布。
        """

        return (
            secrets.randbelow(
                2 * bound + 1
            )
            -
            bound
        )


    # ============================================================
    # Hash -> Z*_(N^2)
    # ============================================================

    def hash_to_group(self, label):

        if isinstance(label, str):

            label = label.encode("utf-8")


        output_bytes = (
            self.N2.bit_length() + 7
        ) // 8


        counter = 0


        while True:

            counter_bytes = counter.to_bytes(
                8,
                byteorder="big"
            )


            digest = hashlib.shake_256(
                label + counter_bytes
            ).digest(
                output_bytes + 16
            )


            h = int.from_bytes(
                digest,
                byteorder="big"
            )


            h %= self.N2


            if (
                h > 1
                and
                math.gcd(h, self.N2) == 1
            ):

                return h


            counter += 1


    # ============================================================
    # signed modular exponentiation
    # ============================================================

    @staticmethod
    def pow_signed(
        base,
        exponent,
        modulus
    ):

        exponent = int(exponent)


        if exponent >= 0:

            return pow(
                base,
                exponent,
                modulus
            )


        inverse = pow(
            base,
            -1,
            modulus
        )


        return pow(
            inverse,
            -exponent,
            modulus
        )


    # ============================================================
    # Enc
    #
    # ct_i =
    #
    # (1 + N)^x_i
    # *
    # H(label)^s_i
    #
    # mod N^2
    # ============================================================

    def encrypt(
        self,
        secret_key,
        message,
        label
    ):

        message = int(message)
        secret_key = int(secret_key)


        # 利用：
        #
        # (1 + N)^x
        # =
        # 1 + xN mod N²

        context = self.prepare_encryption(secret_key, label)
        return self.encrypt_prepared(context, message)

    def prepare_encryption(self, secret_key, label) -> EncryptionContext:
        """Precompute ``H(label)^secret_key`` for repeated same-label encryptions."""
        if isinstance(label, bytearray):
            label = bytes(label)
        cache_key = (int(secret_key), label if isinstance(label, (bytes, str)) else repr(label))
        context = self._encryption_cache.get(cache_key)
        if context is None:
            h = self.hash_to_group(label)
            context = EncryptionContext(
                label=label,
                secret_key=int(secret_key),
                mask_part=self.pow_signed(h, int(secret_key), self.N2),
            )
            self._encryption_cache[cache_key] = context
        return context

    def encrypt_prepared(self, context: EncryptionContext, message) -> int:
        """Encrypt with a context returned by :meth:`prepare_encryption`."""
        message_part = (1 + int(message) * self.N) % self.N2
        return (message_part * int(context.mask_part)) % self.N2


    # ============================================================
    # KeyDer
    #
    # dk_y = Σ s_i * y_i
    # ============================================================

    @staticmethod
    def derive_key(
        secret_keys,
        weights
    ):

        if len(secret_keys) != len(weights):

            raise ValueError(
                "secret_keys and weights length mismatch"
            )


        dk = 0


        for secret_key, weight in zip(
            secret_keys,
            weights
        ):

            dk += (
                int(secret_key)
                *
                int(weight)
            )


        return dk


    # ============================================================
    # Dec
    # ============================================================

    def decrypt(
        self,
        ciphertexts,
        weights,
        derived_key,
        label
    ):

        if len(ciphertexts) != len(weights):

            raise ValueError(
                "ciphertexts and weights length mismatch"
            )


        result = 1


        # Π ct_i ^ y_i

        for ciphertext, weight in zip(
            ciphertexts,
            weights
        ):

            result *= self.pow_signed(
                ciphertext,
                weight,
                self.N2
            )

            result %= self.N2


        # H(label)^(-dk_y)

        cancel_term = self.prepare_decryption(derived_key, label)


        result *= cancel_term

        result %= self.N2


        # ========================================================
        # L function
        # ========================================================

        numerator = (
            result - 1
        ) % self.N2


        if numerator % self.N != 0:

            raise RuntimeError(
                "MCFE decryption consistency check failed"
            )


        message = (
            numerator // self.N
        ) % self.N


        # ========================================================
        # signed decoding
        # ========================================================

        if message > self.N // 2:

            message -= self.N


        return int(message)

    def prepare_decryption(self, derived_key, label) -> int:
        """Precompute the inverse functional-key term for repeated labels."""
        if isinstance(label, bytearray):
            label = bytes(label)
        cache_key = (int(derived_key), label if isinstance(label, (bytes, str)) else repr(label))
        cached = self._decryption_cache.get(cache_key)
        if cached is None:
            h = self.hash_to_group(label)
            cached = self.pow_signed(h, -int(derived_key), self.N2)
            self._decryption_cache[cache_key] = cached
        return cached
