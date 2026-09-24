from __future__ import annotations

import secrets
from typing import Dict, Iterable, Mapping


# ============================================================
# Shamir Secret Sharing
# ============================================================
#
# We need a finite-field prime larger than:
#
#   1. eta_i
#   2. raw X25519 private key (32 bytes = 256 bits)
#
# 2^521 - 1 is large enough for our current DMCFE prototype.
#
# NOTE:
# This is currently a correctness-oriented implementation.
# It matches the Shamir SS structure required by the paper.
# ============================================================

FIELD_PRIME = (1 << 521) - 1


class ShamirSecretSharing:
    """
    Shamir (t, n) secret sharing used by DMCFE-IP.

    Paper mapping:

        SS.Share(t, P, s)
            ->
        share(threshold, participant_ids, secret)

        SS.Comb(t, shares)
            ->
        combine(threshold, shares)

    Each FedML / DMCFE client ID is mapped to a non-zero
    finite-field point:

        x = client_id + 1

    because x = 0 is reserved for the secret:

        f(0) = secret
    """

    def __init__(self, prime: int = FIELD_PRIME):
        if prime <= 2:
            raise ValueError("prime must be > 2")

        self.prime = int(prime)

    # ========================================================
    # Participant handling
    # ========================================================

    @staticmethod
    def _validate_participant_ids(
        participant_ids: Iterable[int],
    ) -> list[int]:

        ids = [int(pid) for pid in participant_ids]

        if not ids:
            raise ValueError(
                "participant_ids must not be empty"
            )

        if len(ids) != len(set(ids)):
            raise ValueError(
                "participant_ids must be unique"
            )

        if any(pid < 0 for pid in ids):
            raise ValueError(
                "participant IDs must be non-negative"
            )

        return ids

    def _x_for_participant(
        self,
        participant_id: int,
    ) -> int:

        # client 0 -> x = 1
        # client 1 -> x = 2
        # ...
        x = int(participant_id) + 1

        if x <= 0 or x >= self.prime:
            raise ValueError(
                "participant ID maps outside the finite field"
            )

        return x

    # ========================================================
    # SS.Share
    # ========================================================

    def share(
        self,
        threshold: int,
        participant_ids: Iterable[int],
        secret: int,
    ) -> Dict[int, int]:
        """
        Split an integer secret into Shamir shares.

        Args:
            threshold:
                Reconstruction threshold t.

            participant_ids:
                Clients receiving the shares.

                In DMCFE Round 1 this should normally be:

                    U1 - {i}

                for dealer client i.

            secret:
                Non-negative integer secret.

        Returns:
            Dict:

                {
                    participant_id: share_value
                }
        """

        ids = self._validate_participant_ids(
            participant_ids
        )

        threshold = int(threshold)
        secret = int(secret)

        # ----------------------------------------------------
        # Validate threshold
        # ----------------------------------------------------

        if threshold < 1:
            raise ValueError(
                "threshold must be >= 1"
            )

        if threshold > len(ids):
            raise ValueError(
                f"threshold={threshold} exceeds "
                f"number of recipients={len(ids)}"
            )

        # ----------------------------------------------------
        # Validate secret
        # ----------------------------------------------------

        if secret < 0:
            raise ValueError(
                "secret must be non-negative"
            )

        if secret >= self.prime:
            raise ValueError(
                "secret must be smaller than the field prime"
            )

        # ====================================================
        # Construct polynomial
        #
        # f(x)
        # =
        # secret
        # + a1*x
        # + a2*x^2
        # + ...
        # + a_(t-1)*x^(t-1)
        #
        # modulo p
        #
        # Therefore:
        #
        # f(0) = secret
        # ====================================================

        coefficients = [secret]

        for _ in range(threshold - 1):

            random_coefficient = secrets.randbelow(
                self.prime
            )

            coefficients.append(
                random_coefficient
            )

        # ====================================================
        # Generate shares
        # ====================================================

        shares: Dict[int, int] = {}

        for participant_id in ids:

            x = self._x_for_participant(
                participant_id
            )

            # -----------------------------------------------
            # Horner polynomial evaluation
            #
            # More efficient than repeatedly calculating x^k
            # -----------------------------------------------

            y = 0

            for coefficient in reversed(
                coefficients
            ):

                y = (
                    y * x + coefficient
                ) % self.prime

            shares[participant_id] = y

        return shares

    # ========================================================
    # SS.Comb
    # ========================================================

    def combine(
        self,
        threshold: int,
        shares: Mapping[int, int],
    ) -> int:
        """
        Reconstruct the secret using at least t shares.

        Args:
            threshold:
                Reconstruction threshold t.

            shares:
                {
                    participant_id: share_value
                }

        Returns:
            Original integer secret.
        """

        threshold = int(threshold)

        if threshold < 1:
            raise ValueError(
                "threshold must be >= 1"
            )

        if len(shares) < threshold:
            raise ValueError(
                f"need at least {threshold} shares, "
                f"got {len(shares)}"
            )

        # ----------------------------------------------------
        # Under the paper's semi-honest model,
        # any t valid shares are sufficient.
        #
        # Sort IDs so reconstruction is deterministic.
        # ----------------------------------------------------

        selected_shares = sorted(
            (
                (
                    int(participant_id),
                    int(share_value) % self.prime,
                )
                for participant_id, share_value
                in shares.items()
            ),
            key=lambda item: item[0],
        )[:threshold]

        participant_ids = [
            participant_id
            for participant_id, _
            in selected_shares
        ]

        if len(participant_ids) != len(
            set(participant_ids)
        ):
            raise ValueError(
                "duplicate participant IDs are not allowed"
            )

        # ====================================================
        # Lagrange interpolation at x = 0
        #
        #
        #             Π (-x_j)
        # λ_i(0) = ----------------
        #           Π (x_i - x_j)
        #
        #
        # secret = f(0)
        #
        #        = Σ y_i * λ_i(0)
        #
        # ====================================================

        secret = 0

        for i, (
            participant_id_i,
            y_i,
        ) in enumerate(selected_shares):

            x_i = self._x_for_participant(
                participant_id_i
            )

            numerator = 1
            denominator = 1

            for j, (
                participant_id_j,
                _,
            ) in enumerate(selected_shares):

                if i == j:
                    continue

                x_j = self._x_for_participant(
                    participant_id_j
                )

                numerator = (
                    numerator * (-x_j)
                ) % self.prime

                denominator = (
                    denominator
                    * (x_i - x_j)
                ) % self.prime

            # Modular inverse:
            #
            # denominator^(-1) mod p

            denominator_inverse = pow(
                denominator,
                -1,
                self.prime,
            )

            lagrange_coefficient = (
                numerator
                * denominator_inverse
            ) % self.prime

            secret = (
                secret
                + y_i * lagrange_coefficient
            ) % self.prime

        return secret

    # ========================================================
    # Byte-secret helpers
    # ========================================================

    def share_bytes(
        self,
        threshold: int,
        participant_ids: Iterable[int],
        secret: bytes,
    ) -> Dict[int, int]:
        """
        Share a fixed-size byte secret.

        Main use in our DMCFE reproduction:

            X25519 private key sk_i^(1)
        """

        if not isinstance(
            secret,
            (bytes, bytearray),
        ):
            raise TypeError(
                "secret must be bytes-like"
            )

        secret_integer = int.from_bytes(
            secret,
            byteorder="big",
            signed=False,
        )

        return self.share(
            threshold=threshold,
            participant_ids=participant_ids,
            secret=secret_integer,
        )

    def combine_bytes(
        self,
        threshold: int,
        shares: Mapping[int, int],
        length: int,
    ) -> bytes:
        """
        Reconstruct a fixed-length byte secret.

        For X25519:

            length = 32
        """

        if length <= 0:
            raise ValueError(
                "length must be > 0"
            )

        secret_integer = self.combine(
            threshold=threshold,
            shares=shares,
        )

        max_value = 1 << (
            8 * length
        )

        if secret_integer >= max_value:
            raise ValueError(
                "reconstructed integer does not fit "
                "in requested byte length"
            )

        return secret_integer.to_bytes(
            length,
            byteorder="big",
            signed=False,
        )