from cryptography.hazmat.primitives import hashes

from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
)

from cryptography.hazmat.primitives.kdf.hkdf import HKDF


def generate_keypair():
    """
    KA.gen()
    """

    private_key = X25519PrivateKey.generate()

    public_key = private_key.public_key()

    return private_key, public_key


def derive_shared_key(
    private_key,
    peer_public_key,
    info=b"dmcfe-key-agreement",
):
    """
    KA.agree()

    X25519:
        private_i + public_j
            ->
        raw shared secret

    再通过 HKDF 得到32字节密钥。
    """

    raw_shared = private_key.exchange(
        peer_public_key
    )

    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=info,
    ).derive(
        raw_shared
    )

    return derived_key