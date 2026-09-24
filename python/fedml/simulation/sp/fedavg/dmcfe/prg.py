import hashlib


def prg_to_signed_int(
    seed,
    label,
    bound=1_000_000
):
    """
    调试版本 PRG。

    输入:
        seed: bytes
        label: bytes

    输出:
        [-bound, bound] 范围整数

    注意：
    这里只用于验证 DMCFE 数学正确性，
    后续正式复现实验再替换密码学 PRG。
    """

    if isinstance(label, str):
        label = label.encode("utf-8")

    digest = hashlib.sha256(
        seed + label
    ).digest()

    value = int.from_bytes(
        digest,
        byteorder="big"
    )

    value %= (
        2 * bound + 1
    )

    value -= bound

    return value