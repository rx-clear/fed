from dataclasses import dataclass, field


@dataclass
class DMCFEClientState:

    client_id: int

    # MCFE secret key s_i
    secret_key: int

    # random noise eta_i
    eta: bytes


    # ==========================
    # KA key pair 1
    # ==========================

    ka_private_1: object = None

    ka_public_1: object = None


    # ==========================
    # KA key pair 2
    # ==========================

    ka_private_2: object = None

    ka_public_2: object = None


    # ==========================
    # Pairwise shared keys
    # ==========================

    pairwise_keys_1: dict = field(
        default_factory=dict
    )

    pairwise_keys_2: dict = field(
        default_factory=dict
    )