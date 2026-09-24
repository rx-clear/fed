import unittest

import torch

from fedml.simulation.sp.fedavg.dmcfe.mcfe import MCFE
from fedml.simulation.sp.fedavg.dmcfe.plaintext_packing import (
    PackedLayout,
    pack_signed,
    quantize_fixed_point,
    unpack_aggregate,
    unpack_signed,
)
from fedml.simulation.sp.fedavg.dmcfe.weight_signatures import (
    WeightSigner,
    verify_weight_proposals,
)
from fedml.simulation.sp.fedavg.dmcfe.real_delta_smoke import (
    run_real_quantized_delta_smoke,
)
from fedml.simulation.sp.fedavg.privacy_baselines import (
    BatchCryptPrototype,
    PairwiseMaskingAggregator,
)


class DMCFEPaperFeaturesTest(unittest.TestCase):
    def test_weight_signatures_bind_round_and_participants(self):
        signers = {client_id: WeightSigner() for client_id in (2, 7, 11)}
        ids = [2, 7, 11]
        proposals = [signers[i].propose(3, ids, i, i + 10) for i in ids]
        keys = {i: signers[i].public_key_bytes for i in ids}
        self.assertEqual(verify_weight_proposals(proposals, keys, 3, ids), (12, 17, 21))
        with self.assertRaises(ValueError):
            verify_weight_proposals(proposals, keys, 4, ids)

    def test_signed_packing_and_aggregate_decoding(self):
        layout = PackedLayout.for_clients(value_bits=8, number_of_clients=3, modulus_bits=64)
        values = [-5, 0, 12]
        self.assertEqual(unpack_signed(pack_signed(values, layout), layout, len(values)), values)
        packed_sum = sum(pack_signed([value, 0, 0], layout) for value in (2, -3, 4))
        self.assertEqual(unpack_aggregate(packed_sum, layout, 3, 1), [3])
        self.assertEqual(quantize_fixed_point([0.5, -1.25], 4), [8, -20])

    def test_mcfe_precomputation_matches_regular_api(self):
        mcfe = MCFE(modulus_bits=96)
        secret = mcfe.keygen(100)
        label = "paper-round-0-coordinate-0"
        context = mcfe.prepare_encryption(secret, label)
        self.assertEqual(mcfe.encrypt(secret, 19, label), mcfe.encrypt_prepared(context, 19))
        derived = mcfe.derive_key([secret], [1])
        ciphertext = mcfe.encrypt_prepared(context, 19)
        self.assertEqual(mcfe.decrypt([ciphertext], [1], derived, label), 19)

    def test_real_delta_path_recovers_early_and_late_dropouts(self):
        vectors = [
            torch.tensor([2, 1, 0], dtype=torch.int64),
            torch.tensor([3, -1, 4], dtype=torch.int64),
            torch.tensor([-2, 5, 1], dtype=torch.int64),
            torch.tensor([7, 2, -3], dtype=torch.int64),
        ]
        result = run_real_quantized_delta_smoke(
            vectors,
            [2, 3, 4, 5],
            round_idx=0,
            vector_dim=3,
            sampled_client_ids=[10, 11, 12, 13],
            dropout_before_encryption_ids=[13],
            dropout_before_recovery_ids=[12],
            recovery_threshold=2,
        )
        expected = vectors[0] * 2 + vectors[1] * 3 + vectors[2] * 4
        self.assertTrue(torch.equal(result["decrypted_vector"], expected))
        self.assertEqual(result["active_client_ids"], [10, 11, 12])

    def test_batchcrypt_and_masking_preserve_weighted_average(self):
        local = [
            (2, {"w": torch.tensor([0.12, -0.20])}),
            (3, {"w": torch.tensor([0.50, 0.10])}),
        ]
        expected = torch.tensor([0.348, -0.020])
        batchcrypt = BatchCryptPrototype(precision=8, modulus_bits=128).aggregate(local)["w"]
        masking = PairwiseMaskingAggregator(seed=7).aggregate(local)["w"]
        self.assertTrue(torch.allclose(batchcrypt.float(), expected, atol=0.01))
        self.assertTrue(torch.allclose(masking.float(), expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
