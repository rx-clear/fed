import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


FEDREP_DIR = (
    Path(__file__).parents[1] / "fedml" / "simulation" / "sp" / "fedrep"
)


def _load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, FEDREP_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PARAMETERS = _load_module("fedrep_parameters_under_test", "parameters.py")
CLIENT = _load_module("fedrep_client_under_test", "client.py")


class TwoPartModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.representation = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 1)

    def forward(self, value):
        return self.head(self.representation(value))


class FakeTrainer:
    def __init__(self, model):
        self.model = model
        self.id = None
        self.phase_calls = []

    def set_id(self, client_idx):
        self.id = client_idx

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, state):
        self.model.load_state_dict(state)

    def train(self, _data, _device, args):
        trainable = []
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.add_(float(args.epochs))
                    trainable.append(name)
        self.phase_calls.append((args.epochs, trainable))

    def test(self, _data, _device, _args):
        return {"test_total": 0, "test_correct": 0, "test_loss": 0}


class FedRepParameterTest(unittest.TestCase):
    def test_explicit_head_partition(self):
        model = TwoPartModel()
        shared, personal, shared_params, personal_params = (
            PARAMETERS.resolve_fedrep_parameter_keys(
                model, model.state_dict(), ["head"]
            )
        )
        self.assertEqual(shared, ["representation.weight", "representation.bias"])
        self.assertEqual(personal, ["head.weight", "head.bias"])
        self.assertEqual(shared_params, shared)
        self.assertEqual(personal_params, personal)

    def test_default_uses_last_parameterized_module(self):
        model = TwoPartModel()
        _, personal, _, _ = PARAMETERS.resolve_fedrep_parameter_keys(
            model, model.state_dict()
        )
        self.assertEqual(personal, ["head.weight", "head.bias"])

    def test_single_layer_model_is_rejected(self):
        model = torch.nn.Linear(2, 1)
        with self.assertRaisesRegex(ValueError, "shared representation"):
            PARAMETERS.resolve_fedrep_parameter_keys(model, model.state_dict())


class FedRepClientTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = TwoPartModel()
        self.trainer = FakeTrainer(self.model)
        self.initial = {
            key: value.detach().clone() for key, value in self.model.state_dict().items()
        }
        self.personalized = {}
        self.args = SimpleNamespace(
            epochs=3,
            fedrep_head_epochs=2,
            fedrep_representation_epochs=3,
        )
        self.client = CLIENT.FedRepClient(
            0,
            [],
            [],
            1,
            self.args,
            torch.device("cpu"),
            self.trainer,
            ["representation.weight", "representation.bias"],
            ["head.weight", "head.bias"],
            ["head.weight", "head.bias"],
            self.personalized,
            {key: self.initial[key] for key in ("head.weight", "head.bias")},
        )

    def test_two_phase_training_and_client_head_persistence(self):
        first = self.client.train(self.initial)
        self.assertTrue(
            torch.allclose(first["representation.weight"], self.initial["representation.weight"] + 3)
        )
        self.assertTrue(torch.allclose(first["head.weight"], self.initial["head.weight"] + 2))

        next_global = {key: value.detach().clone() for key, value in self.initial.items()}
        next_global["representation.weight"].fill_(10)
        self.client.update_local_dataset(1, [], [], 1)
        second_client = self.client.train(next_global)
        self.assertTrue(torch.allclose(second_client["head.weight"], self.initial["head.weight"] + 2))

        self.client.update_local_dataset(0, [], [], 1)
        revisited = self.client.train(next_global)
        self.assertTrue(torch.allclose(revisited["head.weight"], self.initial["head.weight"] + 4))
        self.assertTrue(torch.allclose(revisited["representation.weight"], torch.full_like(revisited["representation.weight"], 13)))
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
