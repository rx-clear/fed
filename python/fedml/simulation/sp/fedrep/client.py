"""Single-process FedRep client with persistent personalized heads."""

from __future__ import annotations

import copy


def _clone_state(state):
    return {key: value.detach().clone() for key, value in state.items()}


class FedRepClient:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer,
        shared_parameter_names,
        personalized_parameter_names,
        personalized_state_keys,
        personalized_state_by_client,
        initial_personalized_state,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.shared_parameter_names = set(shared_parameter_names)
        self.personalized_parameter_names = set(personalized_parameter_names)
        self.personalized_state_keys = list(personalized_state_keys)
        self.personalized_state_by_client = personalized_state_by_client
        self.initial_personalized_state = _clone_state(initial_personalized_state)

    def update_local_dataset(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
    ):
        self.client_idx = int(client_idx)
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(self.client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def _personalized_state(self):
        if self.client_idx not in self.personalized_state_by_client:
            self.personalized_state_by_client[self.client_idx] = _clone_state(
                self.initial_personalized_state
            )
        return self.personalized_state_by_client[self.client_idx]

    def _load_client_model(self, shared_model_state):
        model_state = _clone_state(shared_model_state)
        model_state.update(_clone_state(self._personalized_state()))
        self.model_trainer.set_model_params(model_state)

    def _train_phase(self, trainable_names, epochs):
        if epochs <= 0:
            return
        model = self.model_trainer.model
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name in trainable_names
        phase_args = copy.copy(self.args)
        phase_args.epochs = epochs
        self.model_trainer.train(self.local_training_data, self.device, phase_args)

    def train(self, shared_model_state):
        self._load_client_model(shared_model_state)
        original_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self.model_trainer.model.named_parameters()
        }
        head_epochs = int(getattr(self.args, "fedrep_head_epochs", 1))
        representation_epochs = int(
            getattr(self.args, "fedrep_representation_epochs", self.args.epochs)
        )
        if head_epochs < 0 or representation_epochs < 1:
            raise ValueError(
                "fedrep_head_epochs must be non-negative and "
                "fedrep_representation_epochs must be positive"
            )

        try:
            self._train_phase(self.personalized_parameter_names, head_epochs)
            self._train_phase(self.shared_parameter_names, representation_epochs)
            local_state = _clone_state(self.model_trainer.get_model_params())
            self.personalized_state_by_client[self.client_idx] = _clone_state(
                {key: local_state[key] for key in self.personalized_state_keys}
            )
            return local_state
        finally:
            for name, parameter in self.model_trainer.model.named_parameters():
                parameter.requires_grad = original_requires_grad[name]

    def load_personalized_model(self):
        current_state = self.model_trainer.get_model_params()
        self._load_client_model(current_state)

    def local_test(self, b_use_test_dataset):
        test_data = self.local_test_data if b_use_test_dataset else self.local_training_data
        return self.model_trainer.test(test_data, self.device, self.args)
