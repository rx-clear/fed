"""FedRep single-process simulation with optional DMCFE aggregation."""

import logging

from ..fedavg.fedavg_api import FedAvgAPI
from .client import FedRepClient
from .parameters import resolve_fedrep_parameter_keys


class FedRepAPI(FedAvgAPI):
    """Personalize prediction heads and aggregate only representations."""

    def _get_result_file(self, enable_dmcfe):
        configured_path = getattr(self.args, "experiment_result_file", None)
        if configured_path:
            return str(configured_path)
        if enable_dmcfe:
            return "results/dmcfe_fedrep.csv"
        return "results/fedrep_baseline.csv"

    def _get_algorithm_name(self):
        return "FedRep"

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
        initial_state = model_trainer.get_model_params()
        (
            self.shared_state_keys,
            self.personalized_state_keys,
            self.shared_parameter_names,
            self.personalized_parameter_names,
        ) = resolve_fedrep_parameter_keys(
            model_trainer.model,
            initial_state,
            getattr(self.args, "fedrep_personalized_layers", None),
        )
        self.personalized_state_by_client = {}
        initial_personalized_state = {
            key: initial_state[key] for key in self.personalized_state_keys
        }
        logging.info("FedRep shared state keys: %s", self.shared_state_keys)
        logging.info("FedRep personalized state keys: %s", self.personalized_state_keys)

        for client_idx in range(self.args.client_num_per_round):
            self.client_list.append(
                FedRepClient(
                    client_idx,
                    train_data_local_dict[client_idx],
                    test_data_local_dict[client_idx],
                    train_data_local_num_dict[client_idx],
                    self.args,
                    self.device,
                    model_trainer,
                    self.shared_parameter_names,
                    self.personalized_parameter_names,
                    self.personalized_state_keys,
                    self.personalized_state_by_client,
                    initial_personalized_state,
                )
            )

    def _get_aggregated_param_order(self, model_params):
        missing = [key for key in self.shared_state_keys if key not in model_params]
        if missing:
            raise KeyError("FedRep shared state is missing key(s): " + ", ".join(missing))
        return list(self.shared_state_keys)

    def _prepare_client_for_evaluation(self, client, client_idx):
        client.load_personalized_model()
