import torch

import fedml
from fedml import FedMLRunner
from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist


def load_data(args):
    download_mnist(args.data_cache_dir)
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)

    """
    Please read through the data loader at to see how to customize the dataset for FedML framework.
    """
    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args,
        args.batch_size,
        train_path=args.data_cache_dir + "/MNIST/train",
        test_path=args.data_cache_dir + "/MNIST/test",
    )
    """
    For shallow NN or linear models, 
    we uniformly sample a fraction of clients each round (as the original FedAvg paper)
    """
    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


class FedRepMNISTModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_1 = torch.nn.Linear(input_dim, 128)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(128, output_dim)

    def forward(self, x):
        representation = self.relu(self.linear_1(self.flatten(x)))
        return self.linear_2(representation)


def create_model(args, output_dim):
    """Build the model selected by YAML, with an explicit MLP research option."""
    model_name = str(getattr(args, "model", "mlp")).strip().lower()
    if model_name in {"mlp", "fedrep_mlp"}:
        return FedRepMNISTModel(28 * 28, output_dim)
    return fedml.model.create(args, output_dim)


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()
    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = load_data(args)

    # Build the model selected in the YAML (the custom MLP is opt-in).
    model = create_model(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
