import torch

from rayTune_common.constants import ins, outs
from rayTune_common.model import Net


def config_to_model(config: {}, checkpoint_path: str):
    model = Net(
        len(ins),
        int(config["hidden_layers"]),
        int(config["hidden_layer_width"]),
        len(outs),
        dropout_value=config["dropout"]
    )

    device = "cpu"
    model.to(device)
    model_state, optimizer_state = torch.load(checkpoint_path)
    model.load_state_dict(model_state)

    return model
