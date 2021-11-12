import os

import torch
from ray import tune

from common.storage import quicktest
from data_preperation import prepare_data
from model.model import Net
from rayTune_common.constants import ins, outs


def train(config, checkpoint_dir=None):
    """

    :param config:
    :param checkpoint_dir:
    :return:
    """

    net = Net(
        len(ins),
        int(config["hidden_layers"]),
        int(config["hidden_layer_width"]),
        len(outs)
    )

    # Define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])

    # The `checkpoint_dir` parameter gets passed by Ray Tune when a checkpoint
    # should be restored.
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    # Import training, validation and test data
    train_loader, x_valid, y_valid, val_loader, x_test, y_test = prepare_data(
        INPUT_COLS=ins,
        OUTPUT_COLS=outs,
        train_batch_size=int(config["batch_size"])
    )

    # Train Network
    _n_epochs = 100
    for epoch in range(_n_epochs):
        for inputs, labels in train_loader:
            # Zero the parameter gradients (from last iteration)
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)

            # Compute cost function
            batch_mse = criterion(outputs, labels)

            reg_loss = 0
            for param in net.parameters():
                reg_loss += param.pow(2).sum()

            cost = batch_mse + config["l2"] * reg_loss

            # Backward propagation to compute gradient
            cost.backward()

            # Update parameters using gradient
            optimizer.step()

        # Evaluate model on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
        mse_val /= len(val_loader.dataset)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path)

            tune.report(mean_square_error=mse_val)

        quicktest(mse_val)

    print("Finished Training")
