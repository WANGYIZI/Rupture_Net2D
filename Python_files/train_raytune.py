import torch.distributed as dist
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

import os
import tempfile
import torch
import numpy as np
import torch.nn as nn

import time
import pandas as pd
from pathlib import Path

from model import RuptureNet2D

from datasets import BlockDataset, load_data, make_blocks

import ray
import ray.tune





# INFO setup training

# Define the learning rate decay function
def lr_lambda(step, warmup_steps=4000):
    return (model.embed_dim ** -0.5) * min((step + 1) ** (-0.5), (step + 1) * (warmup_steps ** -1.5))


# FIX Only use when loading trained model
# state_dict = torch.load(checkpoint)
# new_state_dict = {}
#
# for key, value in state_dict.items():
#     if key.startswith('module.'):  # Assume that the model is wrapped in torch.nn.DataParallel before loading
#         key = key[7:]  # Remove the 'module.' prefix
#     new_state_dict[key] = value


def train(model, device, loss_func, train_dataloader, optimizer, scheduler):
    # Get train loss
    train_loss = 0.0
    model.train()
    for inputs, labels in train_dataloader:
        # Forward propogation
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        # the train loss of current batch
        train_loss += loss.item()

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    return train_loss


@torch.inference_mode
def test(model, device, loss_func, test_dataloader):
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            test_loss += loss.item()

        return test_loss


def raytune_objective(config, train_data, max_epochs, device):
    ''' The raytune objective function that is optimized '''

    model = RuptureNet2D()  # a torch.nn.Module object
    model.to(device)

    # Loss function
    criterion = nn.MSELoss(reduction='sum')
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    # Define the learning rate scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Load checkpoint if it exists
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / 'data.pkl'
            model_path = Path(checkpoint_dir) / 'model.pt'

            checkpoint_state = torch.load(data_path, weights_only=false)
            model_state = torch.load(model_path, weights_only=True)

            train_idx = checkpoint_state['train_idx']
            val_idx = checkpoint_state['val_idx']
            start_epoch = checkpoint_state['epoch']
            optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_state['scheduler_state_dict'])
            model.load_state_dict(model_state)
    else:
        start_epoch = 0
        train_idx, val_idx = random_split(len(train_data, train_size=0.8))

    # prepare data
    train_split = SubsetRandomSampler(train_idx)
    val_split = SubsetRandomSampler(val_idx)

    batch_size = 256
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8, sampler=train_split)
    val_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=8, sampler=val_split)

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        train_loss = train(model, device, criterion, train_dataloader, optimizer, scheduler)
        val_loss = test(model, device, criterion, val_dataloader)

        # Calculate the average training set loss
        train_loss /= len(train_dataset)
        val_loss /= len(validate_dataset)

        # save relevant checkpoint data for restoring
        checkpoint_data = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_idx': train_idx,
            'val_idx': val_idx,
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / 'data.pkl'
            model_path = Path(checkpoint_dir) / 'model.pt'
            torch.save(checkpoint_data, data_path)
            torch.save(model.state_dict(), model_path)

            checkpoint = ray.train.Checkpoint.from_directory(checkpoint_dir)
            ray.tune.report(
                {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                },
                checkpoint=checkpoint,
            )


def run_raytune(result_dir, tuning_name, train_data, test_data, config_set, max_epochs, device):
    num_trials = 30
    path = Path('./raytune')

    # strategy to select best parameters
    scheduler = ray.tune.schedulers.ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_epochs,
        grace_period=5,
        reduction_factor=2,
    )

    # the training function with given resources per trial
    trainable = ray.tune.with_resources(
        ray.tune.with_parameters(
            raytune_objective,
            train_data=train_data,
            epochs=max_epochs,
            device=device,
        ),
        resources={'cpu': 10, 'gpu': 0.5},
    )

    # set up hyperparameter tuning or load if previously interrupted
    if ray.tune.Tuner.can_restore(path / tuning_name):
        print(f'Restoring from {path / tuning_name}')
        tuner = ray.tune.Tuner.restore(
            str(path / tuning_name),
            trainable=trainable,
            resume_errored=True,
            resume_unfinished=True,
        )
    else:
        print(f'Creating new tuning in {path}')
        tuner = ray.tune.Tuner(
            trainable=trainable,
            param_space=config_set,
            run_config=ray.train.RunConfig(
                storage_path=path,
                name=tuning_name,
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_score_attribute='val_auc',
                    checkpoint_score_order='max',
                ),
            ),
            tune_config=ray.tune.TuneConfig(
                num_samples=num_trials,
                scheduler=scheduler,
            ),
        )

    # run raytune hyperparameter tuning
    result = tuner.fit()

    result_df = result.get_dataframe(filter_metric='val_loss', filter_mode='min')

    # get best result
    best_result = result.get_best_result(metric="val_loss", mode="min", scope="all")
    best_result_df = best_result.metrics_dataframe
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial path: {best_result.path}")
    print(f"Best trial validation loss: {best_result_df['val_loss'].min()}")

    best_trained_model = RuptureNet2D(best_result.config)

    best_checkpoint = best_result.get_best_checkpoint(metric="val_loss", mode="min")
    with best_checkpoint.as_directory() as checkpoint_dir:

        # data_path = Path(checkpoint_dir) / "data.pkl"
        model_path = Path(checkpoint_dir) / 'model.pt'

        model_state = torch.load(model_path, weights_only=True)
        best_trained_model.load_state_dict(model_state)
        best_trained_model.to(device)

    # final validation on test set
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=16,
                                                  drop_last=False,
                                                  num_workers=4,
                                                  pin_memory=False)

    criterion = torch.nn.MSELoss(reduction='sum')
    test_loss = test(best_trained_model, device, criterion, test_dataloader)
    test_loss = test_loss / len(test_dataloader)

    print(f'Best trial test loss: {test_loss}')

    # saving best model in given result dir
    print(f'Saving in {result_dir}')
    result_dir.mkdir(exist_ok=True, parents=True)
    torch.save(
        {
            'model_state_dict': model_state,
            'model_params': best_result.config
        },
        result_dir / 'best_model_best_val_auc.pt'
    )

    result_df.to_csv(result_dir / 'all_results.csv')
    best_result_df.to_csv(result_dir / 'best_result.csv')


def plot_results(train_losses, valid_losses, actual_iteration):
    # INFO Plot results
    # Plot loss curve
    plt.figure()
    plt.plot(range(actual_iteration), train_losses, label='Train Loss')
    plt.plot(range(actual_iteration), valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # # Create the results folder (if it does not exist)
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the vector image to the results folder
    plt.savefig('results/loss_curve.svg', format='svg')

    # Display svg graphics
    plt.show()


if __name__ == '__main__':
    save_path = Path("./results")  # save model in "result" folder of the current directory
    torch.cuda.set_device(0)

    device = torch.device("cuda:0")

    df = load_data(debug=True)
    train_blocks, test_blocks = make_blocks(df)

    train_dataset = BlockDataset(train_blocks)
    test_dataset = BlockDataset(test_blocks)
    print(len(train_dataset))

    # the hyperparameters to be searched
    config_set = {
        'xx': [],
    }

    run_raytune(
        result_dir=save_path,
        tuning_name='RuptureNet2D',
        train_data=train_dataset,
        test_data=test_dataset,
        config_set=config_set,
        max_epochs=2000,
        device=device
    )




