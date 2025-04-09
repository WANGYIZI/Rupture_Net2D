from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from sklearn.model_selection import train_test_split

import tempfile
import torch
import torch.nn as nn

from tqdm import tqdm

from pathlib import Path

from model import RuptureNet2D

from datasets import BlockDataset, load_data, make_blocks

import ray
import ray.tune as tune

import argparse


def train(model, device, loss_func, train_dataloader, optimizer, scheduler, debug=False):
    # Get train loss
    train_loss = 0.0
    model.train()

    if debug:
        print(f'Current learning rate before epoch: {scheduler.get_last_lr()}')

    for inputs, labels in tqdm(train_dataloader, desc='Training epoch', leave=False, disable=not debug):
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

    if scheduler is not None:
        scheduler.step()

    if debug:
        print(f'Current learning rate after epoch: {scheduler.get_last_lr()}')

    return train_loss


@torch.inference_mode
def test(model, device, loss_func, test_dataloader, debug=False):
    test_loss = 0.0
    model.eval()

    for inputs, labels in tqdm(test_dataloader, desc='Testing epoch', leave=False, disable=not debug):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        test_loss += loss.item()

    return test_loss


def raytune_objective(config, train_data, max_epochs, device, debug=False):
    ''' The raytune objective function that is optimized '''

    model = RuptureNet2D(config)  # a torch.nn.Module object
    model.to(device)

    # Loss function
    criterion = nn.MSELoss(reduction='sum')
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-9)

    # Define the learning rate scheduler
    def lr_lambda(step, warmup_steps=10):
        return (model.embed_dim ** -0.5) * min((step + 1) ** (-0.5), (step + 1) * (warmup_steps ** -1.5))

    # scheduler = None
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Load checkpoint if it exists
    checkpoint = ray.train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / 'data.pkl'
            model_path = Path(checkpoint_dir) / 'model.pt'

            checkpoint_state = torch.load(data_path, weights_only=False)
            model_state = torch.load(model_path, weights_only=True)

            train_idx = checkpoint_state['train_idx']
            val_idx = checkpoint_state['val_idx']
            start_epoch = checkpoint_state['epoch'] + 1
            optimizer.load_state_dict(checkpoint_state['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint_state['scheduler_state_dict'])
            model.load_state_dict(model_state)
    else:
        start_epoch = 0
        train_idx, val_idx = train_test_split(range(len(train_data)), train_size=0.8)

    # prepare data
    train_split = SubsetRandomSampler(train_idx)
    val_split = SubsetRandomSampler(val_idx)

    batch_size = config['batch_size']
    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=8, sampler=train_split)
    val_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=8, sampler=val_split)

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        train_loss = train(model, device, criterion, train_dataloader, optimizer, scheduler, debug=debug)
        val_loss = test(model, device, criterion, val_dataloader, debug=debug)

        # Calculate the average training set loss
        train_loss /= len(train_split)
        val_loss /= len(val_split)

        # save relevant checkpoint data for restoring
        checkpoint_data = {
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_idx': train_idx,
            'val_idx': val_idx,
            'config': config,
        }

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / 'data.pkl'
            model_path = Path(checkpoint_dir) / 'model.pt'
            torch.save(checkpoint_data, data_path)
            torch.save(model.state_dict(), model_path)

            checkpoint = ray.train.Checkpoint.from_directory(checkpoint_dir)
            ray.train.report(
                {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                },
                checkpoint=checkpoint,
            )


def run_raytune(tuning_name, train_data, config_set, max_epochs, device, raytune_address, raytune_storage):
    num_trials = 1

    raytune_storage = Path(raytune_storage)

    if raytune_address != 'localhost':
        ray.init(address=raytune_address)

    # strategy to select best parameters
    scheduler = tune.schedulers.ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=max_epochs,
        grace_period=5,
        reduction_factor=2,
    )

    # the training function with given resources per trial
    trainable = tune.with_resources(
        tune.with_parameters(
            raytune_objective,
            train_data=train_data,
            max_epochs=max_epochs,
            device=device,
        ),
        resources={'cpu': 10, 'gpu': 1},
    )

    # set up hyperparameter tuning or load if previously interrupted
    if tune.Tuner.can_restore(raytune_storage / tuning_name):
        print(f'Restoring from {raytune_storage / tuning_name}')
        tuner = tune.Tuner.restore(
            str(raytune_storage / tuning_name),
            trainable=trainable,
            restart_errored=True,
            resume_unfinished=True,
        )
    else:
        print(f'Creating new tuning in {raytune_storage}')
        tuner = tune.Tuner(
            trainable=trainable,
            param_space=config_set,
            run_config=ray.train.RunConfig(
                storage_path=raytune_storage,
                name=tuning_name,
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_score_attribute='val_loss',
                    checkpoint_score_order='min',
                ),
            ),
            tune_config=tune.TuneConfig(
                num_samples=num_trials,
                scheduler=scheduler,
            ),
        )

    # run raytune hyperparameter tuning
    result = tuner.fit()

    return result


def raytune_gridsearch_helper(config_gridsearch, train_data, max_epochs, device, debug=False):
    """ rewrites gridsearch to match previous config """
    config = {
        'batch_size': 64,
        'lr': 1e-4,
        'conv1_dim': 256,
        'conv1_layers': 2,  # 1st: 256 -> 256, 2nd: 512 -> 512
        'num_transformer_blocks': config_gridsearch['grid'][0],  # 6,
        'num_transformer_heads': config_gridsearch['grid'][1],  # 8,
        'transformer_hidden_dim': config_gridsearch['grid'][2],  # 2048,
        'conv2_dim': 256,  # embed-dim -> 256, 256 -> output
        'dropout_pro': 0.1,
    }

    raytune_objective(config, train_data, max_epochs, device, debug=False)


def run_raytune_gridsearch(tuning_name, train_data, config_set, max_epochs, device, raytune_address, raytune_storage):
    num_trials = 1

    raytune_storage = Path(raytune_storage)
    # strategy to select best parameters
    # scheduler = tune.schedulers.ASHAScheduler(
    #     metric="val_loss",
    #     mode="min",
    #     max_t=max_epochs,
    #     grace_period=5,
    #     reduction_factor=2,
    # )

    if raytune_address != 'localhost':
        ray.init(address=raytune_address)

    # the training function with given resources per trial
    trainable = tune.with_resources(
        tune.with_parameters(
            raytune_gridsearch_helper,
            train_data=train_data,
            max_epochs=max_epochs,
            device=device,
        ),
        resources={'cpu': 10, 'gpu': 1},
    )

    # set up hyperparameter tuning or load if previously interrupted
    if tune.Tuner.can_restore(raytune_storage / tuning_name):
        print(f'Restoring from {raytune_storage / tuning_name}')
        tuner = tune.Tuner.restore(
            str(raytune_storage / tuning_name),
            trainable=trainable,
            restart_errored=True,
            resume_unfinished=True,
        )
    else:
        print(f'Creating new tuning in {raytune_storage}')
        tuner = tune.Tuner(
            trainable=trainable,
            param_space=config_set,
            run_config=ray.train.RunConfig(
                storage_path=raytune_storage,
                name=tuning_name,
                checkpoint_config=ray.train.CheckpointConfig(
                    num_to_keep=2,
                    checkpoint_score_attribute='val_loss',
                    checkpoint_score_order='min',
                ),
            ),
            tune_config=tune.TuneConfig(
                num_samples=num_trials,
                # scheduler=scheduler,
            ),
        )

    # run raytune hyperparameter tuning
    result = tuner.fit()

    return result


def get_best_result(result, result_dir, device, test_data):
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

        data_path = Path(checkpoint_dir) / "data.pkl"
        model_path = Path(checkpoint_dir) / 'model.pt'

        model_state = torch.load(model_path, weights_only=True)
        best_trained_model.load_state_dict(model_state)
        best_trained_model.to(device)

        data_state = torch.load(data_path, weights_only=False)

    # final validation on test set
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=16,
                                                  drop_last=False,
                                                  num_workers=4,
                                                  pin_memory=False)

    criterion = torch.nn.MSELoss(reduction='sum')
    test_loss = test(best_trained_model, device, criterion, test_dataloader)
    test_loss = test_loss / len(test_data)

    print(f'Best trial test loss: {test_loss}')

    # saving best model in given result dir
    print(f'Saving in {result_dir}')
    result_dir.mkdir(exist_ok=True, parents=True)
    torch.save(
        model_state,
        result_dir / 'best_model_state.pt'
    )

    torch.save(
        data_state,
        result_dir / 'best_model_checkpoint_data.pkl',
    )

    result_df.to_csv(result_dir / 'all_results.csv')
    best_result_df.to_csv(result_dir / 'best_result.csv')


def main_hyperparameter(raytune_address, raytune_storage, raytune_name):
    debug = False
    save_path = Path("./results").resolve()  # save model in "result" folder of the current directory
    data_cache_path = Path("./cache/blocks.pt").resolve()
    raytune_storage = Path(raytune_storage).resolve()
    torch.cuda.set_device(0)

    device = torch.device("cuda:0")

    if (data_cache_path).exists():
        print('Loading data from cache')
        train_blocks, test_blocks = torch.load(data_cache_path, weights_only=False)
        print(f'Number of training blocks: {len(train_blocks)}')
        print(f'Number of test blocks: {len(test_blocks)}')
    else:
        df = load_data(debug=debug)
        train_blocks, test_blocks = make_blocks(df, debug=debug)
        data_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save((train_blocks, test_blocks), data_cache_path)

    train_dataset = BlockDataset(train_blocks, debug=debug)
    test_dataset = BlockDataset(test_blocks, debug=debug)

    print('Finished Loading Data')

    # test run function itself in case of debugging
    if debug:
        config = {
            'batch_size': 64,
            'lr': 1e-4,
            'conv1_layers': 2,
            'conv1_dim': 256,
            'num_transformer_blocks': 4,
            'num_transformer_heads': 4,
            'transformer_hidden_dim': 1024,
            'conv2_dim': 256,
            'dropout_pro': 0.1,
        }

        raytune_objective(
            config,
            train_data=train_dataset,
            max_epochs=10,
            device=device,
            debug=True,
        )

    # the hyperparameters to be searched
    config_set = {
        'batch_size': tune.choice([64, 128]),
        'lr': tune.loguniform(1e-3, 1e-5),
        'conv1_layers': tune.choice([1, 2, 3]),
        'conv1_dim': tune.choice([64, 128, 256]),
        'num_transformer_blocks': tune.choice([1, 2, 4]),
        'num_transformer_heads': tune.choice([2, 4, 8]),
        'transformer_hidden_dim': tune.choice([1024, 2048]),
        'conv2_dim': tune.choice([64, 128, 256]),
        'dropout_pro': tune.choice([0.1]),
    }

    result = run_raytune(
        tuning_name=raytune_name,
        train_data=train_dataset,
        config_set=config_set,
        max_epochs=100,
        device=device,
        raytune_address=raytune_address,
        raytune_storage=raytune_storage,
    )

    get_best_result(result, save_path.resolve(), device, test_dataset)


def main_gridsearch(raytune_address, raytune_storage, raytune_name):
    debug = False
    data_cache_path = Path("./cache/blocks.pt").resolve()
    raytune_storage = Path(raytune_storage).resolve()
    torch.cuda.set_device(0)

    device = torch.device("cuda")

    if (data_cache_path).exists():
        print('Loading data from cache')
        train_blocks, test_blocks = torch.load(data_cache_path, weights_only=False)
        print(f'Number of training blocks: {len(train_blocks)}')
        print(f'Number of test blocks: {len(test_blocks)}')
    else:
        df = load_data(debug=debug)
        train_blocks, test_blocks = make_blocks(df, debug=debug)
        data_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save((train_blocks, test_blocks), data_cache_path)

    train_dataset = BlockDataset(train_blocks, debug=debug)

    print('Finished Loading Data')

    # test run function itself in case of debugging
    if debug:
        config = {
            'batch_size': 64,
            'lr': 1e-4,
            'conv1_layers': 2,
            'conv1_dim': 256,
            'num_transformer_blocks': 6,
            'num_transformer_heads': 8,
            'transformer_hidden_dim': 2048,
            'conv2_dim': 256,
            'dropout_pro': 0.1,
        }

        raytune_objective(
            config,
            train_data=train_dataset,
            max_epochs=10,
            device=device,
            debug=True,
        )

    config_set_gridsearch = {
        'grid': tune.grid_search(
            [
                (6, 8, 2048),

                (4, 8, 2048),
                (2, 8, 2048),
                (1, 8, 2048),

                (6, 4, 2048),
                (6, 2, 2048),
                (6, 1, 2048),

                (6, 8, 1024),
                (6, 8, 512),
                (6, 8, 256),
            ]
        )
    }

    run_raytune_gridsearch(
        tuning_name=raytune_name,
        train_data=train_dataset,
        config_set=config_set_gridsearch,
        max_epochs=100,
        device=device,
        raytune_address=raytune_address,
        raytune_storage=raytune_storage,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default='localhost', help='Raytune address')
    parser.add_argument('--storage', default=Path('./raytune').resolve(), help='Raytune base storage')
    parser.add_argument('--name', default='RuptureNet2D', help='Raytune trial folder name')
    parser.add_argument('--gridsearch', default=False, action='store_true', help='Run gridsearch of the paper instead of hyperparamter search')
    args = parser.parse_args()

    if args.gridsearch:
        main_gridsearch(raytune_address=args.address, raytune_storage=args.storage, raytune_name=args.name)
    else:
        main_hyperparameter(raytune_address=args.address, raytune_storage=args.storage, raytune_name=args.name)
