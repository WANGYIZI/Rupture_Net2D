from pathlib import Path
import argparse

import torch
import ray.tune as tune

from model import RuptureNet2D
import pandas as pd

import numpy as np

from sklearn import metrics

from tqdm import tqdm


from datasets import load_data, make_blocks, BlockDataset

from train_raytune import raytune_gridsearch_helper

import matplotlib.pyplot as plt


@torch.inference_mode
def compute_predictions(model, test_data, device):        # final validation on test set

    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=16,
                                                  drop_last=False,
                                                  num_workers=4,
                                                  pin_memory=False)

    model.eval()

    originals = []
    targets = []
    predictions = []
    for inputs, labels in tqdm(test_dataloader, total=len(test_dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        originals.append(inputs)
        targets.append(labels)
        predictions.append(outputs)

    all_originals = torch.cat(originals, dim=0).cpu().numpy()
    all_targets = torch.cat(targets, dim=0).cpu().numpy()
    all_predictions = torch.cat(predictions, dim=0).cpu().numpy()

    return all_originals, all_targets, all_predictions


def plot_loss(results_df, results_dir):
    # Plot loss curve
    plt.figure()
    plt.plot(results_df['training_iteration'], results_df['train_loss'], label='Train Loss')
    plt.plot(results_df['training_iteration'], results_df['val_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the vector image to the results folder
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / 'loss_curve.svg', format='svg')

    # Display svg graphics
    plt.show()


def compute_rae(pred_value, real_value, block_size=100):
    # Relative Error defined here

    rae = np.sum(np.abs(pred_value - real_value)) / np.sum(np.abs(real_value))
    return rae


def correlation(pred_value, real_value, block_size=100):
    mean_pred = np.mean(pred_value)
    mean_real = np.mean(real_value)

    return (np.sum((real_value - mean_real) * (pred_value - mean_pred)) /
            np.sqrt(np.sum((real_value - mean_real)**2) * np.sum((pred_value - mean_pred)**2)))


def compute_metrics(predictions, targets, block_size=100):
    ''' Computes MSE and MAE and MSPE metrics for the test set '''
    predict_front = predictions[:, 0:block_size, 0]
    predict_slip = predictions[:, 0:block_size, 1]
    actual_front = targets[:, 0:block_size, 0]
    actual_slip = targets[:, 0:block_size, 1]
    # Print the number of test set samples
    print(predict_front.shape, actual_front.shape)

    return {
        'front_mae': metrics.mean_squared_error(actual_front, predict_front),
        'slip_mae': metrics.mean_squared_error(actual_slip, predict_slip),
        'front_mse': metrics.mean_absolute_error(actual_front, predict_front),
        'slip_mse': metrics.mean_absolute_error(actual_slip, predict_slip),
        'front_mspe': np.sqrt(metrics.mean_squared_error(actual_front, predict_front)),
        'slip_mspe': np.sqrt(metrics.mean_squared_error(actual_slip, predict_slip)),
        'front_rae': compute_rae(predict_front, actual_front),
        'slip_rae': compute_rae(predict_slip, actual_slip),
        'front_rxy': correlation(predict_front, actual_front),
        'slip_rxy': correlation(predict_slip, actual_slip),
    }


def get_metrics_from_raytune(raytune_result, raytune_trainable, device, test_data):

    if isinstance(raytune_result, (Path, str)):
        # restore with dummy trainable for results only
        restored_tuner = tune.Tuner.restore(str(raytune_result), trainable=raytune_trainable)
        result_grid = restored_tuner.get_results()
    else:
        result_grid = raytune_result

    results_data = []

    for result in result_grid:
        if result.error:
            print('skipped result due to error')
            continue

        config_gridsearch = result.config
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
        # load model
        model = RuptureNet2D(config)
        checkpoint = result.get_best_checkpoint(metric='val_loss', mode='min')
        with checkpoint.as_directory() as checkpoint_dir:
            # data_path = Path(checkpoint_dir) / "data.pkl"
            model_path = Path(checkpoint_dir) / 'model.pt'

            if not model_path.exists():
                continue

            model_state = torch.load(model_path, weights_only=True, map_location=device)
            # data_state = torch.load(data_path, weights_only=False)

        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        originals, predictions, targets = compute_predictions(model, test_data, device)
        metrics = compute_metrics(predictions, targets)

        result_metrics = result.metrics | metrics

        results_data.append(result_metrics)

    # save resulting metrics in result_dir
    results_data = pd.DataFrame(results_data)
    return results_data


def get_results_best_model(test_data, results_dir):
    results_df_path = results_dir / 'best_result.csv'
    model_state_file = results_dir / 'best_model_state.pt'
    checkpoint_state_file = results_dir / 'best_model_checkpoint_data.pkl'

    # load model
    model_state = torch.load(model_state_file, weights_only=True)
    checkpoint_state = torch.load(checkpoint_state_file, weights_only=False)
    model = RuptureNet2D(checkpoint_state['config'])
    model.load_state_dict(model_state)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    # load training statistics
    results_df = pd.read_csv(results_df_path)

    originals, predictions, targets = compute_predictions(model, test_data, device)

    # plot everything
    plot_loss(results_df, results_dir)
    m = compute_metrics(predictions, targets)
    print(m)


def main(raytune_gridsearch_path, results_dir):
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    data_cache_path = Path("./cache/blocks.pt")

    # load data
    if (data_cache_path).exists():
        print('Loading data from cache')
        train_blocks, test_blocks = torch.load(data_cache_path, weights_only=False)
        print(f'Number of training blocks: {len(train_blocks)}')
        print(f'Number of test blocks: {len(test_blocks)}')
    else:
        df = load_data()
        train_blocks, test_blocks = make_blocks(df)
        data_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save((train_blocks, test_blocks), data_cache_path)

    test_data = BlockDataset(test_blocks)

    # get best hyperparameter model
    get_results_best_model(test_data, results_dir)

    # compute metrics from gridsearch
    results = get_metrics_from_raytune(
        raytune_result=raytune_gridsearch_path,
        raytune_trainable=raytune_gridsearch_helper,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        test_data=test_data)

    results.to_csv(results_dir / 'Gridsearch.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default=Path('./results').resolve(), help='The result folder')
    parser.add_argument('--gridsearch_path', default=Path('./raytune/gridsearch').resolve(), help='Path of the raytune gridsearch folder including name')
    args = parser.parse_args()

    main(args.gridsearch_path, args.result_path)
