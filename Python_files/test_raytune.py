from pathlib import Path

import torch
import ray.tune as tune

from model import RuptureNet2D
import pandas as pd

import numpy as np

from sklearn import metrics


from datasets import column_max, column_min, load_data, make_blocks, BlockDataset

import matplotlib.pyplot as plt


@torch.inference_mode
def compute_predictions(model, test_dataloader, device):
    model.eval()

    originals = []
    targets = []
    predictions = []
    for inputs, labels in test_dataloader:
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
    # INFO Plot results
    # Plot loss curve
    plt.figure()
    plt.plot(results_df['training_iteration'], results_df['train_loss'], label='Train Loss')
    plt.plot(results_df['training_iteration'], results_df['val_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # # Create the results folder (if it does not exist)

    # Save the vector image to the results folder
    plt.savefig(results_dir / 'loss_curve.svg', format='svg')

    # Display svg graphics
    plt.show()


def compute_error(pred_value, real_value, block_size=100):
    # Relative Error defined here
    relative_error = 0.0

    sum_real = 0.0
    added_error = 0.0
    for i in range(block_size):
        sum_real += real_value[i, 0]
        added_error += abs(real_value[i, 0] - pred_value[i, 0])
    if (sum_real == 0):
        return 1
    relative_error = added_error / sum_real
    return relative_error


def compute_metrics(predictions, targets, block_size=100):
    ''' Computes MSE and MAE and MSPE metrics for the test set '''
    predict_front = predictions[:, 0:block_size, 0]
    predict_slip = predictions[:, 0:block_size, 1]
    actual_front = targets[:, 0:block_size, 0]
    actual_slip = targets[:, 0:block_size, 1]
    # Print the number of test set samples
    print(predict_front.shape, actual_front.shape)

    return {
        'rupture_mae': metrics.mean_squared_error(actual_front, predict_front),
        'slip_mae': metrics.mean_squared_error(actual_slip, predict_slip),
        'rupture_mse': metrics.mean_absolute_error(actual_front, predict_front),
        'slip_mse': metrics.mean_absolute_error(actual_slip, predict_slip),
        'rupture_mspe': np.sqrt(metrics.mean_squared_error(actual_front, predict_front)),
        'slip_mspe': np.sqrt(metrics.mean_squared_error(actual_slip, predict_slip)),
    }


# def compute_error2(predictions, targets, block_size=100):
#     ''' does this just compute the error with the original slip and front before normalization? '''
#     predict_front = predictions[:, 0:block_size, 0]
#     predict_slip = predictions[:, 0:block_size, 1]
#     actual_front = targets[:, 0:block_size, 0]
#     actual_slip = targets[:, 0:block_size, 1]
#
#     relative_error_front = 0.0
#     relative_error_slip = 0.0
#     # Dimension Reduction
#     restored_predict_slip = predict_slip * (max_Slip - min_Slip) + min_Slip
#     restored_actual_slip = actual_slip * (max_Slip - min_Slip) + min_Slip
#     restored_predict_front = predict_front * (max_Front - min_Front) + min_Front
#     restored_actual_front = actual_front * (max_Front - min_Front) + min_Front
#     # the number of samples
#     sample_num = predictions.shape[0]
#
#     for i in range(sample_num):
#         single_relative_actual_front = 0.0
#         single_relative_predict_front = 0.0
#         single_relative_actual_slip = 0.0
#         single_relative_predict_slip = 0.0
#         for j in range(block_size):
#             single_relative_actual_front += restored_actual_front[i, j]
#             single_relative_predict_front += restored_predict_front[i, j]
#             single_relative_actual_slip += restored_actual_slip[i, j]
#             single_relative_predict_slip += restored_predict_slip[i, j]
#
#         relative_error_front += abs((single_relative_actual_front - single_relative_predict_front) / single_relative_actual_front)
#         relative_error_slip += abs((single_relative_actual_slip - single_relative_predict_slip) / single_relative_actual_slip)
#     relative_error_front /= sample_num
#     relative_error_slip /= sample_num
#     print("Rupture Time Relative Error: " + str(relative_error_front))
#     print("Final slip Relative Error: " + str(relative_error_slip))


def plot_curve(originals, predictions, targets, results_dir, data_idx_start=3673, num_plots=1, block_size=100):
    # Loop here to plot figures
    for sample_idx in range(data_idx_start, data_idx_start + num_plots):

        # Start with sample_idx
        # sample_idx = i + seq

        # Lnuc\StrengthDrop\StressDrop
        feature1_actual_front = targets[sample_idx, 0:block_size, 0] * (column_max[4] - column_min[4]) + column_min[4]
        feature1_actual_front = feature1_actual_front.reshape(-1, 1)
        # print(feature1_actual_front.shape)

        feature1_pred_front = predictions[sample_idx, 0:block_size, 0] * (column_max[4] - column_min[4]) + column_min[4]
        feature1_pred_front = feature1_pred_front.reshape(-1, 1)
        print(feature1_actual_front.shape)

        # restore X
        x = originals[sample_idx, 0:block_size, 1] * (column_max[1] - column_min[1]) + column_min[1]
        x = x.reshape(-1, 1)

        # print(feature1_actual_front.shape)

        # restore Lnuc
        lnuc = originals[sample_idx, 0:block_size, 0] * (column_max[0] - column_min[0]) + column_min[0]
        lnuc = lnuc.reshape(-1, 1)
        # print(feature1_actual_front.shape)

        # restore strengthdrop
        strengthdrop = originals[sample_idx, 0:block_size, 2] * (column_max[2] - column_min[2]) + column_min[2]
        strengthdrop = strengthdrop.reshape(-1, 1)
        # print(feature1_actual_front.shape)

        # restore stressdrop
        stressdrop = originals[sample_idx, 0:block_size, 3] * (column_max[3] - column_min[3]) + column_min[3]
        stressdrop = stressdrop.reshape(-1, 1)
        # print(feature1_actual_front.shape)

        plt.plot(x, lnuc, label="Lnuc")  # Plot nucleation length
        plt.xlabel("X*")
        plt.legend()
        # faultLength = f"Fault length = {round(abs(x[0, 0] - x[-1, 0]), 2)}km"
        plt.show()

        plt.plot(x, strengthdrop, label='StrengthDrop')  # Plot StressDrop StrengthDrop
        plt.plot(x, stressdrop, label='StressDrop')
        plt.xlabel('X*')
        plt.legend()

        plt.show()

        plt.plot(x, feature1_actual_front, label='Actual')  # Plot Rupture time(when Rupture front arrives)
        plt.plot(x, feature1_pred_front, label='Predicted')
        plt.xlabel('X*')
        plt.ylabel('Front')
        plt.title('Prediction vs Actual - Front')
        error_front = f'error={round(compute_error(feature1_pred_front, feature1_actual_front) * 100, 4)}%'
        plt.text(5, 3, error_front, fontsize=12, fontname='SimHei')
        plt.legend()

        # Save the vector image to the results folder
        plt.savefig(results_dir / 'front_example_curve.svg', format='svg')

        plt.show()

        feature2_actual_slip = targets[sample_idx, 0:block_size, 1] * (column_max[5] - column_min[5]) + column_min[5]  # Plot Rupture time(when Rupture front arrives)
        feature2_actual_slip = feature2_actual_slip.reshape(-1, 1)
        feature2_pred_slip = predictions[sample_idx, 0:block_size, 1] * (column_max[5] - column_min[5]) + column_min[5]
        feature2_pred_slip = feature2_pred_slip.reshape(-1, 1)

        plt.plot(x, feature2_actual_slip, label='Actual')
        plt.plot(x, feature2_pred_slip, label='Predicted')
        plt.xlabel('X*')
        plt.ylabel('Slip')
        plt.title('Prediction vs Actual - Slip')
        error_slip = f'error={round(compute_error(feature2_pred_slip, feature2_actual_slip) * 100, 4)}%'
        plt.text(5, 5, error_slip, fontsize=12, fontname='SimHei')
        plt.legend()

        # Save the vector image to the results folder
        plt.savefig(results_dir / 'slip_example_curve.svg', format='svg')

        plt.show()

        print("Error of slip is " + str(compute_error(feature2_pred_slip, feature2_actual_slip)))
        print("Error of front is " + str(compute_error(feature1_pred_front, feature1_actual_front)))
        print("Error of slip is " + str(compute_error(feature2_pred_slip, feature2_actual_slip)))


def get_metrics_from_raytune(raytune_result, results_file, device, test_data):

    if isinstance(raytune_result, Path):
        restored_tuner = tune.Tuner.restore(raytune_result)
        result_grid = restored_tuner.get_results()
    else:
        result_grid = raytune_result

    results_data = {
        'config': [],
        'metric': [],
    }

    for result in result_grid:
        if result.error:
            print('skipped result due to error')
            continue

        config = result.config

        # load model
        model = RuptureNet2D(config)
        checkpoint = result.get_best_checkpoint(metric='val_loss', mode='min')
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            model_path = Path(checkpoint_dir) / 'model.pt'

            model_state = torch.load(model_path, weights_only=True)
            data_state = torch.load(data_path, weights_only=False)

        model.load_state_dict(model_state)
        model.to(device)
        model.eval()

        # final validation on test set
        test_dataloader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=16,
                                                      drop_last=False,
                                                      num_workers=4,
                                                      pin_memory=False)

        originals, predictions, targets = compute_predictions(model, test_dataloader, device)
        metrics = compute_metrics(predictions, targets)

        results_file['config'].append(config)
        results_file['metrics'].append(metrics)
        # calculate metrics from paper and save them for each model!

    # save resulting metrics in result_dir
    results_data.to_csv(results_file)


def main():
    results_dir = Path('./results')
    results_dir.mkdir(exist_ok=True, parents=True)
    results_df_path = results_dir / 'best_result.csv'
    model_state_file = results_dir / 'best_model_state.pt'
    checkpoint_state_file = results_dir / 'best_model_checkpoint_data.pkl'
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
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=16,
                                                  drop_last=False,
                                                  num_workers=4,
                                                  pin_memory=False)

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

    originals, predictions, targets = compute_predictions(model, test_dataloader, device)

    # plot everything
    plot_loss(results_df, results_dir)
    m = compute_metrics(predictions, targets)
    print(m)
    plot_curve(originals, predictions, targets, results_dir)


if __name__ == '__main__':
    main()
