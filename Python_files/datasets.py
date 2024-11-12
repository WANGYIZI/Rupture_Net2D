import pandas as pd
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

max_Lnuc, min_Lnuc = -np.inf, np.inf                                     # Nucleation length
max_X, min_X = -np.inf, np.inf                                           # Nucleation position
max_StrengthDrop, min_StrengthDrop = -np.inf, np.inf                     # Max and Min StrengthDrop
max_StressDrop, min_StressDrop = -np.inf, np.inf                         # Max and Min StressDrop
max_Front, min_Front = -np.inf, np.inf                                   # Max and Min RuptureTime
max_Slip, min_Slip = -np.inf, np.inf                                     # Max and Min FinalSlip
max_Logic, min_Logic = -np.inf, np.inf                                   # Reveal whether or not rupture at the position
max_StrengthwithNuc, min_StrengthwithNuc = -np.inf, np.inf               # Reveal nucleation information
max_RuptureEnergy, min_RuptureEnergy = -np.inf, np.inf                   # Max and Min FractureEnergy
mu = 32.04  # shear elasticity


# Data normalization
# TODO: from data instead of what they did?
# The list represents the maximum or minimum values from left to right
# max_Lnuc, max_X*, max_StrengthDrop, max_StressDrop,max_RuptureTime, max_Slip, max_Logic, max_StrengthwithNuc, max_RuptureEnergy
column_max = [16.239269, 97.6, 125.064626, 66.054391, 59.647999, 86.057297, 1, 102.996661, 20855.37742025637]
# column_max[8] = max(0.5 * Lnuc * StrengthDrop ^ 2)
column_min = [0.003045, -44.8, 4.100412, -10.37305, 0.0, 0.0, 0, -10.37305, 0.7013907496639992]
column_max[8] = column_max[8] / (1.158 * mu)
# now column_max[8] = 0.5 * Lnuc * StrengthDrop ^ 2 / (1.158mu) = fracture energy
column_min[8] = column_min[8] / (1.158 * mu)


def read_200_lines(csv_files):
    df = pd.read_csv(csv_files, index_col=False,
                     names=["Lnuc", "X", "StrengthDrop", "StressDrop", "Front", "Slip", "Logic", "StrengthwithNuc"])
    df = df.iloc[::2, :]
    return df


def read_100_lines(csv_files):
    df = pd.read_csv(csv_files, index_col=False, names=["Lnuc", "X", "StrengthDrop", "StressDrop", "Front", "Slip", "Logic", "StrengthwithNuc"])
    return df


def load_data(debug=False):
    # Read all csv files for training
    global_fracture = 0
    partial_fracture = 0
    data_frames = []
    for csv_file in csv_files_200lines:
        if debug:
            print(csv_file)
        df = read_200_lines((csv_file))
        if 'A' in csv_file:
            global_fracture += len(df) // 100
        else:
            partial_fracture += len(df) // 100
        df["RuptureEnergy"] = (0.5 * df["Lnuc"] * df["StrengthDrop"] ** 2) / (1.158 * mu)  # FractureEnergy directly introduced
        df["RuptureEnergy"] = df["RuptureEnergy"].astype(float)  # Converts the data type to floating point
        if debug:
            print(df.shape)
        data_frames.append(df)

    for csv_file in csv_files_100lines:
        if debug:
            print(csv_file)
        df = read_100_lines((csv_file))
        df["Logic"] = 1.0
        df["StrengthwithNuc"] = 20.0
        df["RuptureEnergy"] = (0.5 * df["Lnuc"] * df["StrengthDrop"] ** 2) / (1.158 * mu)  # FractureEnergy directly introduced
        df["RuptureEnergy"] = df["RuptureEnergy"].astype(float)  # Converts the data type to floating point
        if debug:
            print(df.shape)
        data_frames.append(df)
    df = pd.concat(data_frames, axis=0, ignore_index=True)

    if debug:
        print("Global fracture num = " + str(global_fracture))
        print("Partial fracture num = " + str(partial_fracture))
        print(df.shape)

        print(df.head())
        print(df.describe())

        # print('-----------')
        # # Print Maximum and Minimum
        # print(f"Lnuc maximum: {max(df['Lnuc'])}, minimum: {min(df['Lnuc'])}")
        # print(f"X maximum: {max(df['X'])}, minimum: {min(df['X'])}")
        # print(f"StressDrop maximum: {max(df['StressDrop'])}, minimum: {min(df['StressDrop'])}")
        # print(f"StrengthDrop maximum: {max(df['StrengthDrop'])}, minimum: {min(df['StrengthDrop'])}")
        # print(f"Front maximum: {max(df['Front'])}, minimum: {min(df['Front'])}")
        # print(f"Slip maximum: {max(df['Slip'])}, minimum: {min(df['Slip'])}")
        # print(f"Logic maximum: {max(df['Logic'])}, minimum: {min(df['Logic'])}")
        # print(f"StringthwithNuc maximum: {max(df['StrengthwithNuc'])}, minimum: {min(df['StrengthwithNuc'])}")
        # print(f"RuptureEnergy maximum: {max(df['RuptureEnergy'])}, minimum: {min(df['RuptureEnergy'])}")



    # map each variable to the NN onto the interval [0,1]
    for i in range(0, 9):
        df.iloc[:, i] = (df.iloc[:, i] - column_min[i]) / (column_max[i] - column_min[i])

    return df


def make_blocks(df, block_size=100, debug=False):
    '''Splits df into blocks of size 100, and splits into train and test sets'''

    # compute number of blocks
    num_blocks = len(df) // block_size

    def split_label(block):
        return (
            torch.tensor(block.iloc[:, [0, 1, 2, 3, -1]].values, dtype=torch.float32),
            torch.tensor(block.iloc[:, 4:6].values, dtype=torch.float32)
        )

    # Divide the df object into small blocks(one Block <--> one Sample)
    blocks = [split_label(df[i * block_size:(i + 1) * block_size]) for i in range(num_blocks)]

    train_idx, test_idx = train_test_split(range(len(blocks)), train_size=0.8)

    train_blocks = [blocks[i] for i in train_idx]
    test_blocks = [blocks[i] for i in test_idx]

    if debug:
        print(f"Number of training set samples: {len(train_blocks)}")
        print(f"Number of test set samples: {len(test_blocks)}")

    return train_blocks, test_blocks


class BlockDataset(Dataset):
    '''Dataset for list of blocks. each block is (data, label) tensor created by make_blocks'''

    def __init__(self, blocks, debug=False):
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        return self.blocks[index]


csv_files_200lines = [
    "./datasets/Ainlabel0.csv",
    "./datasets/Ainlabel1.csv",
    # "./datasets/Ainlabel2.csv",
    "./datasets/Ainlabel3.csv",
    "./datasets/Ainlabel68.csv",
    "./datasets/Ainlabel69.csv",
    # the above files are homogeneous rupture  A0-A3 A68-69 (except A2 for testing)
    # all of the above samples are complete rupture


    "./datasets/Ainlabel45.csv",
    "./datasets/Ainlabel46.csv",
    "./datasets/Ainlabel47.csv",
    "./datasets/Ainlabel48.csv",
    # "./datasets/Ainlabel49.csv",
    # "./datasets/Ainlabel50.csv",
    # "./datasets/Ainlabel51.csv",
    "./datasets/Ainlabel52.csv",
    "./datasets/Ainlabel53.csv",
    "./datasets/Ainlabel54.csv",
    "./datasets/Ainlabel55.csv",
    "./datasets/Ainlabel56.csv",
    "./datasets/Ainlabel57.csv",
    "./datasets/Ainlabel58.csv",
    "./datasets/Ainlabel59.csv",
    "./datasets/Ainlabel60.csv",
    "./datasets/Ainlabel61.csv",
    "./datasets/Ainlabel64.csv",
    "./datasets/Ainlabel65.csv",
    # "./datasets/Ainlabel66.csv",
    # "./datasets/Ainlabel67.csv",

    # the above files are One-Asperity "global rupture"
    # A 49-51, 66-67 for testing



    # "./datasets/Ainlabel79.csv",

    # "./datasets/Ainlabel91.csv",


    # "./datasets/Ainlabel109.csv",

    # "./datasets/Ainlabel119.csv",

    "./datasets/Ainlabel143.csv",
    "./datasets/Ainlabel144.csv",
    "./datasets/Ainlabel145.csv",
    "./datasets/Ainlabel146.csv",
    "./datasets/Ainlabel147.csv",
    "./datasets/Ainlabel148.csv",
    "./datasets/Ainlabel149.csv",
    "./datasets/Ainlabel150.csv",
    "./datasets/Ainlabel151.csv",
    "./datasets/Ainlabel152.csv",
    "./datasets/Ainlabel153.csv",
    "./datasets/Ainlabel154.csv",
    "./datasets/Ainlabel155.csv",
    "./datasets/Ainlabel156.csv",
    "./datasets/Ainlabel157.csv",
    "./datasets/Ainlabel158.csv",
    "./datasets/Ainlabel159.csv",
    "./datasets/Ainlabel160.csv",
    "./datasets/Ainlabel161.csv",
    "./datasets/Ainlabel162.csv",
    "./datasets/Ainlabel163.csv",
    "./datasets/Ainlabel164.csv",
    "./datasets/Ainlabel165.csv",
    "./datasets/Ainlabel166.csv",
    "./datasets/Ainlabel167.csv",
    "./datasets/Ainlabel168.csv",
    "./datasets/Ainlabel169.csv",
    "./datasets/Ainlabel170.csv",
    "./datasets/Ainlabel171.csv",
    "./datasets/Ainlabel172.csv",
    "./datasets/Ainlabel173.csv",
    "./datasets/Ainlabel174.csv",
    "./datasets/Ainlabel175.csv",
    "./datasets/Ainlabel176.csv",
    "./datasets/Ainlabel177.csv",
    "./datasets/Ainlabel178.csv",
    "./datasets/Ainlabel179.csv",
    "./datasets/Ainlabel180.csv",
    "./datasets/Ainlabel181.csv",
    "./datasets/Ainlabel182.csv",
    "./datasets/Ainlabel183.csv",
    # "./datasets/Ainlabel184.csv",
    # "./datasets/Ainlabel185.csv",
    # "./datasets/Ainlabel186.csv",
    # "./datasets/Ainlabel187.csv",
    # "./datasets/Ainlabel188.csv",
    # "./datasets/Ainlabel189.csv",
    # the above files are Two-Asperity "global rupture"
    # A 49-51, 66-67, 79, 91, 109, 119, 184, and 185-189 for testing

    "./datasets/Binlabel15.csv",
    "./datasets/Binlabel16.csv",
    "./datasets/Binlabel17.csv",
    "./datasets/Binlabel18.csv",
    "./datasets/Binlabel24.csv",
    "./datasets/Binlabel25.csv",
    "./datasets/Binlabel26.csv",
    "./datasets/Binlabel27.csv",
    "./datasets/Binlabel28.csv",
    "./datasets/Binlabel29.csv",
    "./datasets/Binlabel30.csv",
    "./datasets/Binlabel31.csv",
    "./datasets/Binlabel32.csv",
    "./datasets/Binlabel33.csv",
    "./datasets/Binlabel34.csv",
    "./datasets/Binlabel35.csv",
    "./datasets/Binlabel36.csv",
    "./datasets/Binlabel37.csv",
    "./datasets/Binlabel38.csv",
    "./datasets/Binlabel39.csv",
    "./datasets/Binlabel40.csv",
    "./datasets/Binlabel41.csv",
    "./datasets/Binlabel42.csv",
    "./datasets/Binlabel43.csv",
    "./datasets/Binlabel44.csv",
    "./datasets/Binlabel45.csv",
    "./datasets/Binlabel46.csv",
    "./datasets/Binlabel47.csv",
    "./datasets/Binlabel48.csv",
    # "./datasets/Binlabel49.csv",
    # "./datasets/Binlabel50.csv",
    # "./datasets/Binlabel51.csv",
    "./datasets/Binlabel52.csv",
    "./datasets/Binlabel53.csv",
    "./datasets/Binlabel54.csv",
    "./datasets/Binlabel55.csv",
    "./datasets/Binlabel56.csv",
    "./datasets/Binlabel57.csv",
    "./datasets/Binlabel58.csv",
    "./datasets/Binlabel59.csv",
    "./datasets/Binlabel60.csv",
    "./datasets/Binlabel61.csv",
    "./datasets/Binlabel62.csv",
    "./datasets/Binlabel63.csv",
    "./datasets/Binlabel64.csv",
    "./datasets/Binlabel65.csv",
    # "./datasets/Binlabel66.csv",
    # "./datasets/Binlabel67.csv",
    # the above files are One-Asperity "imcomplete rupture"
    # B 49-51, 66-67 for testing


    "./datasets/Binlabel72.csv",
    "./datasets/Binlabel73.csv",
    "./datasets/Binlabel74.csv",
    "./datasets/Binlabel75.csv",
    "./datasets/Binlabel76.csv",
    "./datasets/Binlabel77.csv",
    "./datasets/Binlabel78.csv",
    # "./datasets/Binlabel79.csv",
    "./datasets/Binlabel80.csv",
    "./datasets/Binlabel81.csv",
    "./datasets/Binlabel82.csv",
    "./datasets/Binlabel83.csv",
    "./datasets/Binlabel84.csv",
    "./datasets/Binlabel85.csv",
    "./datasets/Binlabel86.csv",
    "./datasets/Binlabel87.csv",
    "./datasets/Binlabel88.csv",
    "./datasets/Binlabel89.csv",
    "./datasets/Binlabel90.csv",
    # "./datasets/Binlabel91.csv",
    "./datasets/Binlabel92.csv",
    "./datasets/Binlabel93.csv",
    "./datasets/Binlabel94.csv",
    "./datasets/Binlabel96.csv",
    "./datasets/Binlabel100.csv",
    "./datasets/Binlabel101.csv",
    "./datasets/Binlabel102.csv",
    "./datasets/Binlabel103.csv",
    "./datasets/Binlabel104.csv",
    "./datasets/Binlabel105.csv",
    "./datasets/Binlabel106.csv",
    "./datasets/Binlabel107.csv",
    "./datasets/Binlabel108.csv",
    # "./datasets/Binlabel109.csv",
    "./datasets/Binlabel110.csv",
    "./datasets/Binlabel111.csv",
    "./datasets/Binlabel112.csv",
    "./datasets/Binlabel113.csv",
    "./datasets/Binlabel114.csv",
    "./datasets/Binlabel115.csv",
    "./datasets/Binlabel116.csv",
    "./datasets/Binlabel117.csv",
    "./datasets/Binlabel118.csv",
    # "./datasets/Binlabel119.csv",
    "./datasets/Binlabel120.csv",
    "./datasets/Binlabel121.csv",
    "./datasets/Binlabel122.csv",
    "./datasets/Binlabel123.csv",
    "./datasets/Binlabel124.csv",
    "./datasets/Binlabel125.csv",
    "./datasets/Binlabel126.csv",
    "./datasets/Binlabel127.csv",
    "./datasets/Binlabel128.csv",
    "./datasets/Binlabel129.csv",
    "./datasets/Binlabel130.csv",
    "./datasets/Binlabel131.csv",
    "./datasets/Binlabel132.csv",
    "./datasets/Binlabel133.csv",
    "./datasets/Binlabel134.csv",
    "./datasets/Binlabel135.csv",
    "./datasets/Binlabel136.csv",
    "./datasets/Binlabel137.csv",
    "./datasets/Binlabel138.csv",
    "./datasets/Binlabel139.csv",
    "./datasets/Binlabel140.csv",
    "./datasets/Binlabel141.csv",
    "./datasets/Binlabel142.csv",
    "./datasets/Binlabel143.csv",
    "./datasets/Binlabel144.csv",
    "./datasets/Binlabel145.csv",
    "./datasets/Binlabel146.csv",
    "./datasets/Binlabel147.csv",
    "./datasets/Binlabel148.csv",
    "./datasets/Binlabel149.csv",
    "./datasets/Binlabel150.csv",
    "./datasets/Binlabel151.csv",
    "./datasets/Binlabel152.csv",
    "./datasets/Binlabel153.csv",
    "./datasets/Binlabel154.csv",
    "./datasets/Binlabel155.csv",
    "./datasets/Binlabel156.csv",
    "./datasets/Binlabel157.csv",
    "./datasets/Binlabel158.csv",
    # "./datasets/Binlabel159.csv",
    "./datasets/Binlabel160.csv",
    "./datasets/Binlabel161.csv",
    "./datasets/Binlabel162.csv",
    "./datasets/Binlabel163.csv",
    "./datasets/Binlabel164.csv",
    "./datasets/Binlabel165.csv",
    "./datasets/Binlabel166.csv",
    "./datasets/Binlabel167.csv",
    "./datasets/Binlabel168.csv",
    "./datasets/Binlabel169.csv",
    "./datasets/Binlabel170.csv",
    "./datasets/Binlabel171.csv",
    "./datasets/Binlabel172.csv",
    "./datasets/Binlabel173.csv",
    "./datasets/Binlabel174.csv",
    "./datasets/Binlabel175.csv",
    "./datasets/Binlabel176.csv",
    "./datasets/Binlabel177.csv",
    "./datasets/Binlabel178.csv",
    "./datasets/Binlabel179.csv",
    "./datasets/Binlabel180.csv",
    "./datasets/Binlabel181.csv",
    "./datasets/Binlabel182.csv",
    "./datasets/Binlabel183.csv",
    # "./datasets/Binlabel184.csv",
    "./datasets/Binlabel185.csv",
    "./datasets/Binlabel186.csv",
    "./datasets/Binlabel187.csv",
    "./datasets/Binlabel188.csv",
    "./datasets/Binlabel189.csv",
    # the above files are Two-Asperity "imcomplete rupture"
    # B66-67, 79, 91, 109, 119, 184, and 185-189 for testing
]


csv_files_100lines = [
    # "./datasets/inlabel30.csv",
    # "./datasets/inlabel31.csv",
    # "./datasets/inlabel32.csv",
    # "./datasets/inlabel33.csv",
    # "./datasets/inlabel34.csv",
    # "./datasets/inlabel35.csv",
    # "./datasets/inlabel36.csv",
    # "./datasets/inlabel37.csv",
    # "./datasets/inlabel38.csv",
    # "./datasets/inlabel39.csv",
    # one sample takes 100 lines here
]
