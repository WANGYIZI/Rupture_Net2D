

import time
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import argparse
import numpy as np
import torch.multiprocessing as mp

def read_200_lines(csv_files):
    df = pd.read_csv(csv_files, index_col=False,
                     names=["Lnuc", "X", "StrengthDrop", "StressDrop", "Front", "Slip", "Logic", "StrengthwithNuc"])
    df = df.iloc[::2, :]
    return df

def read_100_lines(csv_files):
    df = pd.read_csv(csv_files, index_col=False,
                     names=["Lnuc", "X", "StrengthDrop", "StressDrop", "Front", "Slip", "Logic", "StrengthwithNuc"])
    return df

# This method is used to divide the data set
def compute_t_times(i, seq_len, prop):
    # compute i * prop
    target_value = i * prop

    # Calculate the nearest integer multiple
    nearest_multiple = round(target_value / seq_len) * seq_len

    return nearest_multiple




class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_pro, hidden_dim):
        super(TransformerEncoderBlock, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_pro = dropout_pro
        self.hidden_dim = hidden_dim
        # Self-Attention
        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim,
                                          bias=False)
        self.key_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value_projection = nn.Linear(self.embed_dim, self.embed_dim,
                                          bias=False)
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                               dropout=self.dropout_pro,
                                               batch_first=True)

        # Feed Forward
        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # Self Attention
        residual1 = x
        transpose = x
        query = self.query_projection(transpose)
        key = self.key_projection(transpose)
        value = self.key_projection(transpose)
        attn_output, _ = self.attention(query, key, value)
        sum1 = residual1 + attn_output  # Add & Norm
        layernorm1 = self.layer_norm(sum1)
        # Feed Forward
        residual2 = layernorm1
        feed_forward = self.feed_forward(residual2)
        sum2 = feed_forward + residual2  # Add & Norm
        output = self.layer_norm(sum2)  # Layer Norm
        return output

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.num_heads = 8
        self.embed_dim = 1024
        self.dropout_pro = 0.1  # for all layers with dropout
        self.hidden_dim = 2048
        self.conv1 = nn.Conv1d(5, 256, kernel_size=1, stride=1, padding=0) #padding = 0, stride = 1, kernel_size = 1
        self.conv2 = nn.Conv1d(256, 256, kernel_size=1, stride=1, padding=0)

        self.conv4 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)

        #embeded_dim of Transformer is 1024
        self.SelfAttention1 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1,hidden_dim=2048)
        self.SelfAttention2 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1,hidden_dim=2048)
        self.SelfAttention3 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1,hidden_dim=2048)
        self.SelfAttention4 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1,hidden_dim=2048)
        self.SelfAttention5 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1,hidden_dim=2048)
        self.SelfAttention6 = TransformerEncoderBlock(embed_dim=1024, num_heads=8, dropout_pro=0.1,hidden_dim=2048)
        self.relu = nn.ReLU()
        self.conv7 = nn.Conv1d(1024,256,kernel_size=1,stride=1,padding=0)
        self.conv8 = nn.Conv1d(256,2,kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #CNN encoder
        transpose1 = torch.transpose(x,1,2)
        conv1 = self.relu(self.conv1(transpose1))
        conv2 = self.relu(self.conv2(conv1))
        cat1 = torch.cat([conv1,conv2],dim = 1)
        conv4 = self.relu(self.conv4(cat1))
        cat2 = torch.cat([cat1,conv4],dim = 1)
        transpose2 = torch.transpose(cat2,1,2)

        #Transformer Processor
        #  1th encoder_block
        block1 = self.SelfAttention1(transpose2)
        # 2nd encoder_block
        block2 = self.SelfAttention2(block1)
        # 3rd encoder_block
        block3 = self.SelfAttention3(block2)
        # 4th encoder_block
        block4 = self.SelfAttention4(block3)
        # 5th encoder_block
        block5 = self.SelfAttention5(block4)
        # 6th encoder_block
        block6 = self.SelfAttention6(block5)

        #CNN decoder
        x = torch.transpose(block6, 1, 2)
        conv7 = self.relu(self.conv7(x))
        conv8 = self.conv8(conv7)
        Front_Slip = torch.transpose(conv8, 1, 2)
        output = self.relu(Front_Slip)
        return output



def createXY(dataset, sample_lines):
    sample_num = len(dataset) // sample_lines
    dataX = []
    dataY = []
    for i in range(0, sample_num):
        dataX.append(dataset[(i) * sample_lines:(i + 1) * sample_lines, [0, 1, 2, 3, -1]])
        dataY.append(dataset[(i) * sample_lines:(i + 1) * sample_lines, 4:6])
    return np.array(dataX), np.array(dataY)


# use this class to implement early-stopping
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=100, verbose=False, delta=0):
        """
        Args:
            save_path : file path to save your model
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.8f} --> {val_loss:.8f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), path)	# The parameters of the best model to date are stored here
        self.val_loss_min = val_loss


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()



if __name__ == '__main__':

    global_fracture = 0
    partial_fracture = 0

    max_Lnuc, min_Lnuc = -np.Inf, np.Inf
    max_X, min_X = -np.Inf, np.Inf
    max_StrengthDrop, min_StrengthDrop = -np.Inf, np.Inf
    max_StressDrop, min_StressDrop = -np.Inf, np.Inf
    max_Front, min_Front = -np.Inf, np.Inf
    max_Slip, min_Slip = -np.Inf, np.Inf
    max_Logic, min_Logic = -np.Inf, np.Inf
    max_StrengthwithNuc, min_StrengthwithNuc = -np.Inf, np.Inf
    max_RuptureEnergy, min_RuptureEnergy = -np.Inf, np.Inf
    mu = 1   #shear elasticity


    # scale range below
    column_max = [16.239269, 97.6, 125.064626, 66.054391, 59.647999, 86.057297, 1, 102.996661, 20855.37742025637]
    # column_max[8] = max(0.5 * Lnuc * StrengthDrop ^ 2)
    column_min = [0.003045, -44.8, 4.100412, -10.37305, 0.0, 0.0, 0, -10.37305, 0.7013907496639992]
    column_max[8] = column_max[8] / (1.158 * mu)
    #now column_max[8] = 0.5 * Lnuc * StrengthDrop ^ 2 / (1.158mu) = fracture energy
    column_min[8] = column_min[8] / (1.158 * mu)


    # The following annotated files are test files
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

    ]

    # Read all csv files for training
    data_frames = []
    for csv_file in csv_files_200lines:
        print(csv_file)
        df = read_200_lines((csv_file))
        if 'A' in csv_file:
            global_fracture += len(df) // 100
        else:
            partial_fracture += len(df) // 100
        df["RuptureEnergy"] = (0.5 * df["Lnuc"] * df["StrengthDrop"] ** 2) / (1.158 * mu)  # RuptureEnergy directly introduced
        df["RuptureEnergy"] = df["RuptureEnergy"].astype(float)  # Converts the data type to floating point
        print(df.shape)
        data_frames.append(df)

    for csv_file in csv_files_100lines:
        print(csv_file)
        df = read_100_lines((csv_file))
        df["Logic"] = 1.0
        df["StrengthwithNuc"] = 20.0
        df["RuptureEnergy"] = (0.5 * df["Lnuc"] * df["StrengthDrop"] ** 2) / (1.158 * mu) # RuptureEnergy directly introduced
        df["RuptureEnergy"] = df["RuptureEnergy"].astype(float)  # Converts the data type to floating point
        print(df.shape)
        data_frames.append(df)
    df = pd.concat(data_frames, axis=0, ignore_index=True)


    print("Global fracture num = " + str(global_fracture))
    print("Partial fracture num = " + str(partial_fracture))
    print(df.shape)

    print(df.head())
    # set random seeds
    random_seed = 307
    np.random.seed(random_seed)

    # n equals rows of df
    n = len(df)

    # block size equals 100 (A sample is 100 rows)
    block_size = 100

    # get the number of blocks
    num_blocks = n // block_size

    # Divide the df object into small blocks(one Block <--> one Sample)
    blocks = [df[i * block_size: (i + 1) * block_size] for i in range(num_blocks)]

    # Shuffle the blocks randomly
    np.random.shuffle(blocks)

    # Combine all block into one
    df_shuffled = pd.concat(blocks, ignore_index=True)
    df = df_shuffled
    print(df.head())



    total_lines = len(df)
    print(total_lines)
    df_for_training1 = df.iloc[0:compute_t_times(total_lines, 100, 0.80)] # train-set: 80%
    df_for_validating1 = df.iloc[compute_t_times(total_lines, 100, 0.80):compute_t_times(total_lines, 100, 1.0)] # validation-set: 20%
    # df_for_testing1=df.iloc[compute_t_times(total_lines,100,1.0):]
    print("Number of training set samples: " + str(len(df_for_training1) // 100))
    print("Number of validation set samples: " + str(len(df_for_validating1) // 100))
    # print("Number of testing set samples: "+str(len(df_for_testing1)//100))




    df_for_training = np.array(df_for_training1)
    df_for_validating = np.array(df_for_validating1)
    # df_for_testing=np.array(df_for_testing1)
    print("train-set scale： " + str(df_for_training.shape))
    print("valid-set scale： " + str(df_for_validating.shape))



    # map each variable to the NN onto the interval [0,1]
    for i in range(0, 9):
        df_for_training[:, i] = (df_for_training[:, i] - column_min[i]) / (column_max[i] - column_min[i])
        df_for_validating[:, i] = (df_for_validating[:, i] - column_min[i]) / (column_max[i] - column_min[i])

    max_Lnuc, min_Lnuc = max(df["Lnuc"]), min(df["Lnuc"])
    max_X, min_X = max(df["X"]), min(df["X"])
    max_StrengthDrop, min_StrengthDrop = max(df["StrengthDrop"]), min(df["StrengthDrop"])
    max_StressDrop, min_StressDrop = max(df["StressDrop"]), min(df["StressDrop"])
    max_Front, min_Front = max(df["Front"]), min(df["Front"])
    max_Slip, min_Slip = max(df["Slip"]), min(df["Slip"])
    max_Logic, min_Logic = max(df["Logic"]), min(df["Logic"])
    max_StrengthwithNuc, min_StrengthwithNuc = max(df["StrengthwithNuc"]), min(df["StrengthwithNuc"])
    max_RuptureEnergy, min_RuptureEnergy = max(df["RuptureEnergy"]), min(df["RuptureEnergy"])
    # Print Maximum and Minimum
    print("Lnuc maximum: " + str(max_Lnuc) + "   minimum: " + str(min_Lnuc))
    print("X maximum: " + str(max_X) + "   minimum: " + str(min_X))
    print("StrengthDrop maximum: " + str(max_StrengthDrop) + "   minimum: " + str(min_StrengthDrop))
    print("StressDrop maximum: " + str(max_StressDrop) + "   minimum: " + str(min_StressDrop))

    print("Front maximum: " + str(max_Front) + "   minimum: " + str(min_Front))
    print("Slip maximum: " + str(max_Slip) + "   minimum: " + str(min_Slip))
    print("Logic maximum: " + str(max_Logic) + "   minimum: " + str(min_Logic))
    print("StrengthwithNuc maximum: " + str(max_StrengthwithNuc) + "   minimum: " + str(min_StrengthwithNuc))
    print("RuptureEnergy maximum: " + str(max_RuptureEnergy) + "   minimum: " + str(min_RuptureEnergy))




    trainX, trainY = createXY(df_for_training, 100)
    validateX, validateY = createXY(df_for_validating, 100)
    # testX,testY=createXY(df_for_testing,100)
    print(trainX.shape, trainY.shape)

    # following code is used for testing the output dimensions of the model
    test_model = MyModel()

    # print(model)
    # random input data
    batch_size = 1
    seq_length = 100
    input_channels = 5
    x = torch.randn(batch_size, seq_length, input_channels)

    # forward propogation and get the output
    output = test_model(x)

    # print the shape of output
    print(output.shape)


    # external arguments
    parser = argparse.ArgumentParser(description="this traning is for testing")
    parser.add_argument('--device_ids', type=str, default="0", help="trainging devices")
    # parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")
    args = parser.parse_args()


    # Transform training, validation, and test sets into custom Dataset objects
    train_dataset = CustomDataset(trainX, trainY)
    validate_dataset = CustomDataset(validateX, validateY)
    # test_dataset = CustomDataset(testX, testY)
    print("Number of training-set samples" + str(len(train_dataset)))

    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    model = MyModel()
    checkpoint = './results/best_network143.pth'
    state_dict = torch.load(checkpoint)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):  # Assume that the model is wrapped in torch.nn.DataParallel before loading
            key = key[7:]  # Remove the 'module.' prefix
        new_state_dict[key] = value

    model.load_state_dict(new_state_dict)

    model.to(device)
    print("successfully loaded model" + ':rank' + str(dist.get_rank()))

    # Create DataLoader of training sets, validation sets, and test sets
    batch_size = 128
    '''
    We use two Gpus for training, 
    each device is assigned a batch_size of 128, 
    and the total batch_size is equivalent to 256
    '''


    sampler_train = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8, sampler=sampler_train)
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    print("successfully loaded data" + ':rank' + str(dist.get_rank()))

    # path to save your model
    save_path = "./results"  # "results" directory to save your model




    # Learning rate decay strategy
    warmup_steps = 4000



    # move the model to GPU
    if torch.cuda.is_available():
        model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    early_stopping = EarlyStopping(save_path)

    # loss function
    criterion = nn.MSELoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    # Record the start time of training
    start_time = time.time()
    # Define a list of loss records
    train_losses = []
    valid_losses = []
    # Defines a list for recording average losses
    mean_train_losses = []
    mean_valid_losses = []



    def lr_lambda(step, model, warmup_steps=4000):
        return (model.module.embed_dim ** -0.5) * min((step + 1) ** (-0.5), (step + 1) * (warmup_steps ** -1.5))

    # Define the learning rate scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, model))

    # Defines an upper limit for the number of epochs
    num_epochs = 2000
    # Actual number of iterations
    actual_iteration = 0
    # Training loop
    for epoch in range(num_epochs):
        sampler_train.set_epoch(epoch)
        # Get train loss
        train_loss = 0.0
        model.train()
        for inputs, labels in train_dataloader:
            # Forward propagation
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # the train loss of current batch
            train_loss += loss.item() * inputs.size(0)

        # Calculate the average training set loss
        train_loss /= (len(train_dataset) // dist.get_world_size() )
        train_loss_tensor = torch.tensor(train_loss).to(device)


        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        mean_train_loss = train_loss_tensor / dist.get_world_size()



        mean_train_losses.append(mean_train_loss.item())

        # The validation set loss calculation is performed in the main process
        if dist.get_rank() == 0:
            # valid loss
            valid_loss = 0.0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validate_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item() * inputs.size(0)

                valid_loss /= len(validate_dataset)

            # Broadcast losses on the main process validation set to other processes
            valid_loss_tensor = torch.tensor(valid_loss).to(device)
            if valid_loss_tensor.is_sparse:
                valid_loss_tensor = valid_loss_tensor.cpu().to_dense().to(device)
            dist.broadcast(valid_loss_tensor, src=0)
            mean_valid_losses.append(valid_loss)
        else:
            # other processes receive broadcast validation set loss
            valid_loss = torch.tensor(0.0).to(device)
            dist.broadcast(valid_loss, src=0)




        # Record losses of training and validation sets
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        actual_iteration = actual_iteration + 1




        # Print the mean losses only from rank 0
        if dist.get_rank() == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs}: Mean Train Loss: {mean_train_loss.item():.8f}, Mean Valid Loss: {valid_loss:.8f}")


        if (epoch % 10 == 0 and dist.get_rank() == 0):
            # Combine training losses and validation losses into a Pandas DataFrame
            loss_data = pd.DataFrame({'train_loss': mean_train_losses, 'valid_loss': mean_valid_losses})
            # print(mean_train_loss)
            # print(valid_loss)
            # Write the DataFrame to the CSV file
            loss_data.to_csv('./results/losses_sugon2cards.csv', index=False)


        # Learning rate decay
        scheduler.step()

        # Release video memory that is no longer needed
        torch.cuda.empty_cache()
        if(dist.get_rank()==0):
            early_stopping(mean_valid_losses[-1], model)
            # early_stop is set to True when the early stop condition is reached
            if early_stopping.early_stop:
                print("Early stopping")
                break  # Exit the iteration and end the training

    import os
    import matplotlib.pyplot as plt
    if(dist.get_rank()==0):
        # actual_iteration、train_losses valid_losses

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
        # plt.show()

        # save your final model
        torch.save(model.state_dict(), 'results/best_weights_final.pth')
        end_time = time.time()
        # total training time
        training_time = end_time - start_time
        print(f"Training time：  {training_time:.2f} seconds")
    cleanup()



#The tutorial for starting the training with the linux command line is as follows
'''
source ~/minconda/bin/activate base   
conda activate pytorch1.11_cuda11.3_cp39
conda activate torch1110cu13
python -m  torch.distributed.run --nproc_per_node=2 --master_port=1238 test.py --device_ids=0,1
salloc -p ksagnormal -N 1 -n 12 --gres=gpu:1
netstat-ntlp | grep 1234
sbatch -p xhhgnormal -N 1 -n 16 -x e03r02 --gres=gpu:2 subtest.sh
'''
