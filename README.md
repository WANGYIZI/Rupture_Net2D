# RuptureNet2D
Here is our source Python codes for <strong>training</strong> and <strong>testing</strong> of RuptureNet2D.

## Dependencies
Run
```
cd RuptureNet2D
pip -r requirements.txt'
```
to install the necessary requirements.

## Usage

Place the training data into `RuptureNet2D/datasets`.

Use
```
python train_raytune.py --help
```
and
```
python test_raytune.py --help
```
for information on running training and testing, respectively. Both methods use `raytune` for hyperparameter- and gridsearch.

for alternative testing<br>
In order to facilitate plotting and measuring some other indicators, we also prepared a ***test.ipynb*** template here<br>

The results will be placed in a folder `results`. A folder `cache` will be created to cache the training and test data.

If you have any questions, please contact author Ziyi Wang at ziyi57161@gmail.com.
