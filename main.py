import os

import pandas as pd
from args import *

# 1. get arguments

args = get_args()
print(args.epochs)

# 2. Iterate over each fold

for fold in range(5):
    train_set = pd.read_csv(os.path.join(args.csv_dir, f"train_fold_{fold}.csv"))
    val_set = pd.read_csv(os.path.join(args.csv_dir, f"val_fold_{fold}.csv"))

