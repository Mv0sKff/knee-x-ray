import os

import pandas as pd
from args import *
from dataset import knee_xray_dataset
from model import MyModel
from torch.utils.data import DataLoader

# 1. get arguments
args = get_args()
print(args.epochs)

# 2. Iterate over each fold

for fold in range(5):
    train_set = pd.read_csv(os.path.join(args.csv_dir, f"train_fold_{fold}.csv"))
    val_set = pd.read_csv(os.path.join(args.csv_dir, f"val_fold_{fold}.csv"))

    # 3. Preparing dataset
    train_dataset = knee_xray_dataset(train_set)
    val_dataset = knee_xray_dataset(val_set)

    # 4. Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # 5. Initializing the model
    model = MyModel(backbone=args.backbone)

    # 6. Train the model


    for i, batch in enumerate(train_loader):
        #print(i, batch["target"])
        pass
