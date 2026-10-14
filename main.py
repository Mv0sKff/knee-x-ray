import os
from datetime import datetime

import pandas as pd
from args import *
from dataset import knee_xray_dataset
from model import MyModel
from torch.utils.data import DataLoader
from trainer import train_model

# 1. get arguments
args = get_args()
print(args.epochs)
run_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")

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
    session_name = f"{run_name}_fold_{fold}"
    train_model(model, train_loader, val_loader, session_name=session_name, save=True)

    for i, batch in enumerate(train_loader):
        # print(i, batch["target"])
        pass
