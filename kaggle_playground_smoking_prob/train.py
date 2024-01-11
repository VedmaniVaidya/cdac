from data_utils import get_data_loaders, prepare_train, get_cv_data_loaders, get_train_dataloaders
from engine import training_step, validation_step
from utils import print_trainable_parameters
from model import ClsModel1, ClsModel2, ClsModel3, save_model, load_model

import pandas as pd
import torch
from torch import nn


# def k_fold_cross_validation_train():
#     df=pd.read_csv("data/train.csv", index_col='id')
#     X_train, y_train, scaler = prepare_train(df)
#     for fold, (train_dataloader, val_dataloader) in enumerate(get_cv_data_loaders(X_train, y_train, n_splits=5, batch_size=128)):
#         print(f"Fold: {fold}")
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         torch.manual_seed(0)
#         model  = ClsModel2(input_size=21, hidden_size=50,num_classes=1)
#         optimzer = torch.optim.Adam(model.parameters(), lr=0.0001)
#         loss_fn = nn.BCELoss()
#         from torchmetrics import AUROC,Accuracy
#         auroc = AUROC(num_classes=3,task='binary')
#         accuracy = Accuracy(task='binary')
#         train,val = get_data_loaders(X_train, y_train, batch_size=64)
        

#         for i in range(50):
#             print(f"Epoch: {i}")
#             average_loss, model, optimizer, scheduler = training_step(model, train_dataloader, loss_fn, optimzer, device)
#             print(f"Average loss: {average_loss}")
#             metrics = validation_step(model, val_dataloader, device, rocauc=auroc, accuracy=accuracy)
#             print(f"Validation metrics: {metrics}")
#             save_model(model=model, optimizer=optimizer, epoch=i,loss=average_loss,filename=f"model_{fold}")
        


def train_no_val():
    df=pd.read_csv("data/train.csv", index_col='id')
    X_train, y_train, scaler = prepare_train(df)
    train_dataloader = get_train_dataloaders(X_train, y_train, batch_size=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    model  = ClsModel3(input_size=21, hidden_size=50,num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    from torchmetrics import AUROC,Accuracy
    auroc = AUROC(num_classes=None,task='binary')
    accuracy = Accuracy(task='binary')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True)
    epoch = 0
    # model, optimizer, epoch, loss, scheduler = load_model(model, optimizer, filename="cls_model_3.pt", scheduler=scheduler)
    for i in range(epoch+1,epoch+1+100):
            print(f"Epoch: {i}")
            average_loss, model, optimizer = training_step(model, train_dataloader, loss_fn, optimizer, device)
            print(f"Average loss: {average_loss}")
            metrics = validation_step(model, train_dataloader, device, rocauc=auroc, accuracy=accuracy)
            print(f"Validation metrics: {metrics}")
            scheduler.step(metrics['rocauc'])
            save_model(model=model, optimizer=optimizer, epoch=i,loss=average_loss,filename=f"cls_model_3.pt", scheduler=scheduler)


def train(fold, train_dataloader,val_dataloader, num_epochs, path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    model  = ClsModel3(input_size=21, hidden_size=50,num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    from torchmetrics import AUROC,Accuracy
    auroc = AUROC(num_classes=1,task='binary')
    accuracy = Accuracy(task='binary')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True)
    epoch = 0
    if path is not None:
        model, optimizer, epoch, loss, scheduler = load_model(model, optimizer, filename="cls_model_3.pt", scheduler=scheduler)
    for i in range(epoch+1,epoch+1+num_epochs):
            print(f"\nEpoch: {i}")
            average_loss, model, optimizer = training_step(model, train_dataloader, loss_fn, optimizer, device)
            print(f"Average loss: {average_loss}")
            metrics = validation_step(model, val_dataloader, device, rocauc=auroc, accuracy=accuracy)
            print(f"Validation metrics: {metrics}")
            if scheduler is not None:
                scheduler.step(metrics['rocauc'])
            save_model(model=model, optimizer=optimizer, epoch=i,loss=average_loss,filename=f"fold_{fold}_cls_model_3.pt", scheduler=scheduler)



def k_fold_train():
    df=pd.read_csv("data/train.csv", index_col='id')
    X_train, y_train, scaler = prepare_train(df)
    for fold, (train_dataloader, val_dataloader) in enumerate(get_cv_data_loaders(X_train, y_train, n_splits=5, batch_size=128)):
        print(f"Fold: {fold}")
        train(fold, train_dataloader, val_dataloader,num_epochs=10)


k_fold_train()