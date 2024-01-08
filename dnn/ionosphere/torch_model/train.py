from data_utils import prepare_data, get_cv_data_loaders
from engine import training_step, validation_step
from model import save_model, load_model, BaseLine
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import torch
from torch import nn

        


def train(fold, train_dataloader,val_dataloader, num_epochs, path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    model  = BaseLine(input_size=33, hidden_size=15,num_classes=1, dropout_rate=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    from torchmetrics import AUROC,Accuracy
    auroc = AUROC(num_classes=1,task='binary')
    accuracy = Accuracy(task='binary')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True)
    epoch = 0
    loss_list = []
    metrics_list = []
    if path is not None:
        model, optimizer, epoch, loss, scheduler = load_model(model, optimizer, filename=path, scheduler=scheduler)
    for i in range(epoch+1,epoch+1+num_epochs):
            print(f"\nEpoch: {i}")
            average_loss, model, optimizer = training_step(model, train_dataloader, loss_fn, optimizer, device)
            print(f"Average loss: {average_loss}")
            metrics = validation_step(model, val_dataloader, device, rocauc=auroc, accuracy=accuracy, loss=loss_fn)
            print(f"Validation metrics: {metrics}")
            if scheduler is not None:
                scheduler.step(metrics['rocauc'])
            save_model(model=model, optimizer=optimizer, epoch=i,loss=average_loss,filename=f"fold_{fold}_model", scheduler=scheduler)
            loss_list.append(average_loss)
            metrics_list.append(metrics)
    return loss_list, metrics_list

def k_fold_train():
    # df=pd.read_csv("data/train.csv", index_col='id')
    X_train, y_train, scaler = prepare_data()
    loss_list_list = []
    metrics_list_list = []
    for fold, (train_dataloader, val_dataloader) in enumerate(get_cv_data_loaders(X_train, y_train, n_splits=5, batch_size=32)):
        print(f"Fold: {fold}")
        loss_list, metrics_list = train(fold, train_dataloader, val_dataloader, num_epochs=30)
        loss_list_list.append(loss_list)
        metrics_list_list.append(metrics_list)
        #TODO remove this break
        break #only for testing
    return loss_list_list, metrics_list_list

        


if __name__ == "__main__":
    loss_list_list, metrics_list_list = k_fold_train()
    #save the loss and metrics list
    # import pickle
    # with open('loss_list_list.pkl', 'wb') as f:
    #     pickle.dump(loss_list_list, f)

    # with open('metrics_list_list.pkl', 'wb') as f:
    #     pickle.dump(metrics_list_list, f)