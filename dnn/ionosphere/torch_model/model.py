from torch import nn
import torch
from utils import print_trainable_parameters


class BaseLine(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.3):
        super().__init__()
        self.model = nn.Sequential(
          nn.Linear(in_features=input_size, out_features=hidden_size*2, dtype=torch.float64),
          nn.LeakyReLU(),
          nn.BatchNorm1d(hidden_size*2,dtype=torch.float64),
          nn.Linear(in_features=hidden_size*2, out_features=hidden_size*2, dtype=torch.float64),
          nn.LeakyReLU(),
          nn.BatchNorm1d(hidden_size*2, dtype=torch.float64),
          nn.Linear(in_features=hidden_size*2, out_features=hidden_size, dtype=torch.float64),
          nn.LeakyReLU(),
          nn.BatchNorm1d(hidden_size, dtype=torch.float64),
          nn.Dropout(dropout_rate),
          nn.Linear(in_features=hidden_size, out_features=num_classes, dtype=torch.float64),
          nn.Sigmoid()
      )
        
        print_trainable_parameters(self)

    def forward(self, data):
        return self.model(data)
    


def save_model(model, optimizer, epoch, loss, filename, scheduler=None):
        path = filename + '.pt'
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(save_dict, path)
    


def load_model(model, optimizer, filename, scheduler=None):
    path = filename
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.eval()
    return model, optimizer, epoch, loss, scheduler