from model import save_model
import torch
from tqdm import tqdm


from tqdm import tqdm

def training_step(model, train_loader, loss_fn, optimizer, device):
    #Training
    model.train()
    accumulated_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for idx, batch in progress_bar:
        try:
            X, y = batch
            X = X.to(device)
            y = y.to(device)
            preds = model(X)
            loss = loss_fn(preds, y.reshape(-1,1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            accumulated_loss += batch_loss
            progress_bar.set_description(f'Loss: {batch_loss:0.3f}')
        except Exception as e:
            print(f'Error occurred in batch {idx}: {e}')
    average_loss = accumulated_loss/len(train_loader)
    return average_loss, model, optimizer


def validation_step(model, val_loader, device, **metric):
    model.eval()
    accumulated_metrics = {name: 0 for name in metric.keys()}
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        for idx, batch in progress_bar:
            try:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                preds = model(X)
                metric_values = []
                for name, fn in metric.items():
                    metric_value = fn(preds, y.reshape(-1,1))
                    accumulated_metrics[name] += metric_value.item()
                    metric_values.append(f'{name}: {metric_value:0.3f}')
                progress_bar.set_description('| '.join(metric_values))
            except Exception as e:
                print(f'Error occurred in batch {idx}: {e}')
    average_metrics = {name: value/len(val_loader) for name, value in accumulated_metrics.items()}
    return average_metrics