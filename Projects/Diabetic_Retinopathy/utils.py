import torch
import pandas as pd 
import numpy as np
import config
from tqdm import tqdm

def make_prediction(model, loader, output_csv="submission.csv"):
    preds = []
    filenames = []
    model.eval()

    for x, y, file in tqdm(loader):
        x.to(config.DEVICE)
        with torch.no_grad():
            pred = model(x).argmax(1)
            preds.append(pred.cpu().numpy())
            filenames.append(file)

    df = pd.DataFrame({"image": filenames, "level": np.concatenate(preds, axis=0)})
    df.to_csv(output_csv, index=False)
    model.train()
    print('done with prediction')

def check_accuracy(loader, model, device=config.DEVICE):
    model.eval()
    all_preds = []
    all_labels = []
    num_correct = 0
    num_total = 0

    for x, y, file in tqdm(loader):
        x.to(device)
        y.to(device)

        with torch.no_grad():
            scores = model(x)

        _, predictions = scores.max(1)
        num_correct += (predictions == y).sum()
        num_total += predictions.shape[0]

        all_preds.append(predictions.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())

        print(f'{num_correct / num_total * 100}%')

    model.train()
    
    return np.concatenate(all_preds, axis=0, dtype=np.int64), np.concatenate(all_labels, axis=0, dtype=np.int64)