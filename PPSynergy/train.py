# Added F1_score metric
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from PPSynergy import PPSynergyNet

from model_function import *
import os
import random
import numpy as np
import pandas as pd
import torch.utils.data as Data
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score,f1_score


# CPU or GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("\nThe code uses GPU...")
else:
    device = torch.device("cpu")
    print("\nThe code uses CPU!!!")

# ---- Dataset and best-model selection metric (command-line arguments) ----
_parser = argparse.ArgumentParser(description="Train CNN/DiffusionCNN for drug synergy prediction.")
_parser.add_argument('--dataset', choices=['ONEIL', 'NCI-ALMANAC'], default='ONEIL',
                     help="dataset to use: ONEIL or NCI-ALMANAC (default: ONEIL)")
_parser.add_argument('--metric', choices=['AUC+ACC', 'AUPR+F1'], default=None,
                     help="best-model selection metric (default: auto -> ONEIL uses AUC+ACC, NCI-ALMANAC uses AUPR+F1)")
_args = _parser.parse_args()
DATASET = _args.dataset
METRIC = _args.metric or ('AUPR+F1' if DATASET == 'NCI-ALMANAC' else 'AUC+ACC')
print(f'Dataset: {DATASET} | Best-model selection metric: {METRIC}')

# ---- Project root and result dir (relative paths so the project is portable) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # project root
RESULT_DIR = os.path.join(BASE_DIR, 'result', DATASET)          # result/<dataset>/
os.makedirs(RESULT_DIR, exist_ok=True)                          # auto-create if missing

x = np.load(os.path.join(BASE_DIR, 'data', f'{DATASET}.npy'), allow_pickle=True)
label = pd.read_csv(os.path.join(BASE_DIR, 'data', DATASET, f'{DATASET}_socre.csv'))
y = np.array(label['label']) 

lenth = len(x)
pot = int(lenth / 5)
print('lenth', lenth)
print('pot', pot)

def seed_everything(seed_value):
    random.seed(seed_value)
    # np.random.seed(seed_value)
    # torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def model_selection_score(metrics_dict, criterion):
    """Return the best-model selection score for the given criterion.

    metrics_dict: dict with keys AUC, PR_AUC, ACC, f1_scores, ...
    criterion:    'AUC+ACC' -> AUC + ACC
                  'AUPR+F1' -> PR_AUC + f1_scores
    """
    if criterion == 'AUC+ACC':
        return metrics_dict['AUC'] + metrics_dict['ACC']
    if criterion == 'AUPR+F1':
        return metrics_dict['PR_AUC'] + metrics_dict['f1_scores']
    raise ValueError(f'Unknown selection metric: {criterion}')


seed = 42
seed_everything(seed)
random_num = random.sample(range(0, lenth), lenth)

# Collect the best metrics of each fold (for the 5-fold summary)
all_fold_best = []
metric_keys = ['AUC', 'PR_AUC', 'ACC', 'BACC', 'PREC', 'TPR', 'KAPPA', 'RECALL', 'f1_scores']

for i_time in range(5):
    ia=(384,196)
    model = PPSynergyNet(image_size=ia)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()

    test_num = random_num[pot * i_time:pot * (i_time + 1)]
    train_num = random_num[:pot * i_time] + random_num[pot * (i_time + 1):]

    x_train = x[train_num]
    x_test = x[test_num]
    x_train = torch.tensor(x_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)

    y_train = y[train_num]
    y_test = y[test_num]


    y_train =  torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    batch_size = 115
    train_torch_dataset = Data.TensorDataset(x_train, y_train)
    test_torch_dataset = Data.TensorDataset(x_test, y_test)
    train_loader = Data.DataLoader(
        dataset=train_torch_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = Data.DataLoader(
        dataset=test_torch_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f'i_time: {i_time+1}')
    
    best_score = -1.0
    best_epoch = -1
    total_epochs = 200

    for epoch in range(total_epochs):
        train_loss, train_acc, train_auc = train(model, device, train_loader, optimizer, loss_fn)
        test_loss, test_acc, test_auc = test(model, device, test_loader, loss_fn)
        T, S, Y = predicting(model, device, test_loader)  # T is the correct label; S is the predicted score; Y is the predicted label
        
        # compute performance
        AUC = roc_auc_score(T, S)
        precision, recall, threshold = metrics.precision_recall_curve(T, S)
        fpr, tpr, thresholds = metrics.roc_curve(T, S, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        PR_AUC = metrics.auc(recall, precision)
        BACC = balanced_accuracy_score(T, Y)
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
        PREC = precision_score(T, Y, zero_division=0)
        ACC = accuracy_score(T, Y)
        KAPPA = cohen_kappa_score(T, Y)
        RECALL = recall_score(T, Y, zero_division=0)
        f1_scores = f1_score(T, Y, zero_division=0)

        # Write per-epoch metrics to the AUCs file
        file_AUCs = os.path.join(RESULT_DIR, f'_{i_time}--AUCs--.txt')
        # Write the header only when the file is newly created (avoid duplicates)
        if not os.path.exists(file_AUCs) or os.path.getsize(file_AUCs) == 0:
            with open(file_AUCs, 'a') as f:
                f.write('Epoch\tAUC_dev\tPR_AUC\tACC\tBACC\tPREC\tTPR\tKAPPA\tRECALL\tf1_scores\n')
        AUCs = [epoch, AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL, f1_scores]
        save_AUCs(AUCs, file_AUCs)

        print('i_time: ', i_time, 'Epoch: ', epoch, 
              '|train_loss: ', train_loss, '| accuracy_train: ', train_acc,  
              '|train_auc: ', train_auc, '|test_loss: ', test_loss, 
              '| accuracy_test: ', test_acc)
  
        # Write per-epoch loss/acc to the results file
        file_results = os.path.join(RESULT_DIR, f'_{i_time}.txt')
        with open(file_results, 'a') as f:
            f.write(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f},'
                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n')

        # ================== Save the best model of the current fold ==================
        metric_dict = {
            'AUC': AUC, 'PR_AUC': PR_AUC, 'ACC': ACC, 'BACC': BACC,
            'PREC': PREC, 'TPR': TPR, 'KAPPA': KAPPA, 'RECALL': RECALL,
            'f1_scores': f1_scores,
        }
        current_score = model_selection_score(metric_dict, METRIC)
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            # Record all metrics at the best epoch of this fold (for the 5-fold summary)
            best_metrics = dict(metric_dict)
            best_metrics['epoch'] = epoch
            best_metrics[METRIC] = current_score

            save_path = os.path.join(RESULT_DIR, f'_{i_time}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'AUC': AUC,
                'ACC': ACC,
                'AUC_ACC_sum': current_score,
                'PR_AUC': PR_AUC,
                'BACC': BACC,
                'PREC': PREC,
                'TPR': TPR,
                'KAPPA': KAPPA,
                'RECALL': RECALL,
                'f1_scores': f1_scores,
            }, save_path)

            print(f'  ★ New best model saved! Epoch {epoch} | {METRIC} = {current_score:.4f} → {save_path}')
        # ===========================================================

    # Print the best result of the current fold
    print(f'>>> Fold {i_time} Finished. Best Epoch = {best_epoch}, Best {METRIC} = {best_score:.4f}')
    all_fold_best.append(best_metrics)

# ================== 5-fold best metrics summary ==================
print('\n========== Best metrics per fold ==========')
header = 'Fold\tEpoch\t' + f'{METRIC}\t' + '\t'.join(metric_keys)
print(header)
summary_lines = [header]
for i, m in enumerate(all_fold_best):
    row = f'{i}\t{m["epoch"]}\t{m[METRIC]:.4f}\t' + '\t'.join(f'{m[k]:.4f}' for k in metric_keys)
    print(row)
    summary_lines.append(row)

summary_path = os.path.join(RESULT_DIR, 'summary_best_metrics.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_lines) + '\n')
print(f'Summary saved to: {summary_path}')

print('\n========== Mean +/- Std over 5 folds (4 decimals) ==========')
for k in metric_keys:
    vals = np.array([m[k] for m in all_fold_best])
    print(f'{k:<10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
