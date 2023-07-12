import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm.auto import tqdm
bar_format = '{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime

from model.model import deepTarget
from dataset.dataset import TrainDataset, Dataset
from torch.utils.data import DataLoader
from utils.sequence import postprocess_result


def train_model(mirna_fasta_file, mrna_fasta_file, train_file, model=None, cts_size=30, seed_match='offset-9-mer-m7', level='gene', batch_size=32, epochs=10, save_file=None, device='gpu'):
    """
    if not isinstance(model, deepTarget):
        raise ValueError("'model' expected <nn.Module 'deepTarget'>, got {}".format(type(model)))

    print("\n[TRAIN] {}".format(model.name))
    """

    if train_file.split('/')[-1] == 'train_set.csv':
        train_set = TrainDataset(train_file)
    else:
        # 实例化
        # train_set = Dataset(mirna_fasta_file, mrna_fasta_file, train_file, seed_match=seed_match, header=True, train=True)  # return (mirna, mrna), label
        train_set = TrainDataset(train_file, cts_size)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

    class_weight = torch.Tensor(compute_class_weight('balanced', classes=np.unique(train_set.labels), y=train_set.labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    '''STLR学习率'''
    '''
    # 定义优化器和学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # 设置初始学习率为2e-5
    # 设置STLR的超参数
    num_epochs = epochs  # 总共的训练轮数
    num_train_steps = len(train_loader) * num_epochs  # 计算总的训练步数
    warmup_proportion = 0.1  # warmup步数占总步数的比例
    num_warmup_steps = int(num_train_steps * warmup_proportion)  # 计算warmup步数
    # 创建STLR学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)
    '''

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)        # default: lr0=1e-3 # cnn/lstm:lr1=1e-4  # fc: lr2=3e-5
    # scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=2,T_mult=2)  # lr1
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-5)
    # scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.5, last_epoch=-1, min_lr=5e-5)


    model = model.to(device)

    for epoch in range(epochs):
        epoch_loss, corrects = 0, 0

        with tqdm(train_loader, desc="Epoch {}/{}".format(epoch+1, epochs), bar_format=bar_format) as tqdm_loader:
            for i, ((mirna, mrna), label) in enumerate(tqdm_loader):
                mirna, mrna, label = mirna.to(device, dtype=torch.int64), mrna.to(device, dtype=torch.int64), label.to(device)
                outputs = model(mirna, mrna)

            #for i, (rna, label) in enumerate(tqdm_loader):
                #rna, label = rna.to(device, dtype=torch.int64), label.to(device)
                #outputs = model(rna)

                loss = criterion(outputs, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                epoch_loss += loss.item() * outputs.size(0)
                corrects += (torch.max(outputs, 1)[1] == label).sum().item()

                if (i+1) == len(train_loader):
                    tqdm_loader.set_postfix(dict(loss=(epoch_loss/len(train_set)), acc=(corrects/len(train_set))))
                else:
                    tqdm_loader.set_postfix(loss=loss.item())

    if save_file is None:
        time = datetime.now()
        save_file = "{}.pt".format(time.strftime('%Y%m%d_%H%M%S_weights'))
    torch.save(model.state_dict(), save_file)
    
    
def predict_result(mirna_fasta_file, mrna_fasta_file, query_file, model=None, weight_file=None, seed_match='offset-9-mer-m7', level='gene', batch_size=32, output_file=None, device='cpu'):
    """
    if not isinstance(model, deepTarget):
        raise ValueError("'model' expected <nn.Module 'deepTarget'>, got {}".format(type(model)))
    """

    if not weight_file.endswith('.pt'):
        raise ValueError("'weight_file' expected '*.pt', got {}".format(weight_file))

    model.load_state_dict(torch.load(weight_file))

    test_set = Dataset2(mirna_fasta_file, mrna_fasta_file, query_file, seed_match=seed_match, header=True, train=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    y_probs = []
    y_predicts = []
    y_truth = []

    model = model.to(device)
    with torch.no_grad():
        model.eval()

        with tqdm(test_loader, bar_format=bar_format) as tqdm_loader:
            for i, ((mirna, mrna), label) in enumerate(tqdm_loader):
                mirna, mrna, label = mirna.to(device, dtype=torch.int64), mrna.to(device, dtype=torch.int64), label.to(device)
                outputs = model(mirna, mrna)

            #for i, (rna, label) in enumerate(tqdm_loader):
                #rna, label = rna.to(device, dtype=torch.int64), label.to(device)
                #outputs = model(rna)

                _, predicts = torch.max(outputs.data, 1)
                probabilities = F.softmax(outputs, dim=1)

                y_probs.extend(probabilities.cpu().numpy()[:, 1])
                y_predicts.extend(predicts.cpu().numpy())
                y_truth.extend(label.cpu().numpy())


                global correct
                # print(predicts.cpu().numpy())
                # print(label.cpu().numpy())
                correct += (predicts == label).sum().item()

        acc = float(correct / len(test_set)) * 100
        print(len(test_set))
        print("acc:", acc, "%")
        #print(classification_report(y_truth, y_predicts, digits=4))

        '''
        if output_file is None:
            time = datetime.now()
            output_file = "result/"+"{}.csv".format(time.strftime('%Y%m%d_%H%M%S_results'))
        results = postprocess_result(test_set.dataset, y_probs, y_predicts,
                                     seed_match=seed_match, level=level, output_file=output_file)

        # print(results)
        '''
    return acc
