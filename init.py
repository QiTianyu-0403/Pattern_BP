import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from model.BP import *
from model.CNN import *
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plot
from torch.utils.data import random_split, DataLoader, TensorDataset
import numpy as np


def init(args):
    target_url = './data/glass.csv'
    
    ## 读取数据集
    glass = pd.read_csv(target_url,header=None,prefix="V")
    glass.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']
    print(glass.head())
    
    ## 数据集统计
    summary = glass.describe()
    # print(summary)
    ncol1 = len(glass.columns)
    
    for i in range(glass.shape[0]):
        glass.iloc[i, 10] = glass.iloc[i, 10] - 1
        if glass.iloc[i, 10] > 3:
            glass.iloc[i, 10] = glass.iloc[i, 10] - 1
    
    print(glass)
    ## 去掉Id列
    glassNormalized = glass.iloc[:, 1:ncol1]   
    ncol2 = len(glassNormalized.columns)
    summary2 = glassNormalized.describe()
    # print(summary2)
    
    ## 归一化
    for i in range(ncol2):
        mean = summary2.iloc[1, i]
        sd = summary2.iloc[2, i]
        glassNormalized.iloc[:,i:(i + 1)] = (glassNormalized.iloc[:,i:(i + 1)] - mean) / sd
        
    # distribute
    array = glassNormalized.values
    plot.boxplot(array)
    plot.xlabel("Attribute Index")
    plot.ylabel(("Quartile Ranges - Normalized "))
    plot.savefig('./plots/boxplot.pdf')
    
    # data set
    input_data = glass.iloc[:, 1:ncol1-1]
    output_data = glass.loc[:, 'Type']
    
    input_data = input_data.to_numpy(dtype=np.float32)
    output_data = output_data.to_numpy(dtype=np.float32)
    if args.model == 'CNN':
        input_data = np.expand_dims(input_data, -1)
    input_data = torch.Tensor(input_data)
    output_data = torch.tensor(output_data, dtype=torch.long)
    print(input_data.shape)
    dataset = TensorDataset(input_data, output_data)
    
    print('\nInput format: ', input_data.shape, input_data.dtype)
    print('\nOutput format: ', output_data.shape, output_data.dtype)
    
    number_rows = len(input_data)
    test_split = int(number_rows*0.3) 
    train_split = number_rows - test_split
    trainset, testset = random_split(dataset, [train_split, test_split])
    
    print(trainset)
    
    trainloader = DataLoader(trainset, batch_size = args.batchsize, shuffle = True)
    testloader = DataLoader(testset, batch_size = 1)
    
    device = torch.device("cpu")
    if args.model == 'BP':
        net = BPNet().to(device)
    if args.model == 'CNN':
        net = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    
    return device, trainloader, testloader, net, criterion, optimizer

