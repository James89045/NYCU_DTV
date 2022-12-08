from train import VGGNET
from torch.utils import data
import torch
import os
import csv
import pandas as pd
import numpy as np
from dataloader import testloader

name_list = os.listdir('Lab1_dataset\\test\\test')
#print(name_list)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
#print("running on ", device)
test_loader = testloader(root="Lab1_dataset\\test\\test", name_list = name_list)
test_data = data.DataLoader(test_loader, batch_size = 1, shuffle=False)
model = VGGNET(in_channels = 3, num_classes = 10)
model = model.to(device)
model.load_state_dict(torch.load('HW1_311605011.pt'))

def evaluate():
    model.eval()
    if os.path.exists('HW1_311605011.csv'):
            os.remove('HW1_311605011.csv')
    with open('HW1_311605011.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['names', 'label'])
    for idx, img in enumerate(test_data):
        img = img.to(device)
        test_pred = model(img)
        test_pred = test_pred.argmax(dim = 1)
        test_pred = test_pred.cpu().numpy().tolist()
        #print(test_pred)
            
        with open('HW1_311605011.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([name_list[idx], str(test_pred[0])])

def sort():
    names = pd.read_csv('HW1_311605011.csv', usecols = ['names'])
    #print(names.shape)
    names = np.squeeze(names)
    #print(names.shape)
    names = names.tolist()
    label = pd.read_csv('HW1_311605011.csv', usecols = ['label'])
    label = np.squeeze(label)
    label = label.tolist()
    
    a = zip(names, label)
    csvlist=[]
    for i in zip(names, label):
        csvlist.append(i)


    for i in range(len(csvlist)):
        for j in range(len(csvlist)):
            if ((j+1 < len(csvlist) and int(csvlist[j][0][:-4]) > int(csvlist[j+1][0][:-4]))):
                temp = csvlist[j]
                csvlist[j] = csvlist[j+1]
                csvlist[j+1] = temp
    
    with open('HW1_311605011.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['names', 'label'])

    for i in csvlist:
        with open('HW1_311605011.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([i[0], str(i[1])])
        
evaluate()
sort()