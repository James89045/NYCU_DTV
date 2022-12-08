import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchsummary import summary
import pandas as pd
import numpy as np
import argparse
from torch.utils import data
from dataloader import dataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default = 0.001, type = float, help = 'learning rate')
    parser.add_argument('--batch_size', default = 64, type = int, help = 'batch size')
    parser.add_argument('--epochs', default = 40, type = int, help = 'epochs')
    args = parser.parse_args()
    return args


class VGGNET(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super(VGGNET, self).__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels = self.in_channels, out_channels = 8, kernel_size = (3, 3), stride = 1, padding = (1, 1)),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = (3, 3), stride = 1, padding = (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = (1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1,15),stride=(1,15)),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1,52),stride=1),
        )

    def forward(self, x):
        x = self.net(x)
        #print(x.size())
        x = x.view(x.shape[0],32,1,-1)
        x = self.conv2(x)
        x = x.view(x.shape[0],-1)
        return x


abc = VGGNET().cuda()
summary(abc, (3, 224, 224))
x = torch.randn(64, 3, 224, 224).cuda()
#abc = VGGNET()
print(abc(x).size())      


#dataloader

args = parse_args()
train_data = dataloader(root = 'Lab1_dataset\\train\\train', mode = 'train')
train_loader = data.DataLoader(train_data, batch_size = args.batch_size, shuffle= True)
val_data = dataloader(root = 'Lab1_dataset\\val\\val', mode = 'val')
val_loader = data.DataLoader(val_data, batch_size = args.batch_size, shuffle = True)


def train(args, train_loader, val_loader, train_data, val_data):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    print("running on ", device)
    loss = nn.CrossEntropyLoss()
    model = VGGNET(in_channels = 3, num_classes = 10)
    para_sum = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("parameter sum : ", para_sum)
    model = model.to(device)
    train_acc = []#saving accuracy of each epoch
    train_loss_list = []#saving loss of each epoch
    val_loss_list  = []
    val_acc = []

    for i in range(1, 1 + args.epochs):
        model.train()
        correct_train = 0
        iter_loss = 0
        correct_val = 0
        for idx , (trains, labels) in enumerate(train_loader):
            trains = trains.to(device, dtype = torch.float)
            labels = labels.to(device, dtype = torch.long)
            optimizer = optim.Adam(model.parameters(), lr = args.lr)
            pred = model(trains)
            Loss = loss(pred, labels)
            iter_loss += loss(pred, labels)
            pred = pred.argmax(dim = 1)
            correct_train += len(pred[pred == labels])

            #update
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

        train_acc.append(correct_train/len(train_data))
        train_loss_list.append(iter_loss.detach().cpu().numpy()/len(train_data))
        print("training accuracy: ", correct_train/len(train_data))
        print("Loss: ", iter_loss/len(train_data))

        val_loss = 0
        model.eval()
        for idx, (val, val_labels) in enumerate(val_loader):
            val = val.to(device, dtype = torch.float)
            val_labels = val_labels.to(device, dtype = torch.long)
            val_pred = model(val)
            val_loss += loss(val_pred, val_labels)
            val_pred = val_pred.argmax(dim = 1)
            correct_val += len(val_pred[val_pred == val_labels])
        
        val_loss_list.append(val_loss.detach().cpu().numpy()/len(val_data))
        val_acc.append(correct_val/len(val_data))
        print("validation accuracy: ", correct_val/len(val_data))

    
    #save model weight
    torch.save(model.state_dict(), 'HW1_311605011.pt')

    
    ###plot###
    plt.plot(range(1, args.epochs+1), np.array(train_acc), label = 'training accuracy')
    plt.plot(range(1, args.epochs+1), np.array(val_acc), label = 'validation accuracy')
    plt.legend(loc = 'lower right')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.show()

    plt.plot(range(1, args.epochs+1), np.array(train_loss_list), label = 'training loss')
    plt.plot(range(1, args.epochs+1), np.array(val_loss_list), label = 'validation loss')
    plt.legend(loc = 'lower right')
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.show()


def main():
    #pass
    train(args, train_loader, val_loader, train_data, val_data)


if __name__ == "__main__":
    main()
