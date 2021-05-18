import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import time
import argparse
from models import *
from misc import progress_bar
from torch import nn

def main():
    parser = argparse.ArgumentParser(description="cifar-10 with PyTorch")
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='number of epochs tp train for')
    parser.add_argument('--trainBatchSize', default=100, type=int, help='training batch size')
    parser.add_argument('--testBatchSize', default=100, type=int, help='testing batch size')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether cuda is in use')
    args = parser.parse_args()

    solver = Solver(args)
    solver.load_model()
    solver.run()

def mlp():
    d=2048
    hid_dim_1 = 2048
    hid_dim_2 = 1024
    d_out = 1000
    model = nn.Sequential(nn.Linear(d,hid_dim_1),
                     nn.Tanh(),
                     nn.Linear(hid_dim_1, hid_dim_2),
                     nn.Tanh(),
                     nn.Linear(hid_dim_2, d_out)
                     )
    return model
class Solver(object):
    def __init__(self, config):
        self.model = None
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.device = None
        self.cuda = config.cuda
        self.train_loader = None
        self.test_loader = None

    def load_data(self):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.train_batch_size, shuffle=True)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.test_batch_size, shuffle=False)

    def load_model(self):
        if self.cuda:
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        #self.model = LeNet().to(self.device)
        self.model = AlexNet().to(self.device)
        # self.model = VGG11().to(self.device)
        # self.model = VGG13().to(self.device)
#         self.model = VGG16().to(self.device)
#         self.model = mlp().to(self.device)
#         self.model = VGG16_opt().to(self.device)
        # self.model = VGG19().to(self.device)
        # self.model = GoogLeNet().to(self.device)
        #self.model = resnet18().to(self.device)
        # self.model = resnet34().to(self.device)
        #self.model = ResNet18().to(self.device)
#         self.model = ResNet50().to(self.device)
#         self.model = ResNet50_opt().to(self.device)
        # self.model = ResNet101().to(self.device)
        # self.model = ResNet152().to(self.device)
#         self.model = inception_v4().to(self.device)
        # self.model = DenseNet121().to(self.device)
        # self.model = DenseNet161().to(self.device)
        # self.model = DenseNet169().to(self.device)
        # self.model = DenseNet201().to(self.device)
        #self.model = WideResNet(depth=28, num_classes=10).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[75, 150], gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train(self):
        print("train:")
        train_loss = 0
        train_correct = 0
        total = 0

        # for batch_num, (data, target) in enumerate(self.train_loader):
        #while True:
        bs=448
        since=time.time()
        print("batchsize:",bs)
        for i in range(1000000000000000):
#             data = torch.rand([bs,3,224,224])
            data=torch.rand([bs,2048])
#             data = torch.rand([bs,3,299,299])
#             data = torch.rand([64,3,299,299])
            target = torch.zeros([bs],dtype=torch.long)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            prediction = torch.max(output, 1)  # second param "1" represents the dimension to be reduced
            total += target.size(0)
        print(bs*100/(time.time()-since))
        return True
    def test(self):
        print("test:")
        self.model.eval()
        test_loss = 0
        test_correct = 0
        total = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                prediction = torch.max(output, 1)
                total += target.size(0)
                test_correct += np.sum(prediction[1].cpu().numpy() == target.cpu().numpy())

                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))

        return test_loss, test_correct / total

    def save(self):
        model_out_path = "model.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def run(self):
        N=32
        x = torch.randn(N,3,227,227)
        self.load_model()
        x = x.to(self.device)
        since=time.time()
        for i in range(1000000):
            _=self.model(x)
        print(N/(2*(time.time()-since)))
        print((time.time()-since))
        

if __name__ == '__main__':
    main()
