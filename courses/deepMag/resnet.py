import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
import copy
from torch.utils.data import DataLoader
from torchModelDataLoader import modelDataset
import magneticsForward as MF
from tqdm.auto import tqdm

batch_size = 10
dataset = modelDataset(directory='./models', dims=3, scale=1)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


device = torch.device("cuda")


def make_three_gaussians(size, noise=1.0):

    r = 3.0

    k = size//3

    X = np.zeros([0,2])

    theta = np.linspace(0, 2*np.pi, 4)

    for i in range(3):

        xi = r*np.cos(theta[i])

        yi = r*np.sin(theta[i])

        Xi = np.random.randn(k, 2)*noise/3
        Xi[:,0] = Xi[:,0] + xi
        Xi[:,1] = Xi[:,1] + yi

        X = np.concatenate((X, Xi), axis=0)

    return X



class simpleResNet(nn.Module):



    def __init__(self, nin, nhid=128, nlayers=5):

        super().__init__()


        self.Kopen = nn.Parameter(1e-1*torch.randn(nhid, nin))

        self.K1 = nn.Parameter(1e-2*torch.randn(nlayers,nhid, nhid))

        self.K2 = nn.Parameter(1e-2*torch.randn(nlayers,nhid, nhid))

        self.KR = nn.Parameter(1e-2*torch.randn(nlayers,nhid, 2))

        #BN = nn.ParameterList()

        #for i in range(nlayers):

        # BNi = nn.BatchNorm1d(nhid)

        # BN.append(BNi)

        #self.BN = BN


    def forward(self, z, r):


        z = self.Kopen@z

        z = F.silu(z)

        for i in range(self.K1.shape[0]):

            dz = self.K1[i]@(z + self.KR[i]@r)

            #dz = self.BN[i](dz.t()).t()

            dz = F.silu(dz)

            dz = self.K2[i]@dz

            z = z + dz


        return z 



class inverseNet(nn.Module):



    def __init__(self, net, forMat, nhid):

        super().__init__()



        self.forMat = forMat

        self.net = net

        self.nlayers = 10



        # dictionary

        self.D = nn.Parameter(1e0*torch.rand(2, nhid))



    def forward(self, data):

        # x = torch.zeros(2, data.shape[1], device=data.device)

        z = torch.zeros(self.D.shape[1], data.shape[1], device=data.device)

        D = self.D.to(device)

        for i in range(self.nlayers):


            # data fitting step

            x = D @ z

            r = data - self.forMat(x)


            g = self.forMat.adjoint(r)

            g = D.T @ g

            Adg = self.forMat(D @ g) 

            with torch.no_grad():

                alpha = (r*Adg).mean()/(Adg*Adg).mean()
            
            z = z + alpha*g

            # correction step

            z = self.net(z, r)

        x = D@z

        return x



def loss_fn(x, forMat, inverseNet):

    data = forMat(x)

    xrec = inverseNet(data)

    loss = F.mse_loss(xrec, x)/F.mse_loss(x*0, x)

    return loss


shape = torch.tensor([128, 128, 64])
h = torch.tensor([100.0, 100.0, 100.0])
dirs = torch.tensor([np.pi/2, np.pi/2, np.pi/2, np.pi/2])
forMat = MF.magnetics(dim=shape, h=h, dirs=dirs, device=device)

nhid = 256

net = simpleResNet(nin=nhid, nhid=nhid, nlayers=6)
net = net.to(device)

inet = inverseNet(net, forMat, nhid)
inet = inet.to(device)

optimizer = Adam(inet.parameters(), lr=1e-4)
# Train the network if needed

niterations = 10000

# x = make_three_gaussians(10**4)
# x = torch.tensor(x, dtype=torch.float32).t()
# x = x.to(device)

hh = torch.zeros(niterations)
bestloss = 1e11


for i in range(niterations):

    for j, x in tqdm(enumerate(loader, 0)): 

        optimizer.zero_grad()
        loss = loss_fn(x, forMat, inet) 
        loss.backward()

        torch.nn.utils.clip_grad_norm_(parameters=inet.parameters(), max_norm=0.1, norm_type=2.0)
        optimizer.step()

        hh[i] = loss.item()
        # if i%500 == 0:
        print('iter = %3d loss = %3e best_loss = %3e'%(i, loss, bestloss))

        if loss<bestloss:

            bestloss = loss

            bestnet = copy.deepcopy(net)

            print('iter = %3d best loss = %3e'%(i, bestloss))

            torch.save(bestnet.state_dict(),'net_dictionary.pt')



# Inference without data

model = bestnet

# model.load_state_dict(torch.load('net3G.pt'))

# model.to(device)

# model.eval()



sample_size = 5000
x = make_three_gaussians(sample_size)
x = torch.tensor(x, dtype=torch.float32).t()
x = x.to(device)
