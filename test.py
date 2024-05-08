import torch
from   torch                  import nn
from   torch.utils.data       import DataLoader
from   torchvision            import datasets
from   torchvision.transforms import ToTensor

import matplotlib.pyplot      as plt
import time


class NeuralNetwork(nn.Module):

    def __init__(self):

        super(NeuralNetwork, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.flatten = nn.Flatten()
        
        self.full_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        logits = self.full_layers(self.flatten(self.conv_layers(x)))
        return logits




def train(dataloader, model, loss_fn, optimizer):

    model.train()
    running_loss = 0.
    losses = []
    
    for batch_num, (X, y) in enumerate(dataloader, 1):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
    
    return losses


def test(dataloader, model, loss_fn):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (test_loss, correct)


if __name__ == "__main__":

    train_set = datasets.CIFAR10(root='data', train=True , download=True, transform=ToTensor())
    test_set  = datasets.CIFAR10(root='data', train=False, download=True, transform=ToTensor())
    
    t0 = time.time()

    train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
    test_loader  = DataLoader(test_set , batch_size=1000, shuffle=True)
    
    print('Loader creation time:',time.time() - t0)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = NeuralNetwork().to(device)
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    losses = []
    test_losses = []
    t0 = time.time()
    for _ in range(20):
        losses += train(train_loader, model, loss_fn, optimizer)
        test_losses.append(test(test_loader, model, loss_fn)[0])
        print(time.time()-t0)
        t0 = time.time()
    
    plt.plot(range(len(losses)), losses)
    plt.plot(range(50,50*(len(test_losses)+1),50), test_losses)
    plt.grid(True)
    plt.show()



# conda create -n nn_env python=3.8
# conda activate nn_env
# pip3 install torch==2.3.0+cu121 torchvision==0.18+cu121 matplotlib -f https://download.pytorch.org/whl/cu121/torch_stable.html
# conda install ipykernel
# python -m ipykernel install --user --name=nn_env --display-name="Neural Networks"
# conda install jupyterlab
# jupyter lab