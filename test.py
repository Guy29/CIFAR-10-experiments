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
            nn.Conv2d( 3,  32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            
            nn.Conv2d(32,  64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        
        self.flatten = nn.Flatten()
        
        self.full_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        logits = self.full_layers(self.flatten(self.conv_layers(x)))
        return logits




def train_and_test(model, train_set, test_set, num_batches, loss_fn, optimizer, num_epochs, device):
    
    assert len(train_set) % num_batches == 0
    assert len(test_set ) % num_batches == 0
    
    train_batch_size = len(train_set) // num_batches
    test_batch_size  = len(test_set)  // num_batches

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader  = DataLoader(test_set , batch_size=test_batch_size , shuffle=True)
    
    train_losses = []
    test_losses  = []
    accuracies   = []
    
    for epoch in range(num_epochs):
    
        for train_batch, test_batch in zip(train_loader, test_loader):
            
            model.eval()
            X, y = test_batch
            X, y = X.to(device), y.to(device)
            
            with torch.no_grad():
                pred = model(X)
                loss = loss_fn(pred, y)
                test_losses.append(loss.item())
                num_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
                accuracies.append(num_correct / test_batch_size)
            
            model.train()
            X, y = train_batch
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            train_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f'Epoch #{epoch+1} done')
    
    num_datapoints = num_batches * num_epochs
    
    return train_losses, test_losses, accuracies, num_epochs, num_batches, num_datapoints



def exp_smooth(nums, alpha=0.05):
    smoothed = [nums[0]]
    for i in range(1,len(nums)):
        smoothed.append(nums[i]*alpha + smoothed[-1]*(1-alpha))
    return smoothed



def plot_metrics(train_losses, test_losses, accuracies, num_epochs, num_batches, num_datapoints):

    general_gap = [a-b for (a,b) in zip(test_losses, train_losses)]
    
    smoothed_train_losses = exp_smooth(train_losses)
    smoothed_test_losses  = exp_smooth(test_losses)
    smoothed_general_gap  = exp_smooth(general_gap)
    smoothed_accuracies   = exp_smooth(accuracies)
    
    epochs = [v/num_batches for v in range(1, num_datapoints+1)]
    
    fig, ax1 = plt.subplots()
    
    l1, = ax1.plot(epochs, train_losses, color='tab:blue'  , alpha=0.15)
    l2, = ax1.plot(epochs, test_losses , color='tab:green' , alpha=0.15)
    #l3, = ax1.plot(epochs, general_gap , color='tab:orange', alpha=0.15)
    
    l4, = ax1.plot(epochs, smoothed_train_losses, color='tab:blue'  , label='Training loss')
    l5, = ax1.plot(epochs, smoothed_test_losses , color='tab:green' , label='Testing loss')
    l6, = ax1.plot(epochs, smoothed_general_gap , color='tab:orange', label='Generalization gap')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_ylim((0,3))
    
    ax2 = ax1.twinx()
    l7, = ax2.plot(epochs, smoothed_accuracies  , color='tab:purple', label='Accuracy')
    
    ax2.set_ylabel('Accuracy score')
    ax2.set_ylim((0,1))
    
    plt.xlim((0,num_epochs))
    plt.grid(True)
    
    lines  = [l4, l5, l6, l7]
    labels = [line.get_label() for line in lines]
    plt.legend(lines, labels)
    
    plt.show()
    



if __name__ == "__main__":
    
    if   torch.cuda.is_available():         device='cuda'
    elif torch.backends.mps.is_available(): device='mps'
    else:                                   device='cpu'

    print(f"Device: {device}")

    model = NeuralNetwork().to(device)

    train_set = datasets.CIFAR10(root='data', train=True , download=True, transform=ToTensor())
    test_set  = datasets.CIFAR10(root='data', train=False, download=True, transform=ToTensor())
    
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    metrics = train_and_test(model, train_set, test_set, 50, loss_fn, optimizer, 50, device)
    
    plot_metrics(*metrics)



# conda create -n nn_env python=3.8
# conda activate nn_env
# pip3 install torch==2.3.0+cu121 torchvision==0.18+cu121 matplotlib -f https://download.pytorch.org/whl/cu121/torch_stable.html
# conda install ipykernel
# python -m ipykernel install --user --name=nn_env --display-name="Neural Networks"
# conda install jupyterlab
# jupyter lab