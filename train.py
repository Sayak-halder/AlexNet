import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data import *
from model.model import *
from utils import *

def train(epochs,batch_size,learning_rate,momentum,weight_decay,device):
    
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_worker=4
    train_loader,test_loader=get_cifar10_loader(batch_size=batch_size,num_worker=num_worker)

    model=AlexNet(num_classes=10).to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),
                        lr=learning_rate,
                        momentum=momentum,
                        weight_decay=weight_decay
                        )
    scheduler=optim.lr_scheduler.StepLR(optimizer,
                                        step_size=10,
                                        gamma=0.1
                                        )

    best_acc=0.0
    for epoch in range(epochs):
        model.train()
        train_loss=0.0
        for inputs,targets in tqdm(train_loader,desc=f"Epoch {epoch}/{epochs}"):
            inputs,targets=inputs.to(device),targets.to(device)

            outputs=model(inputs)
            loss=criterion(outputs,targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()*inputs.size(0)

        scheduler.step()

        model.eval()
        test_loss=0.0
        top1_acc=0.0
        
        with torch.no_grad():
            for inputs,targets in test_loader:
                inputs,targets=inputs.to(device),targets.to(device)
                outputs=model(inputs)
                loss=criterion(outputs,targets)

                test_loss+=loss.item()*inputs.size(0)
                acc1,_=accuracy(outputs,targets,topk=(1,))
                top1_acc+=acc1*inputs.size(0)
            
            n_test=len(test_loader.dataset)
            epoch_loss=test_loss/len(train_loader.dataset)
            epoch_test_loss=test_loss/n_test
            epoch_acc=top1_acc/n_test

            print(f"Epoch {epoch}:\n Train Loss: {epoch_loss:.4f} Test Loss: {epoch_test_loss:.4f} Test Acc: {epoch_acc*100:.2f}%\n")

            is_best=epoch_acc>best_acc
            best_acc=max(epoch_acc,best_acc)
            save_checkpoint({
                'epoch':epoch,
                'model_state':model.state_dict(),
                'best_acc':best_acc,
                'optimizer':optimizer.state_dict()
            },is_best)

if __name__=='__main__':
    train(epochs=20,
          batch_size=128,
          learning_rate=0.001,
          momentum=0.9,
          weight_decay=5e-4,
          device='cuda'
          )