import torch
from data import *
from model.model import *
from utils import *

def test(model_path,batch_size,device):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_worker=4
    _,test_loader=get_cifar10_loader(batch_size=batch_size,num_worker=num_worker)

    model=AlexNet(num_classes=10).to(device)

    checkpoint=torch.load(model_path,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    total_acc=0.0

    with torch.no_grad():
        for inputs,targets in test_loader:
            inputs,targets=inputs.to(device),targets.to(device)

            outputs=model(inputs)
            acc1,=accuracy(outputs,targets,topk=(1,))

            total_acc+=acc1*inputs.size(0)

        final_acc=total_acc/len(test_loader.dataset)

        print(f'Test Acc: {final_acc*100:.2f}%')

if __name__=='__main__':
    test(model_path='checkpoint/best.pth',batch_size=128,device='cuda')