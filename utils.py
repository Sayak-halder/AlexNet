import torch
import os

def accuracy(output,target,topk=(1,)):
    with torch.no_grad():
        maxk=max(topk)
        batch_size=target.size(0)

        _,pred=output.topk(maxk,1,True,True)
        pred=pred.t()
        correct=pred.eq(target.view(1,-1).expand_as(pred))

        res=[]
        for k in topk:
            correct_k=correct[:k].reshape(-1).float().sum(0,keepdim=True)
            res.append((correct_k/batch_size).item())
        return res
    
def save_checkpoint(state,is_best,checkpoint_dir='checkpoint',filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir,exist_ok=True)
    path=os.path.join(checkpoint_dir,filename)
    torch.save(state,path)
    if is_best:
        best_path=os.path.join(checkpoint_dir,'best.pth')
        torch.save(state,best_path)
