import os
import torch

def save_model_w_condition(model, model_dir, model_name, f1, auroc, target_f1, target_auroc, log=print):
    '''
    model: this is not the multigpu model
    '''
    if f1 > target_f1 and auroc > target_auroc:
        log('saving a checkpoint')
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(model.state_dict(), f=os.path.join(model_dir, (model_name + 'AUROC_{0:.4f}_F1_{1:.4f}.pth').format(auroc, f1)))
 