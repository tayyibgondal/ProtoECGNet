import os
import torch

def save_model_w_condition(model, model_dir, model_name, auroc, target_auroc, log=print):
    '''
    model: this is not the multigpu model
    '''
    if auroc > target_auroc:
        log('saving a checkpoint')
        torch.save(model.state_dict(), f=os.path.join(model_dir, (model_name + 'AUROC_{0:.4f}.pth').format(auroc)))
 