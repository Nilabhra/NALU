import numpy as np
import torch


def get_batches(data, target, batch_size,
                mode='test', use_gpu=False):
    idx = np.arange(0, data.shape[0])
    
    if mode == 'train':
        np.random.shuffle(idx)

    while idx.shape[0] > 0:
        batch_idx = idx[:batch_size]
        idx = idx[batch_size:]
        batch_data = data[batch_idx]
        batch_target = target[batch_idx]
        
        batch_data = torch.from_numpy(batch_data)
        batch_data = batch_data.float()
        batch_target = torch.from_numpy(batch_target)
        batch_target = batch_target.float().view(-1, 1)
        
        if use_gpu:
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()
        
        yield batch_data, batch_target


def get_eval_loss(model, criterion, data,
                  targets, batch_size, use_gpu=False):
    preds, targets = get_eval_preds(model, data, targets,
                                    batch_size, use_gpu)
    loss = criterion(preds, targets)
    return loss.item()


def get_eval_preds(model, data, targets,
                   batch_size, use_gpu=False):
    with torch.no_grad():
        model.eval()
        model_preds = []
        tensor_targets = []
        for x, y in get_batches(data, targets, batch_size,
                                mode='test', use_gpu=use_gpu):
            model_preds.append(model(x))
            tensor_targets.append(y)
        model_preds = torch.cat(model_preds, dim=0)
        tensor_targets = torch.cat(tensor_targets, dim=0)
    return model_preds, tensor_targets