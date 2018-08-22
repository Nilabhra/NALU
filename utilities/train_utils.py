import numpy as np
import torch
from tqdm import tqdm


def get_batches(data, target, batch_size,
                mode='test', use_gpu=False):
    '''
    Generator function to yield minibatches of data and targets
    '''
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
    '''
    Function to calculate loss on the validation and test set
    detached from the compute graph
    '''
    preds, targets = get_eval_preds(model, data, targets,
                                    batch_size, use_gpu)
    loss = criterion(preds, targets)
    return loss.item()


def get_eval_preds(model, data, targets,
                   batch_size, use_gpu=False):
    '''
    Function to perfrom inference on validation and test set
    detached from the compute graph
    '''
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


def train(model, criterion, opt, train_data, train_targets,
          valid_data, valid_targets, patience=15, batch_size=32,
          num_epochs=10000, checkpoint='best_model.sav'):
    '''
    Function to train a model given a criterion, optimizer,
    training and validation data. Early stopping using the validation
    loss is used to terminat the training routine. The returned model
    is loaded with the best set of weights found during training.
    '''
    running_patience = patience
    running_batch = 0
    running_loss = 0
    min_loss = float('inf')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()
    optimizer = opt(model.parameters())
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total = int(np.ceil(train_data.shape[0]/batch_size))
        with tqdm(total=total, desc="Epoch: {}".format(epoch),
                  leave=False) as prog_bar:

            for x, y in get_batches(train_data, train_targets, batch_size,
                                    mode='train', use_gpu=use_gpu):
                output = model(x)
                loss = criterion(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                running_batch += 1
                prog_bar.set_description_str('[Epoch: {}] Training loss after {} batches: {:.3f}'.format(
                            epoch, running_batch, running_loss/running_batch))
                prog_bar.update(1)
                
        valid_loss = get_eval_loss(model, criterion, valid_data,
                                   valid_targets, batch_size, False)
        if valid_loss < min_loss:
            min_loss = valid_loss
            with open(checkpoint, 'wb') as f:
                torch.save(model.state_dict(), f)
                running_patience = patience
        else:
            running_patience -= 1
        if running_patience == 0:
            # Ran out of patience, early stopping employed!'
            break
    model.load_state_dict(torch.load(checkpoint))
    return model