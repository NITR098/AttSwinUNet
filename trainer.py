import torch
import numpy as np
import copy 
from sklearn.metrics import f1_score
from tqdm import tqdm


def trainer(config,Net,train_loader,test_loader,optimizer,criteria,args):
    best_F1_score = 0.0
    for ep in range(int(config['epochs'])):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Net.train()
        epoch_loss = 0
        for itter, batch in enumerate(train_loader):
            img = batch['image'].to(device, dtype=torch.float)
            msk = batch['mask'].to(device)
            mask_type = torch.float32
            msk = msk.to(device=device, dtype=mask_type)
            msk_pred = Net(img)
            loss          = criteria(msk_pred, msk) 
            optimizer.zero_grad()
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()  
        if itter%int(float(config['progress_p']) * len(train_loader))==0:
            print(f' Epoch>> {ep+1} and itteration {itter+1} Loss>> {((epoch_loss/(itter+1)))}')

    predictions = []
    gt = []

    if (ep+1)%args.eval_interval==0:
        with torch.no_grad():
            print('val_mode')
            val_loss = 0
            Net.eval()
            for itter, batch in tqdm(enumerate(test_loader)):
                img = batch['image'].to(device, dtype=torch.float)
                msk = batch['mask']
                msk_pred = Net(img)

                gt.append(msk.numpy()[0, 0])
                msk_pred = msk_pred.cpu().detach().numpy()[0, 0]
                msk_pred  = np.where(msk_pred>=0.4, 1, 0)
                predictions.append(msk_pred)        

        predictions = np.array(predictions)
        gt = np.array(gt)

        y_scores = predictions.reshape(-1)
        y_true   = gt.reshape(-1)

        y_scores2 = np.where(y_scores>0.5, 1, 0)
        y_true2   = np.where(y_true>0.5, 1, 0)

        #F1 score
        F1_score = f1_score(y_true2, y_scores2, labels=None, average='binary', sample_weight=None)
        print ("\nF1 score (F-measure) or DSC: " +str(F1_score))    
        if (F1_score) > best_F1_score:
            print('New best loss, saving...')
            best_F1_score = copy.deepcopy(F1_score)
            state = copy.deepcopy({'model_weights': Net.state_dict(), 'test_F1_score': F1_score})
            torch.save(state, args.saved_model)