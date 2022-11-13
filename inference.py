import torch
import numpy as np
import tqdm
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from sklearn.metrics import confusion_matrix,f1_score

@torch.no_grad()
def inference(Net,test_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictions = []
    gt = []
    val_loss = 0
    Net.eval()
    for itter, batch in tqdm(enumerate(test_loader)):
        img = batch['image'].to(device, dtype=torch.float)
        msk = batch['mask']
        msk_pred = Net(img)

        gt.append(msk.numpy()[0, 0])
        msk_pred = msk_pred.cpu().detach().numpy()[0, 0]
        msk_pred  = np.where(msk_pred>=0.43, 1, 0)
        msk_pred = binary_opening(msk_pred, structure=np.ones((6,6))).astype(msk_pred.dtype)
        msk_pred = binary_fill_holes(msk_pred, structure=np.ones((6,6))).astype(msk_pred.dtype)
        predictions.append(msk_pred)              
    predictions = np.array(predictions)
    gt = np.array(gt)
    y_scores = predictions.reshape(-1)
    y_true   = gt.reshape(-1)
    y_scores2 = np.where(y_scores>0.47, 1, 0)
    y_true2   = np.where(y_true>0.5, 1, 0)
    #F1 score
    F1_score = f1_score(y_true2, y_scores2, labels=None, average='binary', sample_weight=None)
    print ("\nF1 score (F-measure) or DSC: " +str(F1_score))
    confusion = confusion_matrix(np.int32(y_true), y_scores2)
    print (confusion)
    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))
    print ("Accuracy: " +str(accuracy))
    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])
    print ("Specificity: " +str(specificity))
    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])
    print ("Sensitivity: " +str(sensitivity))


