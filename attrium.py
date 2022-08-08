import pytorch_lightning as pl
from model import UNet
import torch
import nibabel as nib

import tqdm
import numpy as np
import  matplotlib.pyplot  as plt
import cv2 

def normalize(full_volume):
    mu = full_volume.mean()
    std = np.std(full_volume)
    normalized = (full_volume - mu) / std
    return normalized

def standardize(normalized):
    standardized = (normalized - normalized.min()) / (normalized.max() - normalized.min())
    return standardized

class DiceLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, mask):
        pred = torch.flatten(pred)
        mask = torch.flatten(mask)

        counter = (pred * mask).sum()
        denum   = pred.sum() + mask.sum() + 1e-8
        dice    = (2 * counter) / denum

        return 1 - dice

class AtriumSegmentation(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.model = UNet()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.loss_fn   = DiceLoss()


    def forward(self, data):
        return torch.sigmoid(self.model(data))


    def training_step(self, batch, batch_idx):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)

        loss = self.loss_fn(pred, mask)

        self.log('Train Dice', loss)

        if batch_idx % 50 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), 'Train')
        
        return loss


    def validation_step(self, batch, batch_idx):
        mri, mask = batch
        mask = mask.float()
        pred = self(mri)

        loss = self.loss_fn(pred, mask)

        self.log('Val Dice', loss)

        if batch_idx % 2 == 0:
            self.log_images(mri.cpu(), pred.cpu(), mask.cpu(), 'Val')
        
        return loss


    def log_images(self, mri, pred, mask, name):
    
        pred = pred > 0.5

        fig, axis = plt.subplots(1, 2)

        axis[0].imshow(mri[0][0], cmap='bone')
        mask_ = np.ma.masked_where(mask[0][0] == 0, mask[0][0])
        axis[0].imshow(mask_, alpha=0.6)

        axis[1].imshow(mri[0][0], cmap='bone')
        mask_ = np.ma.masked_where(mask[0][0] == 0, pred[0][0])
        axis[1].imshow(mask_, alpha=0.6)

        self.logger.experiment.add_figure(name, fig, self.global_step)


    def configure_optimizers(self):
        return [self.optimizer]


model = AtriumSegmentation.load_from_checkpoint("epoch=1-step=484.ckpt")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval();
model.to(device);

preds  = []
labels = []

# for slice, label in tqdm(val_dataset):
#     slice = torch.tensor(slice).to(device).unsqueeze(0)
#     with torch.no_grad():
#         pred = model(slice)
#         preds.append(pred.cpu().numpy())
#         labels.append(label)

# preds  = np.array(preds)
# labels = np.array(labels)

# model.loss_fn(torch.from_numpy(preds), torch.from_numpy(labels))
def extract(subject, st, placeholder_image):
    subject_mri = nib.load(subject).get_fdata()
    subject_mri = subject_mri[32:-32, 32:-32]
    standardized_scan = standardize(normalize(subject_mri))

    preds = []
    fig = plt.figure(figsize = (10, 5))
    for i in range(standardized_scan.shape[-1]):
        slice = standardized_scan[:, :, i]
        with torch.no_grad():
            pred = model(torch.tensor(slice).unsqueeze(0).unsqueeze(0).float().to(device))[0][0]
            pred = pred > 0.5
        preds.append(pred.cpu())


    for i in range(standardized_scan.shape[-1]):
        

        plt.imshow(standardized_scan[:, :, i], cmap='bone')
        mask = np.ma.masked_where(preds[i] == 0, preds[i])
        plt.imshow(mask, alpha=0.5)
        # plt.savefig('x',dpi=400)
        placeholder_image.pyplot(fig)
        # st.pyplot(fig)
