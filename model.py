
import torch

import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn

from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class LitResnet(pl.LightningModule):
    def __init__(self, lr, opt, num_classes):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = Net()
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.conf_mat = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        train_loss = self.loss(logits, y)
        train_acc = self.train_acc(preds, y)
        # self.log('train_acc_step', train_acc)
        # self.log('train_loss_step', train_loss)
        return {"loss": train_loss, "preds": preds, "targ": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        val_loss = self.loss(logits, y)
        val_acc = self.val_acc(preds, y)
        # self.log('val_acc_step', val_acc)
        # self.log('val_loss_step', val_loss)
        return {"loss": val_loss, "preds": preds, "targ": y}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        test_loss = self.loss(logits, y)
        test_acc = self.test_acc(preds, y)
        # self.log('test_acc_step', test_acc)
        # self.log('test_loss_step', test_loss)
        return {"loss": test_loss, "preds": preds, "targ": y}
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        # self.log('valid_loss_epoch', avg_val_loss)
        # self.log('valid_acc_epoch', self.val_acc.compute())
        # Method 2
        self.logger.experiment.add_scalar("valid_loss_epoch",
                                            avg_val_loss,
                                            self.current_epoch)
        self.logger.experiment.add_scalar("valid_acc_epoch",
                                            self.val_acc.compute(),
                                            self.current_epoch)
        self.val_acc.reset()
        preds = torch.cat([x['preds'] for x in outputs])
        targs = torch.cat([x['targ'] for x in outputs]) 
        print(f'Val Preds: {len(preds)}')
        print(f'Val Targs: {targs(preds)}')
        
    def test_epoch_end(self, outputs):
        avg_test_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        # self.log('test_loss_epoch', avg_test_loss)
        # self.log('test_acc_epoch', self.test_acc.compute())
        self.logger.experiment.add_scalar("test_loss_epoch",
                                            avg_test_loss,
                                            self.current_epoch)
        self.logger.experiment.add_scalar("test_acc_epoch",
                                            self.test_acc.compute(),
                                            self.current_epoch)
        self.test_acc.reset()
        preds = torch.cat([x['preds'] for x in outputs])
        targs = torch.cat([x['targ'] for x in outputs])        
        # confmat = self.conf_mat(preds, targs)
        # torch.save(confmat, f"test-confmat.pt")
        
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.hstack([x['loss'] for x in outputs]).mean()        
        # self.log('train_loss_epoch', avg_train_loss)
        # self.log('train_acc_epoch', self.train_acc.compute())
        self.logger.experiment.add_scalar("train_loss_epoch",
                                            avg_train_loss,
                                            self.current_epoch)
        self.logger.experiment.add_scalar("train_acc_epoch",
                                            self.train_acc.compute(),
                                            self.current_epoch)
        self.train_acc.reset()
        preds = torch.cat([x['preds'] for x in outputs])
        targs = torch.cat([x['targ'] for x in outputs]) 
        print(f'Train Preds: {len(preds)}')
        print(f'Train Targs: {targs(preds)}')

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        
        return {"optimizer": optimizer}