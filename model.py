
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
        output = F.relu(x)
        # output = F.log_softmax(x, dim=1)
        return output
    
# class LinearModel(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(28 * 28, hidden_dim),
#             nn.ReLU(True),
#             nn.Linear(hidden_dim, 10)
#             )

#     def forward(self, x):
#         """
#         x: (B, 1, 28, 28) batch of images
#         """
#         x = nn.Flatten(x)
#         # x = rearrange(x, 'b 1 x y -> b (x y)', x=28, y=28)
#         return self.net(x)
    
    
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
        return out

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
        print(f'Val Preds: {len(preds)}\n')
        print(f'Val Targs: {len(targs)}\n')
        
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
        print(f'Train Outputs for rank {self.global_rank}: {len(outputs)}\n')
        print(f'Train Preds: {len(preds)}\n')
        print(f'Train Targs: {len(targs)}\n')
        all_out = self.all_gather(outputs)
        torch.save(all_out, f"all_out.pt")
        # print(f'All Out shape: {all_out.shape[0]} * {all_out.shape[1]}\n')
        all_train_loss = torch.cat([x['loss'].flatten().cpu() for x in all_out]).tolist()
        all_train_preds = torch.cat([x['preds'].flatten().cpu() for x in all_out])
        all_train_targ = torch.cat([x['targ'].flatten().cpu() for x in all_out])
        print(f'Len of All Train Loss: {len(all_train_loss)}\n')
        print(f'Len of All Train Preds: {len(all_train_preds)}\n')
        print(f'Len of All Train Targ: {len(all_train_targ)}\n')
        acc = (torch.sum(torch.eq(all_train_preds,all_train_targ)) / len(all_train_preds)).item()*100
        print(f'Accuracy: {acc}')

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        
        return {"optimizer": optimizer}