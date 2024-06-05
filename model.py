# contains the model class
# the model is directly used in LighteningCLI()
# All the training steps are defined in this file

import torch
import lightning as l

from model.module import PredDOA
from model.tcrnn import CRNN
from packaging.version import Version

import loss as lss

class TrustedRCNN(l.LightningModule):
    def __init__(
        self,
        input_dim: int = 4,
        num_classes: int = 180,
        hidden_dim: int = 64,
        lr=0.0005,
        tar_useVAD: bool = True,
        ch_mode: str = 'MM',
        fs: int = 16000,
        method_mode: str = 'IDL',
        source_num_mode: str = 'KNum',
        max_num_sources: int = 1,
        return_metric: bool = True,
        compile: bool = False,
        device: str = "cuda",
        lamdba_peochs: int = 10,
    ):
        super().__init__()
        # Model init
        self.model = CRNN(
            max_num_sources=max_num_sources,
            input_dim=input_dim,
            cnn_dim=hidden_dim,
            num_classes=num_classes,
        )

        torch.set_float32_matmul_precision('medium')

        if compile:
            print("Compiling the model!")
            assert Version(torch.__version__) >= Version(
                '2.0.0'), torch.__version__
            self.model = torch.compile(self.model)

        # save all the parameters to self.hparams
        self.save_hyperparameters(ignore=['model'])
        self.tar_useVAD = tar_useVAD
        self.method_mode = method_mode
        self.dev = device
        self.source_num_mode = source_num_mode
        self.max_num_sources = max_num_sources
        self.ch_mode = ch_mode
        self.lamdba_epochs = lamdba_peochs

        self.fre_max = fs / 2
        self.return_metric = return_metric
        self.get_metric = PredDOA()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams.lr)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8988, last_epoch=-1)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=0)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                # 'monitor': 'valid/loss',
            }
        }

    """ 
            Function: training_step
            Args:
                    batch: input data
                    batch_idx: batch index
            Returns:
                    loss
    """
    def training_step(self, batch, batch_idx: int):
        # the batch here is returned by the __getitem__ in the dataloader (dataset.py)
        # batch[0] is the input, batch[1] is the target
        mic_sig_batch = batch[0]  # [2, 4, 256, 299] bs, c, f, t
        # ground truth
        gt_batch = batch[1]

        # predict the output by input
        pred_batch = self.model(mic_sig_batch)

        # calculate the loss using the predicted output and the target
        loss, evidence, U = self.ce_loss_uncertainty(
            pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)

        # log the loss to the tensorboard
        self.log("train/loss", loss, prog_bar=True,
                 on_epoch=True, sync_dist=True)

        # return a dictionary containing the loss
        # other data that
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        gt_batch = batch[1]

        pred_batch = self(mic_sig_batch)
        loss, evidence, U = self.ce_loss_uncertainty(
            pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)

        # loss = lss.ce_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("valid/loss", loss, sync_dist=True, on_epoch=True)

        metric = self.get_metric(pred_batch=pred_batch, gt_batch=gt_batch)
        for m in metric:
            self.log('valid/'+m, metric[m].item(),
                     sync_dist=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx: int):
        mic_sig_batch = batch[0]
        gt_batch = batch[1]

        pred_batch = self(mic_sig_batch)  # [2, 24, 512]
        loss, evidence, U = lss.ce_loss_uncertainty(
            pred_batch=pred_batch, gt_batch=gt_batch, current_epoch=self.current_epoch)
        # loss = self.ce_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        self.log("test/loss", loss, sync_dist=True)
        metric = self.get_metric(pred_batch=pred_batch, gt_batch=gt_batch)
        # print(metric)
        for m in metric:
            self.log('test/'+m, metric[m].item(), sync_dist=True)

    def predict_step(self, batch, batch_idx: int):

        mic_sig_batch = batch[0]
        pred_batch = self.forward(mic_sig_batch)

        return pred_batch