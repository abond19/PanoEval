import torch
from torch import nn
from torch import Tensor
from lightning.pytorch.utilities import rank_zero_only
from torchmetrics import Metric
import math
from torchmetrics.image.fid import _compute_fid
import os
from einops import rearrange
import lightning as L
import tempfile
from PIL import Image
import wandb

###################### Panfusion/models/faed/modules.py ######################
class CircularPadding(nn.Module):

    def __init__(self, pad):
        super(CircularPadding, self).__init__()
        self.pad = pad

    def forward(self, x):
        if self.pad == 0:
            return x
        x = torch.nn.functional.pad(x,
                                    (self.pad, self.pad, self.pad, self.pad),
                                    'constant', 0)
        x[:, :, :, 0:self.pad] = x[:, :, :, -2 * self.pad:-self.pad]
        x[:, :, :, -self.pad:] = x[:, :, :, self.pad:2 * self.pad]
        return x


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super(Conv2d, self).__init__()
        self.pad = CircularPadding(padding)
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding=0)

    def forward(self, x):
        x = self.conv2d(self.pad(x))
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResBlock, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride=1,
                            padding=padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels,
                            out_channels,
                            kernel_size,
                            stride=1,
                            padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        out = self.relu(self.batchnorm1(self.conv1(x)))
        out = self.batchnorm2(self.conv2(out))
        out += x

        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super(ConvBlock, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride,
                            padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.relu(self.batchnorm1(self.conv1(x)))

        return x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.upsampling = nn.functional.interpolate

        self.upconv2_rgb = ConvBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.upconv3_rgb = ConvBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.upconv4_rgb = ConvBlock(in_channels=128,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.upconv5_rgb = ConvBlock(in_channels=64,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.outconv_rgb = Conv2d(in_channels=32,
                                  out_channels=3,
                                  kernel_size=9,
                                  stride=1,
                                  padding=4)

        self.upres2_rgb = ResBlock(in_channels=128,
                                   out_channels=128,
                                   kernel_size=3,
                                   padding=1)
        self.upres3_rgb = ResBlock(in_channels=128,
                                   out_channels=128,
                                   kernel_size=5,
                                   padding=2)
        self.upres4_rgb = ResBlock(in_channels=64,
                                   out_channels=64,
                                   kernel_size=7,
                                   padding=3)
        self.upres5_rgb = ResBlock(in_channels=32,
                                   out_channels=32,
                                   kernel_size=9,
                                   padding=4)

    def forward(self, x):

        x = self.upsampling(x,
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)
        rgb = x[:, :128]

        rgb = self.upconv2_rgb(rgb)
        rgb = self.upres2_rgb(rgb)
        rgb = self.upsampling(rgb,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        rgb = self.upconv3_rgb(rgb)
        rgb = self.upres3_rgb(rgb)
        rgb = self.upsampling(rgb,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        rgb = self.upconv4_rgb(rgb)
        rgb = self.upres4_rgb(rgb)
        rgb = self.upsampling(rgb,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        rgb = self.upconv5_rgb(rgb)
        rgb = self.upres5_rgb(rgb)
        rgb = self.upsampling(rgb,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        rgb = self.outconv_rgb(rgb)
        rgb = torch.tanh(rgb)

        return rgb


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.downconv1_rgb = Conv2d(in_channels=3,
                                    out_channels=32,
                                    kernel_size=9,
                                    stride=1,
                                    padding=4)

        self.downconv2_rgb = ConvBlock(in_channels=32,
                                       out_channels=64,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downconv3_rgb = ConvBlock(in_channels=64,
                                       out_channels=128,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downconv4_rgb = ConvBlock(in_channels=128,
                                       out_channels=128,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downconv5_rgb = ConvBlock(in_channels=128,
                                       out_channels=128,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downconv6_rgb = ConvBlock(in_channels=128,
                                       out_channels=128,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downres1_rgb = ResBlock(in_channels=32,
                                     out_channels=32,
                                     kernel_size=9,
                                     padding=4)

        self.downres2_rgb = ResBlock(in_channels=64,
                                     out_channels=64,
                                     kernel_size=7,
                                     padding=3)

        self.downres3_rgb = ResBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=5,
                                     padding=2)

        self.downres4_rgb = ResBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     padding=1)

        self.downres5_rgb = ResBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     padding=1)

        self.fuse = Conv2d(in_channels=128,
                           out_channels=128,
                           kernel_size=3,
                           stride=1,
                           padding=1)

    def forward(self, x):
        rgb = x[:, :3]
        rgb = self.downconv1_rgb(rgb)
        rgb = self.downres1_rgb(rgb)
        rgb = self.downconv2_rgb(rgb)
        rgb = self.downres2_rgb(rgb)
        rgb = self.downconv3_rgb(rgb)
        rgb = self.downres3_rgb(rgb)
        rgb = self.downconv4_rgb(rgb)
        rgb = self.downres4_rgb(rgb)
        rgb = self.downconv5_rgb(rgb)
        rgb = self.downres5_rgb(rgb)
        rgb = self.downconv6_rgb(rgb)

        x = self.fuse(rgb)

        return x
    

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
###############################################################################


###################### Panfusion/models/modules/utils.py ######################
def tensor_to_image(image):
    if image.dtype != torch.uint8:
        image = (image / 2 + 0.5).clamp(0, 1)
        image = (image * 255).round()
    image = image.cpu().numpy().astype('uint8')
    image = rearrange(image, '... c h w -> ... h w c')
    return image

###################### Panfusion/models/faed/FAED.py ##########################
class WandbLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.temp_dir = tempfile.TemporaryDirectory()

    def temp_wandb_image(self, image, prompt=None):
        if isinstance(image, torch.Tensor):
            image = tensor_to_image(image)
        img_path = tempfile.NamedTemporaryFile(
            dir=self.temp_dir.name, suffix=".jpg", delete=False).name
        Image.fromarray(image.squeeze()).save(img_path)
        return wandb.Image(img_path, caption=prompt if prompt else None)

    def __del__(self):
        self.temp_dir.cleanup()


class FAED(WandbLightningModule):
    def __init__(
            self,
            lr: float = 1e-4,
            lr_decay: float = 0.99,
            ):
        super().__init__()
        self.save_hyperparameters()
        self.net = AutoEncoder()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_decay)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        pano_pred = self.net(batch['pano'].squeeze(1)).unsqueeze(1)
        loss = torch.nn.functional.l1_loss(pano_pred, batch['pano'])
        self.log('train/loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pano_pred = self.net(batch['pano'].squeeze(1)).unsqueeze(1)
        self.log_val_image(pano_pred, batch['pano'], batch['pano_id'])

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, pano_pred, pano, pano_id):
        log_dict = {
            'val/pano_pred': self.temp_wandb_image(
                pano_pred[0, 0], pano_id[0] if pano_id else None),
            'val/pano_gt': self.temp_wandb_image(
                pano[0, 0], pano_id[0] if pano_id else None),
        }
        self.logger.experiment.log(log_dict)


class FrechetAutoEncoderDistance(Metric):
    higher_is_better = False

    def __init__(self, pano_height: int):
        super().__init__()
        ckpt_path = os.path.join('weights', 'faed.ckpt')
        faed = FAED.load_from_checkpoint(ckpt_path)
        self.encoder = faed.net.encoder

        num_features = pano_height * 4
        mx_num_feets = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_num_feets).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_num_feets).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def get_activation(self, imgs):
        imgs = (imgs.type(torch.float32) / 127.5) - 1
        features = self.encoder(imgs)
        mean_feature = torch.mean(features, dim=3)
        weight = torch.cos(
            torch.linspace(math.pi / 2, -math.pi / 2, mean_feature.shape[-1], device=mean_feature.device)
            ).unsqueeze(0).unsqueeze(0).expand_as(mean_feature)
        mean_feature = weight * mean_feature
        mean_vector = mean_feature.view(-1, (mean_feature.shape[-2] * mean_feature.shape[-1]))
        return mean_vector

    def update(self, imgs: Tensor, real: bool):
        features = self.get_activation(imgs)
        features = features.double()
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)