import os 
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#CNN model
import pytorchvideo.models.resnet

class DatasetModule(pytorch_lightning.LightningDataModule):

    #Configuration of datasets
    DATA_PATH = "path"
    CLIP_LENGTH = 10
    BATCH_SIZE = 8

    def train_dataloader(self):
        train_data = pytorchvideo.data.LabeledVideoDataset(
            labeled_video_paths = os.path.join(self.DATA_PATH, "train"),
            clip_sampler = pytorchvideo.data.make_clip_sampler("random", self.CLIP_LENGTH),
            decode_audio = False,
        )

        return torch.utils.data.DataLoader(
            train_data,
            batch_size=self.BATCH_SIZE,
        )


    def validation_loader(self):

        validation_data = pytorchvideo.data.LabeledVideoDataset(
            labeled_video_paths = os.path.join(self.DATA_PATH, "test"),
            clip_sampler = pytorchvideo.data.make_clip_sampler("uniform", self.CLIP_LENGTH),
            decode_audio = False,
        )

        return torch.utils.data.DataLoader(
            validation_data,
            batch_size=self.BATCH_SIZE,
        )

#For not classifying after CNN output, output features
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x    

#Model
class CNN_RNN_Model(nn.Module):
    def __init__(self, params_model):

        dr_rate= params_model["dr_rate"]
        pretrained = params_model["pretrained"]
        rnn_hidden_size = params_model["rnn_hidden_size"]
        rnn_num_layers = params_model["rnn_num_layers"]
        
        cnn_model = models.resnet50(pretrained=pretrained)
        num_features = cnn_model.fc.in_features
        cnn_model.fc = Identity()

        self.cnn_model = cnn_model
        self.dropout = nn.Dropout(dr_rate)
        self.rnn = nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers)

    def forward(self, x):
        b_z, ts, c, h, w = x.shape
        i = 0
        y = self.cnn_model((x[:,i]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))

        for i in range(1, ts):
            y = self.baseModel((x[:,i]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))

        out = self.dropout(out[:,-1])

        return out


