import os 
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
#CNN model
import pytorchvideo.models.resnet

class DatasetModule(pytorch_lightning.LightningDataModule):

    #Configuration of datasets
    DATA_PATH = "C:\\Users\\jerem\\Documents\\CUCAI\\ProjectX2022\\datasets\\videos\\tv_human_interactions_videos"
    CLIP_LENGTH = 10
    BATCH_SIZE = 1

    def train_dataloader(self):
        train_data = pytorchvideo.data.labeled_video_dataset(
            data_path = self.DATA_PATH,
            # labeled_video_paths = os.path.join(self.DATA_PATH, "train"),
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
class CNN_RNN_Model(pytorch_lightning.LightningModule):
    def __init__(self, params_model):
        super(CNN_RNN_Model, self).__init__()
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
        b_z, c, ts, h, w = x.shape
        x = np.transpose(x, (0, 2, 1, 3, 4))
        print(x.shape)
        i = 0
        print(x[:,i].shape)
        y = self.cnn_model((x[:,i]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))

        for i in range(1, ts):
            y = self.cnn_model((x[:,i]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))

        out = self.dropout(out[:,-1])

        return out

# def train(model, params):
#     num_epochs=params["num_epochs"]
#     loss_func=params["loss_func"]
#     opt=params["optimizer"]
#     train_dl=params["train_dl"]

#     for epoch in range(num_epochs):
#         model.train()

#     return 


def main():
    videos = DatasetModule()

    params_model={
        "dr_rate": 0.1,
        "pretrained" : True,
        "rnn_num_layers": 1,
        "rnn_hidden_size": 100,}

    model = CNN_RNN_Model(params_model)
    train_dl = videos.train_dataloader()
    out = []
    for batch_idx, batch in enumerate(train_dl):
        out.append(model(batch["video"]))
    print(out)
    print("Done")

main()