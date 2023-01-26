import pandas as pd
import numpy as np
import os

path = "/u/boumghak/ProjectX2022/"
nom_csv = "tv_human_interaction.csv"
dataset_folder = "tv_human_interactions_videos/"

def csv_to_label_unique(csv):
    return list(pd.read_csv(csv)['label'].unique())

def folder_creater(path, labels):
    for x in labels:
        os.mkdir(path + x)

def csv_to_name(csv):
    return list(pd.read_csv(csv)['filename'])

def move_video(nom_video, labels, path_of_dataset):

    for nom, label in zip(nom_video, labels):
        os.rename(path_of_dataset + nom, path_of_dataset + label + "/" + nom)

    


def main():

    unique_labels = csv_to_label_unique(path + "datasets/videos/" + nom_csv)
    labels = list(pd.read_csv(path + "datasets/videos/" + nom_csv)['label'])
    nom_video = csv_to_name(path + "datasets/videos/" + nom_csv)


    #folder_creater(path + "/datasets/videos/" + dataset_folder, unique_labels)
    move_video(nom_video, labels, path + "/datasets/videos/" + dataset_folder)




#  -----------------------------------------------------------------------

if __name__ == "__main__":
    main()