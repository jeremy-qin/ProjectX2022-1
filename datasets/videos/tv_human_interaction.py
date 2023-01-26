import pandas as pd
import numpy as np
import os

path = "/u/boumghak/ProjectX2022/"
def generate_tv_human_interaction_df(videos_directory, df_path):
    """ 
    Generate dataframe with columns `filename` and `label` for TV Human 
    interaction dataframe

    Parameters
    ----------
    directory: str
        absolute or relative path of the `tv_human_interactions_videos` files. 
    """
    data = []

    # get labels for all videos
    for filename in os.listdir(videos_directory):
        label = filename.split('_')[0].lower()
        data.append([filename, label])

    # create dataset
    df = pd.DataFrame(data, columns=['filename', 'label'])

    # sort dataset
    df = df.sort_values(by=['filename', 'label'])

    # save dataset
    df.to_csv(df_path, index=False)

    return df

#  -----------------------------------------------------------------------


def main():
    df = generate_tv_human_interaction_df(path + 'datasets/videos/tv_human_interactions_videos', path + 'datasets/videos/tv_human_interaction.csv')
    pd.DataFrame.info(df)



#  -----------------------------------------------------------------------

if __name__ == "__main__":
    main()
