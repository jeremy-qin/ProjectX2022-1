import pandas as pd
import os
import re

from os.path import abspath

#  Description: http://lear.inrialpes.fr/people/marszalek/data/hoha/readme.txt


#  -----------------------------------------------------------------------

def get_hollywood_annotations_df(hollywood_path, df_path):
    """ 
    Parameters
    ----------
    hollywood_path: str
        path to holywood directory

    """
    data = []
    # read annotations files: filename, frames, label
    annotation_path = f'{hollywood_path}/annotations'
    for filename in os.listdir(annotation_path):
        file_path = os.path.join(annotation_path, filename)
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()

                # get annotations
                video_name = get_substring_between_char(line, '\"',  '\"')
                frames = get_substring_between_char(line, '(',  ')')
                label = get_substring_between_char(line, '<',  '>')
                frame_start, frame_end = frames.split('-')

                # add to list
                data.append([video_name, label, frame_start, frame_end])

    # create dataframe
    df = pd.DataFrame(data, columns=['filename', 'label', 'frame_start', 'frame_end'])

    # sort
    df = df.sort_values(by=['filename', 'label'])

    # save dataframe
    df.to_csv(df_path, index=False)




def get_substring_between_char(string, start_char, end_char):
    """ 

    Assumes that start_char is before end_char and that char are in the string

    Example
    -------
    >>> get_substring_between_char('"Fargo - 01984.avi" (188-266) <AnswerPhone>', start_char = '<', end_char='>')
    >>> AnswerPhone
    """
    regex = f'\\{start_char}(.*?)\\{end_char}'
    return re.findall(regex, string)[0]

#  -----------------------------------------------------------------------

def main():
    get_hollywood_annotations_df('/home/yukikongju/Downloads/hollywood', 'datasets/videos/hollywood.csv')
    

if __name__ == "__main__":
    main()
