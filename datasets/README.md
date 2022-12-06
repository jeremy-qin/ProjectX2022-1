# Datasets

Guide to install the datasets

## Pretrained Word Embedding (for NLP)

Download the following word embedding insides the `datasets/embeddings/` folder. 

- [X] [GloVe](https://github.com/stanfordnlp/GloVe)
- [X] [FastText](https://fasttext.cc/docs/en/english-vectors.html)

## Datasets


**TV Human Interaction**

1. Download `tv_human_interactions_videos.tar.gz` and `tv_human_interactions_annotations.tar.gz` from [here](https://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html)
2. Unzip the files with `tar xvzf <file>.tar.gz`
3. To get the dataframe with the filename and its label, run the script: `python3 videos/tv_human_interactions.py`

TODO: video annotations (frame with the action), profile (front, behind, left, ..)

**Hollywood Human Interaction (CVPR08)**

1. Download `hollywood.tar.gz` from [here](https://www.di.ens.fr/~laptev/download.html#actionclassification). Annotations for this datasets can be found in this link
2. Unzip tar file with `tar xvzf hollywood.tar.gz`
3. To get the dataframe with the filename, label and frames , run the script: `python3 videos/hollywood.py`

**UT - Interaction**


TODO



# More Datasets

- [ ] [Reddit and Twitter Sentiment Analysis](https://www.kaggle.com/datasets/cosmos98/twitter-and-reddit-sentimental-analysis-dataset?select=Twitter_Data.csv)
- [ ] [Emotions Datasets for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp?select=train.txt)
- [ ] [Knowledge Graph and NLP](https://www.kaggle.com/code/pavansanagapati/knowledge-graph-nlp-tutorial-bert-spacy-nltk)

