import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import minkowski
from sklearn.manifold import TSNE


class PretrainedEmbedding(object):

    def __init__(self, embedding_path):
        self.embedding_path = embedding_path
        self.embedded_dict = self._init_embedding_dict()

    def __len__(self):
        return len(self.embedded_dict)
        

    def _init_embedding_dict(self):
        embedded_dict = {}
        with open(self.embedding_path, 'r') as f:
            for line in f.readlines():
                values = line.split(' ')
                word = values[0]
                vector = np.asarray(values[1:], dtype=np.float32)
                embedded_dict[word] = vector

        return embedded_dict

        
    def find_similar_words(self, vector, n, dist_fun=minkowski): # TODO: check if n > vocab size, vector good size
        """ 
        Given word as vector, return top n words closest to it
        """
        nearest = sorted(self.embedded_dict.keys(), key=lambda word: dist_fun(self.embedded_dict[word], vector))
        top_nearest = nearest[:n]
        return top_nearest
        

    def get_word_vector(self, word): 
        """ 
        given a word in the embedded dict, return its vectorization
        """
        try:
            return self.embedded_dict[word]
        except ValueError:
            print('The word {word} is not in the dictionary')

    def get_vocab(self):
        return list(self.embedded_dict.keys())

    def visualize(self):
        n = 100
        tsne = TSNE(n_components=2)
        vocabs = self.get_vocab()[:n]
        vectors = np.asarray([self.embedded_dict[word] for word in vocabs])
        y = tsne.fit_transform(vectors)
        plt.scatter(y[:, 0], y[:, 1])
        for label, x, y in zip(vocabs, y[:, 0], y[:, 1]):
            plt.annotate(label, xy=(x,y), xytext=(0,0), textcoords='offset points')
        plt.title("TSNE Word Embedding visualization")
        plt.show()

        


#  -----------------------------------------------------------------------

def test_find_similar_words():
    #  embed_path = 'datasets/glove.6B.50d.txt' # from https://github.com/stanfordnlp/GloVe
    embed_path = 'datasets/glove.6B.50d.subset.txt' # head -n 1000 glove.6B.50d.txt > glove.6B.50d.txt
    embedding = PretrainedEmbedding(embed_path)
    vector = embedding.get_word_vector('stage')
    nearest_words = embedding.find_similar_words(vector, 10)
    print(nearest_words)

def test_visualization():
    embed_path = 'datasets/glove.6B.50d.subset.txt' # head -n 1000 glove.6B.50d.txt > glove.6B.50d.txt
    embedding = PretrainedEmbedding(embed_path)
    embedding.visualize()
    

#  -----------------------------------------------------------------------


def main():
    #  test_find_similar_words()
    test_visualization()
    


if __name__ == "__main__":
    main()
