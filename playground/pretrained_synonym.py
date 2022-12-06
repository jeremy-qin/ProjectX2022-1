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
        
    def _get_vector_dim(self):
        return len(self.embedded_dict[next(iter(self.embedded_dict))])


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


    def get_new_word_vect(self, words, ponderations): 
        """ 
        given words and ponderation, create new words: w'=p1w1+p2w2+...+pnwn
        """
        # check if words in dict
        for word in words: 
            if word not in self.embedded_dict.keys():
                raise ValueError(f'the word {word} is not a key in the dict. please check')

        # check if ponderation sum to 1
        if sum(ponderations) != 1:
            raise ValueError("the sum of ponderation should be 1. check")

        # generate new vect
        new_vect = np.zeros(self._get_vector_dim())
        for word, ponderation in zip(words, ponderations):
            vector = self.embedded_dict[word]
            new_vect += ponderation * vector

        return new_vect

    


#  -----------------------------------------------------------------------

def test_find_similar_words():
    #  embed_path = 'datasets/glove.6B.50d.txt' # from https://github.com/stanfordnlp/GloVe
    embed_path = 'datasets/embeddings/glove.6B.50d.subset.txt' # head -n 1000 glove.6B.50d.txt > glove.6B.50d.txt
    embedding = PretrainedEmbedding(embed_path)
    vector = embedding.get_word_vector('stage')
    nearest_words = embedding.find_similar_words(vector, 10)
    print(nearest_words)

def test_visualization():
    embed_path = 'datasets/embeddings/glove.6B.50d.subset.txt' # head -n 1000 glove.6B.50d.txt > glove.6B.50d.txt
    embedding = PretrainedEmbedding(embed_path)
    embedding.visualize()
    
def test_get_vocab():
    embed_path = 'datasets/embeddings/glove.6B.50d.subset.txt' # head -n 1000 glove.6B.50d.txt > glove.6B.50d.txt
    embedding = PretrainedEmbedding(embed_path)
    print(embedding.get_vocab())
    


def test_get_new_word_vect():
    embed_path = 'datasets/embeddings/glove.6B.50d.subset.txt' # head -n 1000 glove.6B.50d.txt > glove.6B.50d.txt
    embedding = PretrainedEmbedding(embed_path)
    words = ['summer', 'tour', 'morning', 'championship']
    ponderations = [0.3, 0.1, 0.5, 0.1]
    new_vect = embedding.get_new_word_vect(words, ponderations)
    nearest_words = embedding.find_similar_words(new_vect, 10)
    print(nearest_words)


#  -----------------------------------------------------------------------


def main():
    #  test_find_similar_words()
    #  test_visualization()
    #  test_get_vocab()
    test_get_new_word_vect()
    


if __name__ == "__main__":
    main()
