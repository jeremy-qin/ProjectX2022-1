import numpy as np

from scipy.spatial.distance import minkowski


class PretrainedEmbedding(object):

    def __init__(self, embedding_path):
        self.embedding_path = embedding_path
        self.embedded_dict = self._init_embedding_dict()

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




#  -----------------------------------------------------------------------

def test_find_similar_words():
    #  embed_path = 'datasets/glove.6B.50d.txt' # from https://github.com/stanfordnlp/GloVe
    embed_path = 'datasets/glove.6B.50d.subset.txt' # head -n 1000 glove.6B.50d.txt > glove.6B.50d.txt
    embedding = PretrainedEmbedding(embed_path)
    vector = embedding.get_word_vector('stage')
    nearest_words = embedding.find_similar_words(vector, 10)
    print(nearest_words)


#  -----------------------------------------------------------------------


def main():
    test_find_similar_words()
    


if __name__ == "__main__":
    main()
