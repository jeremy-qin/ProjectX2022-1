import numpy as np

from scipy.spatial.distance import minkowski


def read_pretrained_embed(embed_path): 
    """ 
    Given file with word and its vector, return dictionary word and vector

    """
    embedded_dict = {}
    with open(embed_path, 'r') as f:
        for line in f.readlines():
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float32)
            embedded_dict[word] = vector
    return embedded_dict
    

def find_similar_words(embedded_dict, vector, n, dist_fun=minkowski): # TODO: check if n > vocab size, 
    """ 
    Given word as vector, return top n words closest to it
    """
    nearest = sorted(embedded_dict.keys(), key=lambda word: dist_fun(embedded_dict[word], vector))
    top_nearest = nearest[:n]
    return top_nearest

def get_word_vector(embedded_dict, word): # TODO: check if word in dict
    """ 
    given a word in the embedded dict, return its vectorization
    """
    return embedded_dict[word]
    



#  -----------------------------------------------------------------------

def test_read_pretrained_embed():
    #  embed_path = 'datasets/glove.6B.50d.txt' # from https://github.com/stanfordnlp/GloVe
    embed_path = 'datasets/glove.6B.50d.subset.txt' # head -n 1000 glove.6B.50d.txt > glove.6B.50d.txt
    embed_dict = read_pretrained_embed(embed_path)

def test_find_similar_words():
    embed_path = 'datasets/glove.6B.50d.subset.txt'
    embedded_dict = read_pretrained_embed(embed_path)
    vector = get_word_vector(embedded_dict, 'stage')
    nearest_words = find_similar_words(embedded_dict, vector, 10)
    print(nearest_words)

    




#  -----------------------------------------------------------------------


def main():
    #  test_read_pretrained_embed()
    test_find_similar_words()
    


if __name__ == "__main__":
    main()
