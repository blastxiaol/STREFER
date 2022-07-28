import nltk
import re
from collections import defaultdict

wnl =  nltk.stem.WordNetLemmatizer()
def lemmatize_all(word):
    w, tag = nltk.pos_tag(nltk.word_tokenize(word))[0]
    if tag.startswith('NN'):
        return wnl.lemmatize(w, pos='n'), tag
    elif tag.startswith('VB'):
        return wnl.lemmatize(w, pos='v'), tag
    # elif tag.startswith('JJ'):
    #     return wnl.lemmatize(w, pos='a')
    elif tag.startswith('R'):
        return wnl.lemmatize(w, pos='r'), tag
    else:
        return w, tag

def count_word(file):
    word_count = defaultdict(int)
    word2tag = defaultdict(set)
    tag2word = defaultdict(set)
    for data in file:
        token = data['language']['token']
        for word in token:
            word = word.lower().split('-')
            for w in word:
                w = re.sub('[^\w ]','', w)
                if w:
                    w, tag = lemmatize_all(w)
                    word_count[w] += 1
                    word2tag[w].add(tag)
                    tag2word[tag].add(w)
    return word_count, word2tag, tag2word