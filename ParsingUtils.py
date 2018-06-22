import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def parseClean(text2clean):
    import re
    cleanedText = re.sub('[^a-zA-Z]',' ',text2clean)
    cleanedText = cleanedText.lower().split()
    ps = PorterStemmer()
    cleanedText = [ps.stem(word).encode("ascii") for word in cleanedText if not word in set(stopwords.words('english'))]
    return cleanedText

def vectorizeCorpus(corpus):
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    X=cv.fit_transform(corpus).toarray()
    return X

def importData(dataDirectory):
    """
    :param fileName:
    :return:
    """
    f = open(dataDirectory)
    labels = []
    smsWords = []
    corpus = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        if linedatas[0] == 'ham':
            labels.append(0)
        elif linedatas[0] == 'spam':
            labels.append(1)
        words = parseClean(linedatas[1])
        smsWords.append(words)
        corpus.append(' '.join(words))
    X=vectorizeCorpus(corpus)
    return X,labels