import os
import os.path
from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO
from io import open
import pandas as pd
import gensim                 
import nltk
import re                      
import unicodedata
import sklearn
from sklearn.cluster import KMeans

def readPDF(pdfFile):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    process_pdf(rsrcmgr, device, pdfFile)
    device.close()

    content = retstr.getvalue()
    retstr.close()
    return content

def getFilePaths (path):
    filePaths = []
    for root, dirs, files in os.walk(path):
        for name in files:
            filePath = os.path.join(root,name)
            filePaths.append(filePath)
    return filePaths

def getPdfContent (filePaths):
    pdfContents = {}
    fileIndex = 0
    for filePath in filePaths:
        print(filePath)
        pdfFile = open(filePath, 'rb')
        pdfContents[fileIndex] = readPDF(pdfFile)
        pdfFile.close()
        fileIndex = fileIndex + 1
    return pdfContents

def corpus2DF (pdfContents):
    pdfDF = pd.DataFrame.from_dict(pdfContents, orient='index')
    return pdfDF

def setPreprocess ():
    default = [True, True, True, True, True, 0, True, "snowBall", "english", True, "english"]
    reset = input('Do you want to use default preprocess parameters or reset them?\n(y/n)')
    if reset == 'y':
        return default
    else:
        
        changingEncoding = input('Do you want to remove non-ascii symbols or convert non-ascii symbols?\n(y/n)')
        if changingEncoding == 'y':
            default[0] = True
        else:
            default[0] = False
        
        convertLower = input('Do you want to convert all text into lower case text?\n(y/n)')
        if convertLower == 'y':
            default[1] = True
        else:
            default[1] = False

        removeSpace = input('Do you want to remove multiple white space?\n(y/n)')
        if removeSpace == 'y':
            default[2] = True
        else:
            default[2] = False

        removeNumerics = input('Do you want to remove numerics?\n(y/n)')
        if removeNumerics == 'y':
            default[3] = True
        else:
            default[3] = False

        removePunct = input('Do you want to remove any punctuation symbols?\n(y/n)')
        if removePunct == 'y':
            default[4] = True
        else:
            default[4] = False

        removeLength = 0
        removeLength = int(input('What is the minimum length of word you want to retain?\n'))
        default[5] = removeLength

        stemWords = input('Do you want to remove endings of words?\n(y/n)')
        if stemWords == 'y':
            default[6] = True
        else:
            default[6] = False

        stopWords = input('Do you want to remove stopwords?\n(y/n)')
        if stopWords == 'y':
            default[9] = True
        else:
            default[9] = False

        return default

def charSet(text, char_ext_type):
    if char_ext_type == "ascii":
        return re.sub(r'[^\x00-\x7f]',r'',text)
    elif char_ext_type == "utf2ascii":
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    else:
        return text

def stemmerProc(text, stemmer, stemmer_language):
    if stemmer.lower() == "porter":
        stemmer = nltk.stem.porter.PorterStemmer()
        return stemmerProc1(text, stemmer, stemmer_language)
    elif stemmer.lower() == "snowball":
        if stemmer_language in nltk.stem.snowball.SnowballStemmer.languages:
            stemmer = nltk.stem.snowball.SnowballStemmer(stemmer_language)
            return stemmerProc1(text, stemmer, stemmer_language)
        else:
            return text
    else:
        return text

def stemmerProc1(text, stemmer, stemmer_language):
    list_text = nltk.tokenize.word_tokenize(text)
    tokens_stemmed = []
    for token in list_text:
        tokens_stemmed.append(stemmer.stem(token))
    return_text = ""
    for token in tokens_stemmed:
        return_text = return_text + " " + token
    return return_text.strip()

def stopwords(text, stopword_list):
    list_text = nltk.tokenize.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words(stopword_list)
    tokens_filtered = []
    for token in list_text:
        if token not in stopwords:
            tokens_filtered.append(token)
    return_text = ""
    for token in tokens_filtered:
        return_text += " "+token
    return return_text.strip()

def preprocess(text,
               feature,
               changingEncoding=False,
               convertLower=True,
               removeSpace=True,
               removeNumerics=True,
               removePunct=False,
               removeLength=0,
               stemWords=False,
               stemmer_algorithm="snowBall",
               stemmer_language="english",
               stopWords=False,
               stopwords_corpus="english"):
    (rows, _) = text.shape
    for row in range(0, rows):

        if changingEncoding == True:
            text[feature][row] = charSet(text[feature][row], changingEncoding)

        if convertLower == True:
            text[feature][row] = text[feature][row].lower()

        if removeSpace == True:
            text[feature][row] = gensim.corpora.textcorpus.strip_multiple_whitespaces(text[feature][row])

        if removeNumerics == True:
            text[feature][row] = gensim.parsing.preprocessing.strip_numeric(text[feature][row])

        if removePunct == True:
            text[feature][row] = gensim.parsing.preprocessing.strip_punctuation2(text[feature][row])

        if removeLength>0:
            text[feature][row] = gensim.corpora.textcorpus.remove_short(text[feature][row],minsize=removeLength)

        if stemWords == True:
            text[feature][row] = stemmerProc(text[feature][row], stemmer_algorithm, stemmer_language)

        if stopWords == True:
            text[feature][row] = stopwords(text[feature][row], stopwords_corpus)

    return text


def DF2Series (dataFrame):
    pdfSeries = pd.Series(dataFrame[0].values)
    return pdfSeries

def readPdf2Series(s):
    filePaths = [s]
    pdfContents = getPdfContent (filePaths)
    pdfDF = corpus2DF(pdfContents)
    default = [True, True, True, True, True, 0, True, "snowBall", "english", True, "english"]
    pdfDF = preprocess(pdfDF,
                       0,
                       changingEncoding=default[0],
                       convertLower=default[1],
                       removeSpace=default[2],
                       removeNumerics=default[3],
                       removePunct=default[4],
                       removeLength=default[5],
                       stemWords=default[6],
                       stemmer_algorithm="snowBall",
                       stemmer_language="english",
                       stopWords=default[9],
                       stopwords_corpus="english")
    testPDF = DF2Series(pdfDF)
    return testPDF



# #################################################
# Main code sections

# reading a corpus
filePaths = getFilePaths("D:\\1914666\\corpus")
pdfContents = getPdfContent (filePaths)
pdfDF = corpus2DF(pdfContents)
print(pdfDF)
print('\n')

default = setPreprocess()
print(default)
print('\n')

pdfDF = preprocess(pdfDF,
                   0,
                   changingEncoding=default[0],
                   convertLower=default[1],
                   removeSpace=default[2],
                   removeNumerics=default[3],
                   removePunct=default[4],
                   removeLength=default[5],
                   stemWords=default[6],
                   stemmer_algorithm="snowBall",
                   stemmer_language="english",
                   stopWords=default[9],
                   stopwords_corpus="english")
print(pdfDF)
print('\n')

#Learning

pdfSeries = DF2Series(pdfDF)
print(pdfSeries)
print('\n')
testPDF = readPdf2Series('D:\\1914666\\testcorpus//1909.00822.pdf')
totalPDF = pdfSeries.append(testPDF)

count_vect = sklearn.feature_extraction.text.CountVectorizer()
bow = count_vect.fit_transform(totalPDF)
print('Follwing is bow\n')
print(bow.toarray())
print('\n')

tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer()
tfidf = tfidf_transformer.fit_transform(bow)
print('Follow is tfidf\n')
print(tfidf.toarray())
print('\n')

kmeans = KMeans(n_clusters=3)
kmeans.fit(bow[:-1])
print('Followinng is kmeans\n')
print(kmeans.cluster_centers_)
print(kmeans.labels_)
    

kmeans2 = KMeans(n_clusters=3, max_iter = 2000)
kmeans2.fit(tfidf[:-1])
print('Followinng is kmeans2\n')
print(kmeans2.labels_)
print(kmeans2.cluster_centers_)

(rowtfidf, _) = tfidf.shape
distancelist = []
for n in range(0, rowtfidf-1):
    center = kmeans2.cluster_centers_[kmeans2.labels_[n]]
    point = 0
    distance = 0 
    for value in tfidf.toarray()[n]:
        centerpoint = kmeans2.cluster_centers_[kmeans2.labels_[n]][point]
        distance = distance + ((value - centerpoint)*(value - centerpoint))
        point = point + 1
    
    distancelist.append(distance)

print(distancelist)
print(max(distancelist))
print('\n')


#predict#################################
print('Predict by bow kmeans\n')
print(kmeans.predict(bow[-1]))

print('Predict by tfidf kmeans\n')
point = 0
testdistance = 0
for value in tfidf.toarray()[-1]:
    centerpoint = kmeans2.cluster_centers_[kmeans2.predict(tfidf[-1])[0]][point]
    testdistance = testdistance + ((value - centerpoint)*(value - centerpoint))
    point = point + 1
print(testdistance)
if testdistance > 1:
    print('The document is in a new field, so no relevant documents exists in the corpus.')
else:
    print(kmeans2.predict(tfidf[-1]))














        
