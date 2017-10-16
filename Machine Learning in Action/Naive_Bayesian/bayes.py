'''
    朴素贝叶斯分类器有两种实现方式：
        1.基于贝努利模型实现：只考虑是否，不考虑次数，即假设等权重
        2.基于多项式模型实现：考虑次数

        p(c,w)=p(w|c)*p(c)/p(w)
        p(w|c)=p(w0|c)*p(w1|c)...*p(wn|c)
'''

from numpy import *
import re
import feedparser

def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

'''
    字典创建
    dataSet：n篇文章
'''
def createVocabList(dataSet):
    vocabSet =set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

'''
    出现次数
    一篇文章在字典中各个单词出现的次数
'''
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [1] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 2
        else:
            print("the word: %s is not in my Vocabulary!" % word )
    return returnVec

'''
    计算p(c)  p(w0|c)*p(w1|c)*...*p(wn|c)
'''
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  # 训练集数量
    numWords = len(trainMatrix[0])   # 字典单词数量
    pAbusive = sum(trainCategory)/float(numTrainDocs) # 计算p(c) sum计算出现侮辱词的个数/整个训练集数量
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 2
    p1Denom = 2
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:       # 当前文章是侮辱类文章
            p1Num += trainMatrix[i]     # 统计各个单词出现的次数
            p1Denom += sum(trainMatrix[i]) # 总共不重复单词数量
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive     # 计算各类文章各个字典单词出现的频率p(wn|c) p(c)

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):
    p1 = sum(vec2Classify * p1Vec) + log(pClass)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass)
    if(p1>p0):
        return 1
    else:
        return 0

def testNB():
    data, clas = loadDataSet()
    vocab = createVocabList(data)
    trainMat = []
    for doc in data:
        trainMat.append(setOfWords2Vec(vocab, doc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(clas))
    testEntry = ['love','my','dalmation']
    thisDoc = array(setOfWords2Vec(vocab, testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(vocab,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc, p0V, p1V, pAb))

'''
    词袋模型
'''
def bagOfWords2VecMN(vocal, inputSet):
    returnVec = [1] * len(vocal)
    for word in inputSet:
        if word in vocal:
            returnVec[vocal.index(word)] += 1
        # else:
            # print("the word: %s is not in my Vocabulary!" % word )
    return returnVec

'''
    切割文本，找到单词
'''
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

'''
    垃圾邮件分类
'''
def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1, 26):

        wordList = textParse(open('email\spam\%d.txt' %i).read())

        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email\ham\%d.txt' %i).read())
        # print(i)
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocab = createVocabList(docList)
    trainSet = list(range(50))
    testSet=[]
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat =[]
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setOfWords2Vec(vocab, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocab, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount+=1
    print('the error rate is:', float(errorCount)/len(testSet))

def calcMostFreq(vocab, fullText):
    import operator
    freqDict = {}
    for token in vocab:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    docList=[]
    classList=[]
    fullText=[]
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocab = createVocabList(docList)
    top30Words = calcMostFreq(vocab, fullText)
    for pairW in top30Words:        # 去掉高频词，即一些基本词
        if pairW[0] in vocab :
            vocab.remove(pairW[0])
    trainSet = list(range(2*minLen))
    testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainSet)))
        testSet.append(trainSet[randIndex])
        del(trainSet[randIndex])
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(bagOfWords2VecMN(vocab, docList[docIndex]))
        trainClass.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClass))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocab, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount)/ len(testSet))
    return vocab, p0V, p1V

def getTopWord(ny, sf):
    import operator
    vocab, p0V, p1V = localWords(ny, sf)
    # print(vocab)
    # print(p0V)
    # print(p1V)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > - 7 : topSF.append((vocab[i], p0V[i]))
        if p1V[i] > - 7 : topNY.append((vocab[i], p1V[i]))
    sortedSF = sorted(topSF, key= lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    count = 1
    for item in sortedSF:
        if count % 10 == 0: print()
        print(item[0],end=' ')
        count += 1;
    print()
    sortedNY = sorted(topNY, key= lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    count = 1
    for item in sortedNY:
        if count % 10 == 0: print()
        print(item[0],end=' ')
        count += 1

if __name__ == "__main__":
    # testNB()
    # spamTest()

    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # vocab,pSF, pNY = localWords(ny, sf)
    getTopWord(ny, sf)

