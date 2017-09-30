# 数字识别kNN
from numpy import *
from os import listdir
import operator

def classify0(intX, dataSet, labels, k):
    # 计算距离
    dataSetSize = dataSet.shape[0]  #纵向长度，即数据集大小
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet # tile将输入值向量变成矩阵，减去训练集得到每个特征点的差距
    sqDiffMat = diffMat ** 2    # 差距平方 欧式距离 (x-a)^2 + (y-b)^2+ ...最后开根号
    sqDistances = sqDiffMat.sum(axis=1) # 计算各个特征点的距离平方和
    distances = sqDistances ** 0.5  # 开根号  这就是到每个训练集的差距了
    sortedDistIndicies = distances.argsort()    # 排序,返回从小到大的索引
    # 选择前k个点,统计出现次数
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序，选出最高的那个标签
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def img2vector(filename):
    one = zeros((1, 32*32))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            one[0, 32*i+j] = int(line[j])
    return one

def handwritingClassTest():
    label=[]
    trainFileList = listdir('train')
    m = len(trainFileList)
    trainMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split('_')[0])
        label.append(classNumStr)
        trainMat[i,:] = img2vector('train/%s' % fileNameStr)
    testFileList = listdir('test')
    errorCount = 0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split(".")[0]
        classNumStr = int(fileStr.split("_")[0])
        vectorUnderTest = img2vector('test/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainMat, label, 3)
        print("the classifier came back with: %d, the real answer is: %d" %(classifierResult, classNumStr))
        if(classifierResult != classNumStr):
            errorCount += 1
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount/mTest))

if __name__ == "__main__":
    handwritingClassTest()
    # print(img2vector('train/0_1.txt')[0,0:31])