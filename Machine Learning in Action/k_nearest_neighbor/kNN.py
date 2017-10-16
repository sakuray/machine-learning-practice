from numpy import *
import operator
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0],[0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''
    1.计算输入数据和训练集的每条记录之间的距离
    2.按距离递增排序
    3.选择前k个记录
    4.确定k个记录标签出现的频率
    5.返回最大的频率做为结果
    
    inX:输入的数据特征向量
    dataSet:训练集
    labels:对应训练集的标签
    k:选择前n个最短距离的训练集数据
'''
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

'''
    约会配对
    条件：1.每年获得的飞行常客里程数
          2.玩视频游戏所耗时间百分比
          3.每周消费的冰淇淋公升数
    标签：1.不喜欢的人
         2.魅力一般的人
         3.魅力高的人
'''
def file2martix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLine = len(arrayOLines)
    returnMat = zeros((numberOfLine, 3))
    labelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split("\t")
        returnMat[index,:] = listFromLine[0:3]
        labelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, labelVector


def showPicture(symbol1, symbol2, label):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(symbol1, symbol2, 15 * array(label), 15 * array(label))
    plt.show()

"""
    计算距离的时候会发现由于特征的取值范围悬殊，导致欧式距离计算取决于值大的那个特征，如:
        （12-8）^2+（26000-24000）^2
    很显然会由第二个特征决定距离，这和所想的不一样，所以需要对特征的取值归一化，都放缩到[0,1]或者[-1,1]之间
        （n-min）/(max-min)
    dataSet:特征值矩阵
"""
def autoNorm(dataSet):
    minVals = dataSet.min(0)    # 每列最小
    maxVals = dataSet.max(0)    # 每列最大
    ranges = maxVals - minVals
    m = dataSet.shape[0]    # 记录数
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

'''
    测试训练器成功率
'''
def datingClassTest(data, label):
    hoRatio = 1 # 前百分之多少是测试集，后面的都是训练集
    testMat, testLabel = file2martix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(testMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        result = classify0(normMat[i,:], data, label, 3)
        print("the classifier came back with: %s, the real answer is : %s" %(result, testLabel[i]))
        if(result != testLabel[i]):
            errorCount += 1
    print("the total error rate is: %f" %(errorCount/numTestVecs))

def classifyPerson():
    resultList=['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    tranMat, tranLabel = file2martix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(tranMat)
    inArr = array([ffMiles, percentTats, iceCream])
    result = classify0((inArr-minVals) / ranges, normMat, tranLabel, 3)
    print("you will probably like this person: ", resultList[result - 1])

if __name__ == "__main__":
    # group, labels = createDataSet()
    # print(classify0([0, 0], group, labels, 3))

    # data, signs = file2martix("datingTestSet2.txt")
    # print(signs[0:20])
    # showPicture(data[:,0], data[:,1], signs)

    # normMat, ranges, minVals = autoNorm(data)
    # print(normMat)
    # datingClassTest(normMat, signs)

    classifyPerson()

