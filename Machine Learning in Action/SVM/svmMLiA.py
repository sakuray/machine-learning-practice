
from numpy import *

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineAddr = line.strip().split('\t')
        dataMat.append([float(lineAddr[0]), float(lineAddr[1])])
        labelMat.append(float(lineAddr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H :
        aj = H
    if L > aj:
        aj = L
    return aj

'''
    简略版SMO算法：
    创建一个alpha向量，初始化为0向量
    迭代次数小于最大迭代次数（外循环）
        对数据集中每个数据向量处理（内循环）：
            如果该数据向量可以被优化：
                随机选择另外一个数据向量
                同时优化这两个向量
                如果两个向量都不能被优化，退出内循环
        如果所有的向量都没有被优化，增加迭代次数，进行下一次循环
'''
def smoSimple(dataMat, label, C, toler, maxIter):
    dataMatrix = mat(dataMat)
    labelMat = mat(label).transpose()
    b = 0
    m,n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b      # 预测的类别
            Ei = fXi - float(labelMat[i])   # 误差
            if((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)): # 误差很大进行优化
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix*dataMatrix[j,:].T)) +b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j ] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H : print("L==H"); continue
                eta = 2.0 * dataMatrix[i,:]* dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T  # alpha[j] 最优修改量
                if eta >= 0: print("eta>=0");continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H, L)
                if(abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough");continue
                alphas[i] += labelMat[j]*labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei -labelMat[i] *(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej -labelMat[i] *(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T-labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if(0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1+b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i: %d, pairs changed %d" %(iter, i, alphaPairsChanged))
        if(alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b, alphas


class optStruct:
    def __init__(self, dataMatIn, classLales, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLales
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X, self.X[i,:], kTup)

def calcEk(oS, k):
    # fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b
    # Ek = fXk - float(oS.labelMat[k])
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if(len(validEcacheList)) > 1:
        for k in validEcacheList:
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if(deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
        ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if(oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C+oS.alphas[j] -oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H : print("L==H"); return 0
        # eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j, j]  # 核函数版本
        if eta >=0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H, L)
        updateEk(oS, j)
        if(abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)
        # b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold) * oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        # b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold) * oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        b1 = oS.b - Ei - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if(0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif(0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else:oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=('lin',0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter<maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" %(iter, i, alphaPairsChanged))
            iter += 1
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-nound, iter: %d i: %d, pairs change %d" %(iter, i, alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False
        elif(alphaPairsChanged == 0): entireSet = True
        print("iteration number:%d" % iter)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w+= multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w


'''
    分布为内外的情况，使用核函数：径向基核函数
'''
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:]-A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2))
    else:raise NameError('Houston We have a problem that Kernel is not recognized')
    return K

def testRbf(k1=1.3):
    data, label = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(data, label, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(data)
    labelMat = mat(label).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :],('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) +b
        if sign(predict) != sign(label[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    data, label = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(data)
    labelMat = mat(label).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:],('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(label[i]): errorCount += 1
    print("the test error rate is: %f" %(float(errorCount)/m))

'''
    使用SVM完成手写识别字的算法
'''
def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def testDigits(kTup=('rbf', 10)):
    data, label = loadImages('trainingDigits')
    b, alphas = smoP(data, label, 200, 0.0001, 10000, kTup)
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    svInd = nonzero(alphas.A>0)[0]
    sVs = dataMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d support Vectors" % shape(sVs)[0])
    m, n = shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, dataMat[i, :],kTup)
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) +b
        if sign(predict) != sign(label[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    data, label = loadImages('testDigits')
    errorCount = 0
    datMat = mat(data)
    labelMat = mat(label).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i,:],kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(label[i]): errorCount += 1
    print("the test error rate is: %f" %(float(errorCount)/m))

if __name__ == '__main__':
    # data, label = loadDataSet('testSet.txt')
    # # b, alphas = smoSimple(data, label, 0.6, 0.001, 40)
    # b, alphas = smoP(data, label, 0.6, 0.001, 40)
    # ws = calcWs(alphas, data, label)
    # print(ws)
    #
    # datMat = mat(data)
    # print(datMat[0]*mat(ws)+b)

    # testRbf()
    testDigits(('rbf', 20))