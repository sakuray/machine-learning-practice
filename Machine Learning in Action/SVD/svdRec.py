
'''
    D(m,n) = U(m,m) * 奇异值∑(m,n) * V.T(n,n)
'''
from numpy import *
from numpy import linalg as la

def loadExData():
    # return [
    #     [1,1,1,0,0],
    #     [2,2,2,0,0],
    #     [1,1,1,0,0],
    #     [5,5,5,0,0],
    #     [1,1,0,2,2],
    #     [0,0,0,3,3],
    #     [0,0,0,1,1]
    # ]
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
'''
    协同过滤 相似度计算
'''
def ecluidSim(inA, inB):
    return 1.0/(1.0+la.norm(inA-inB))

'''
    皮尔逊相似度
'''
def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar=0)[0][1]

'''
    余弦相似度
    cosΘ = A * B / (||A||*||B||)
'''
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

'''
    推荐菜肴:
    比如物品1，2，3，A用户没有对3打过分， 其他用户评论过所有物品分值，对3进行A用户可能打的分进行估分：
    simTotal = 其他用户对1打分和3打分的分值近似度+对2打分和对3打分的近似度
    ratSimTotal = (3,1)近似度*A对1的打分+（3,2)近似度*A对2的打分
    估分 = ratSimTotal / simTotal
    基本思想：
    用户A对1，2打过分，没有打过3的分，其他用户对1，2，3进行打过分，
    所以预测用户如果对1打分，对3可能打多少分，用户对2打分，对3可能打多少分。
    所有的预测相加 平均一下就是A对3可能打的分数，近似度就相当于权重了
'''
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(dataMat[:,item].A > 0, dataMat[:,j].A > 0))[0]
        if len(overLap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print("the %d and %d similarity is : %f" %(item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal / simTotal

def recommend(dataMat, user, N = 3, simMeans=cosSim, estMethod = standEst):
    unratedItems = nonzero(dataMat[user,:].A == 0)[1]
    if len(unratedItems) == 0: return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimateScore = estMethod(dataMat, user, simMeans, item)
        itemScores.append((item, estimateScore))
    return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]

'''
    评分估计
'''
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = mat(eye(4)*Sigma[:4])
    xformedItems = dataMat.T * U[:,:4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T, \
                             xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal

'''
    压缩图片 原本是32*32=1024个数据，经过svd压缩后，只需要保存U sigma VT就可以了，numSV=2时就是32*2+32*2+2=130,压缩近10倍
'''
def printMat(inMat, thresh = 0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1,end=' ')
            else: print(0,end=' ')
        print('')

def imgCompress(numSV=3, thresh = 0.8):
    myl = []
    for line in open("0_5.txt").readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix****")
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print("****reconstructed matrix using %d singular values****" % numSV)
    printMat(reconMat, thresh)


if __name__ == "__main__":
    # U, Sigma, T = linalg.svd([[1,1],[7,7]])
    # print(U)
    # print(Sigma)
    # print(T)

    # Data = loadExData()
    # U, Sigma, VT = linalg.svd(Data)
    # print(Sigma)
    # Sig3 = mat([[Sigma[0], 0, 0],[0, Sigma[1], 0], [0, 0, Sigma[2]]])
    # D = U[:,:3]*Sig3*VT[:3,:]
    # print(D)

    myMat = mat(loadExData())
    # # print(ecluidSim(myMat[:,0], myMat[:,4]))
    # # print(ecluidSim(myMat[:,0], myMat[:,0]))
    # # print(pearsSim(myMat[:,0], myMat[:,4]))
    # # print(pearsSim(myMat[:,0], myMat[:,0]))
    # # print(cosSim(myMat[:,0], myMat[:,4]))
    # # print(cosSim(myMat[:,0], myMat[:,0]))
    #
    # myMat[0,1] = myMat[0,0] = myMat[1,0] = myMat[2,0] = 4
    # myMat[3,3] = 2
    # # print(myMat)
    # print(recommend(myMat,2))
    # print(recommend(myMat,2, simMeans=ecluidSim))
    # print(recommend(myMat,2, simMeans=pearsSim))

    # U, Sigma, VT = la.svd(mat(loadExData2()))
    # print(Sigma)

    # print(recommend(myMat, 1, estMethod=svdEst))
    # print(recommend(myMat, 1, estMethod=svdEst, simMeans=pearsSim))

    imgCompress(2)