
from time import sleep


def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

'''
    支持度计算，筛选频繁度高的
'''
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

'''
    生成关联规则
'''
def generateRules(L, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            # print(H1,"---",freqSet)
            if(i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return  bigRuleList

'''
    计算 freqSet-conseq -> conseq的可信度
    conseq的集合是H
'''
def calcConf(freqSet, H, supportData, brl, minConf = 0.7):
    prunedH = []
    for conseq in H :
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet-conseq,"-->",conseq, "conf:",conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    # print(freqSet,"===",H)
    if(len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H, m+1) # 进行组合
        # print(Hmp1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf) # 计算可信度：如{0，2}->{1,3}可信
        if(len(Hmp1) > 1): # 存在可信的，继续计算其推导的可信度{0}->{1,2,3}的可信的，如果上面不可信，后面不需要进行推导
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


if __name__ == "__main__":
    dataSet = loadDataSet()
    # C1 = createC1(dataSet)
    # print(C1)
    # D = map(set, dataSet)
    # L1, suppData0 = scanD(list(D), C1, 0.5)
    # print(L1)

    # L, suppData = apriori(dataSet)
    # print(L)

    # L, suppData = apriori(dataSet, minSupport=0.5)
    # print(L)
    # rules = generateRules(L, suppData, minConf=0.7)
    # print(rules)

    # 毒蘑菇
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    L, suppData = apriori(mushDatSet, minSupport=0.3)
    # rules = generateRules(L, suppData, minConf=0.9)
    # print(rules)
    # 查看毒蘑菇
    for item in L[1]:
        if item.intersection('2'):
            print(item)