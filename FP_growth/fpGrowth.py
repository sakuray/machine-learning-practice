
'''
    FP-growth树结构
'''
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind = 1):
        print(' '* ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

'''
    创建树和头指针表
'''
def createTree(dataSet, minSup = 1):
    headerTable = {}
    # 构建元素表
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 过滤掉非频繁集
    for k in list(headerTable):
        if headerTable[k] < minSup:
            headerTable.pop(k)
    # 构建头指针表
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # 构建树
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        # 遍历一个数据项的各个个体总的出现次数
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)] # 按个体出现频繁度排序（只针对与该数据项）
            updateTree(orderedItems, retTree, headerTable, count) # 构建FP树
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:     # 该个体在当前树的孩子节点上，更新计数
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree) # 不在创建，并更新headerTable的指针指向
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:  # 该数据项超过一个个体，递归构建树节点
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    while(nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    simpDat = [
        ['r','z','h','j','p'],
        ['z','y','x','w','v','u','t','s'],
        ['z'],
        ['r','x','n','o','s'],
        ['y','r','x','z','q','t','p'],
        ['y','z','x','e','q','s','t','m']
    ]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

'''
    获取节点的前缀路径
'''
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p : p[0])]
    # print(bigL)
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        # print(newFreqSet, freqItemList)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # print(condPattBases)
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            # print('conditional tree for: ',newFreqSet)
            # myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

if __name__ == "__main__":
    # rootNode = treeNode('pyramid', 9, None)
    # rootNode.children['eye'] = treeNode('eye',13, None)
    # rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    # rootNode.disp()

    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, 3)
    # myFPtree.disp()
    # print(findPrefixPath('x', myHeaderTab['x'][1]))
    freqItems = []
    mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
    print(freqItems)

    # 该文件过大，没有提供，去https://www.manning.com/books/machine-learning-in-action下载源代码
    # parsedDat = [line.split() for line in open("kosarak.dat").readlines()]
    # initSet = createInitSet(parsedDat)
    # myFPtree, myHeaderTab = createTree(initSet, 100000)
    # myFreqList=[]
    # mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)
    # print(myFreqList)