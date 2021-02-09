# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:19:43 2019

@author: vedant
"""

# imports here
import pandas as pd
import numpy as np
import random as rd
import queue as qu
import sys

# required input for execution of file

if len(sys.argv)!=7:
    sys.exit("Please give the required amount of arguments #- <L> <K> <training-set> <validation-set> <test-set> <to-print> L: integer (used in the post-pruning algorithm) K: integer (used in the post-pruning algorithm) to-print:{yes,no}")
else:
    L = sys.argv[1]
    K = sys.argv[2]
    trainPath = sys.argv[3]
    validationPath = sys.argv[4]
    testPath = sys.argv[5]
    if sys.argv[6]=='yes':
        toPrint = True
    elif sys.argv[6]=='no':
        toPrint=False
    
# reading all files
trainingSet = pd.read_csv(trainPath)
validationSet = pd.read_csv(validationPath)
testSet = pd.read_csv(testPath)

print('Please wait till the results are published...')

# Data Structure used in creating Decision Tree
class Tree:
    def __init__(self):
        self.left = None
        self.right = None
        self.node = None
        self.leftData = None
        self.rightData = None
        self.parent = None

# Entropy calculation for Information Gain Heuristics
def entropy(dataset):
    var, count = np.unique(dataset['Class'], return_counts=True)
    entropy=0
    for i in range(len(var)):
        entropy+=(-count[i]/sum(count))*np.log2(count[i]/sum(count))
    return entropy

# Calculating Final Information Gain Heuristics to choose best attribute to split
def InfoGain(dataset,splitAttr):
    totalEntropy = entropy(dataset)
    var, count = np.unique(dataset[splitAttr], return_counts=True)
    weightedEntropy = 0
    for i in range(len(var)):
        weightedEntropy += (count[i]/sum(count))*entropy(dataset.where(dataset[splitAttr]==var[i]).dropna())
    return totalEntropy-weightedEntropy
    
# Inpurity calculation for Gain in Variance Impurity
def VarImpurity(dataset):
    var, count = np.unique(dataset['Class'], return_counts=True)
    mult=1
    for i in count:
        mult*=(i/sum(count))
    return mult
    
# Calculating Gain in Variance Impurity to choose best attribute
def ImpurityGain(dataset,splitAttr):    
    varImpurity = VarImpurity(dataset)
    var, count = np.unique(dataset[splitAttr], return_counts=True)
    weightedImpurity = sum([(count[i]/sum(count))*VarImpurity(dataset.where(dataset[splitAttr]==var[i]).dropna()) for i in range(len(var))])
    return varImpurity-weightedImpurity

# Creating Decision Tree based on a given strategy, return type dictionary and printing the tree
def DecisionTreeAlgo(algoStrategy,dataset,originaldata,features,toPrint,pipe,ParentNodeClass = None, ):
    if len(np.unique(dataset['Class'])) <= 1:
        subtree = np.unique(dataset['Class'])[0]
        if toPrint:
            print(' {0}'.format(subtree), end=" ") 
        return subtree;
    elif len(dataset)==0:
        subtree = np.unique(originaldata['Class'])[np.argmax(np.unique(originaldata['Class'],return_counts=True)[1])]
        if toPrint:
            print(' {0}'.format(subtree), end=" ") 
        return subtree
    elif len(features) ==0:
        if toPrint:
            print(' {0}'.format(ParentNodeClass), end=" ") 
        return ParentNodeClass
    else:
        ParentNodeClass = np.unique(dataset['Class'])[np.argmax(np.unique(dataset['Class'],return_counts=True)[1])]
        if algoStrategy=='InfoGain':
            itemValues = [InfoGain(dataset,feature) for feature in features]
        elif algoStrategy=='ImpurityGain':
            itemValues = [ImpurityGain(dataset,feature) for feature in features]
        bestFeatureIndex = np.argmax(itemValues)
        bestFeature = features[bestFeatureIndex]
        tree = {bestFeature+str(ParentNodeClass):{}}
        features = [i for i in features if i != bestFeature]
        for value in np.unique(dataset[bestFeature]):
            if toPrint:    
                print()
                for i in range(0,pipe):
                    print("| ",end=" ")
                print('{0} = {1} :'.format(bestFeature, value), end=" ")
            croppedData = dataset.where(dataset[bestFeature] == value).dropna()
            croppedData = croppedData.astype('int64')
            subtree = DecisionTreeAlgo(algoStrategy,croppedData,dataset,features,toPrint,pipe+1,ParentNodeClass)
            tree[bestFeature+str(ParentNodeClass)][value] = subtree
        return(tree)   

# Calculate percentage of accuracy based on given dataset and decision tree (in dictionary format) 
def Prediction(Dataset, DecisionTree):
    c=0
    for index, row in Dataset.iterrows():
        DecisionTreeCopy = DecisionTree.copy()
        while DecisionTreeCopy!=0 and DecisionTreeCopy!=1:
            val = row[list(DecisionTreeCopy)[0][:-1]]
            try:
                DecisionTreeCopy = DecisionTreeCopy[list(DecisionTreeCopy)[0]][val]
            except:
                break
        if DecisionTreeCopy==row['Class']:
            c+=1
    return (c/Dataset.shape[0])*100


def PostPruning(l, k, DecisionTree, BSTree, validationSet):
    BestDecisionTree = DecisionTree.copy()
    for i in range(l):
        NewDecisionTree = DecisionTree.copy()
        m = rd.randint(1,k)
        for j in range(m):
            n = CountNonLeafNodes(BSTree)
            p = rd.randint(1,n)
            li = BFSTraversal(BSTree, p)
            tempTree = NewDecisionTree
            for i in range(1,len(li)):
                if tempTree[li[i-1]][0]==tempTree(li[i]):
                   tempTree = tempTree[li[i-1]][0]
                elif tempTree[li[i-1]][1]==tempTree(li[i]):
                    tempTree = tempTree[li[i-1]][1]
            tempTree[li[:-1]] = int(li[-1])
        if Prediction(validationSet, BestDecisionTree) < Prediction(validationSet, NewDecisionTree):
            BestDecisionTree = NewDecisionTree.copy()
    return BestDecisionTree
   
# converts decision tree in dictionary to Tree Data Structure as defined above        
def PopulateTree(DecisionTree, parElement=None):
    if DecisionTree ==0 or DecisionTree ==1:
        return DecisionTree
    else:
        element = Tree()
        element.parent = parElement
        element.node = list(DecisionTree)[0]
        element.leftData = list(DecisionTree[list(DecisionTree)[0]])[0]
        element.rightData = list(DecisionTree[list(DecisionTree)[0]])[1]
        element.left = PopulateTree(DecisionTree[list(DecisionTree)[0]][element.leftData], element)
        element.right = PopulateTree(DecisionTree[list(DecisionTree)[0]][element.rightData], element)
        return element
    
# counts total number of non leaf nodes (n) used in Post Pruning Function
def CountNonLeafNodes(root):
    if root in [0,1]:
        return 0
    elif root.left in [0,1] and root.right in [0,1]:
        return 0
    else:
        return 1+CountNonLeafNodes(root.left)+CountNonLeafNodes(root.right)
 
# returns a list which consist of the column names from parent till pth node 
def BFSTraversal(root, p):
    queue = qu.Queue(maxsize=0)
    counter=1
    queue.put(root)
    while counter<=p:
        temp = queue.get()
        if temp.leftData not in [0,1]:
            queue.put(temp.leftData)
        if temp.rightData not in [0,1]:   
            queue.put(temp.rightData)
        counter+=1;
    li = []
    while temp.parent != None:
        li.append(temp.node)
        temp = temp.parent
    li = li.reverse
    return li
             
InfoGainTree = DecisionTreeAlgo('InfoGain', trainingSet,trainingSet,trainingSet.columns[:-1],toPrint,0,"")

print()
print('Accuracy found on Training Set before post-pruning in Information Gain Tree: ',Prediction(trainingSet, InfoGainTree))
print()
print('Accuracy found on Validation Set before post-pruning in Information Gain Tree: ',Prediction(validationSet, InfoGainTree))
print()
print('Accuracy found on Test Set before post-pruning in Information Gain Tree: ',Prediction(testSet, InfoGainTree))
print()

InfoBSTree = PopulateTree(InfoGainTree)
BestDecisionInfoGainTree = PostPruning(int(L),int(K),InfoGainTree,InfoBSTree, validationSet)

print('Accuracy found on Test Set after post-pruning in Information Gain Tree: ',Prediction(testSet, BestDecisionInfoGainTree))
input('Press Enter to continue...')
print()

VarImpurityTree = DecisionTreeAlgo('ImpurityGain', trainingSet,trainingSet,trainingSet.columns[:-1],toPrint,0,"")

print()
print('Accuracy found on Training Set before post-pruning in Variance Impurity Tree: ',Prediction(trainingSet, VarImpurityTree))
print()
print('Accuracy found on Validation Set before post-pruning in Variance Impurity Tree: ',Prediction(validationSet, VarImpurityTree))
print()
print('Accuracy found on Test Set before post-pruning in Variance Impurity Tree: ',Prediction(testSet, VarImpurityTree))
print()

VarBSTree = PopulateTree(VarImpurityTree)
BestDecisionInfoGainTree = PostPruning(int(L),int(K),VarImpurityTree,VarBSTree, validationSet)

print('Accuracy found on Test Set after post-pruning in Information Gain Tree: ',Prediction(testSet, BestDecisionInfoGainTree))
input('Press Enter to continue...')
print()