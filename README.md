def aStarAlgo(start_node, stop_node):
open_set = set(start_node)
closed_set = set()
g = {} #store distance from starting node
parents = {}# parents contains an adjacency map of all nodes
#ditance of starting node from itself is zero
g[start_node] = 0
#start_node is root node i.e it has no parent nodes
#so start_node is set to its own parent node
parents[start_node] = start_node

while len(open_set) > 0:
n = None
#node with lowest f() is found
for v in open_set:
if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):
n = v

if n == stop_node or Graph_nodes[n] == None:
pass
else:
for (m, weight) in get_neighbors(n):
#nodes 'm' not in first and last set are added to first
#n is set its parent
if m not in open_set and m not in closed_set:
open_set.add(m)
parents[m] = n
g[m] = g[n] + weight

#for each node m,compare its distance from start i.e g(m) to the
#from start through n node
else:
if g[m] > g[n] + weight:
#update g(m)
g[m] = g[n] + weight
#change parent of m to n
parents[m] = n
#if m in closed set,remove and add to open
if m in closed_set:
closed_set.remove(m)
open_set.add(m)

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 2
if n == None:
print('Path does not exist!')
return None
# if the current node is the stop_node
# then we begin reconstructin the path from it to the start_node
if n == stop_node:
path = []
while parents[n] != n:
path.append(n)
n = parents[n]
path.append(start_node)
path.reverse()
print('Path found: {}'.format(path))
return path

# remove n from the open_list, and add it to closed_list
# because all of his neighbors were inspected
open_set.remove(n)
closed_set.add(n)
print('Path does not exist!')
return None
#define fuction to return neighbor and its distance
#from the passed node
def get_neighbors(v):
if v in Graph_nodes:
return Graph_nodes[v]
else:
return None
#for simplicity we ll consider heuristic distances given
#and this function returns heuristic distance for all nodes
def heuristic(n):
H_dist = {
'A': 11,
'B': 6,
'C': 99,
'D': 1,
'E': 7,
'G': 0,
}
return H_dist[n]

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 3
#Describe your graph here
Graph_nodes = {
'A': [('B', 2), ('E', 3)],
'B': [('C', 1),('G', 9)],
'C': None,
'E': [('D', 6)],
'D': [('G', 1)],
}
aStarAlgo('A', 'G')



2)class Graph:
def __init__(self, graph, heuristicNodeList, startNode): #instantiate graph object with graph
topology, heuristic values, start node
self.graph = graph
self.H=heuristicNodeList
self.start=startNode
self.parent={}
self.status={}
self.solutionGraph={}
def applyAOStar(self): # starts a recursive AO* algorithm
self.aoStar(self.start, False)
def getNeighbors(self, v): # gets the Neighbors of a given node
return self.graph.get(v,'')
def getStatus(self,v): # return the status of a given node
return self.status.get(v,0)
def setStatus(self,v, val): # set the status of a given node
self.status[v]=val
def getHeuristicNodeValue(self, n):
return self.H.get(n,0) # always return the heuristic value of a given node
def setHeuristicNodeValue(self, n, value):
self.H[n]=value # set the revised heuristic value of a given node

def printSolution(self):
print("FOR GRAPH SOLUTION, TRAVERSE THE GRAPH FROM THE START
NODE:",self.start)
print("------------------------------------------------------------")
print(self.solutionGraph)
print("------------------------------------------------------------")
def computeMinimumCostChildNodes(self, v): # Computes the Minimum Cost of child nodes of
a given node v
minimumCost=0
costToChildNodeListDict={}
costToChildNodeListDict[minimumCost]=[]
flag=True
for nodeInfoTupleList in self.getNeighbors(v): # iterate over all the set of child node/s
cost=0
nodeList=[]
for c, weight in nodeInfoTupleList:
cost=cost+self.getHeuristicNodeValue(c)+weight

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 5
nodeList.append(c)
if flag==True: # initialize Minimum Cost with the cost of first set of child
node/s
minimumCost=cost
costToChildNodeListDict[minimumCost]=nodeList # set the Minimum Cost child
node/s
flag=False
else: # checking the Minimum Cost nodes with the current Minimum
Cost
if minimumCost>cost:
minimumCost=cost
costToChildNodeListDict[minimumCost]=nodeList # set the Minimum Cost child
node/s

return minimumCost, costToChildNodeListDict[minimumCost] # return Minimum Cost
and Minimum Cost child node/s

def aoStar(self, v, backTracking): # AO* algorithm for a start node and backTracking status
flag
print("HEURISTIC VALUES :", self.H)
print("SOLUTION GRAPH :", self.solutionGraph)
print("PROCESSING NODE :", v)
print("-----------------------------------------------------------------------------------------")
if self.getStatus(v) >= 0: # if status node v >= 0, compute Minimum Cost nodes of v
minimumCost, childNodeList = self.computeMinimumCostChildNodes(v)
self.setHeuristicNodeValue(v, minimumCost)
self.setStatus(v,len(childNodeList))
solved=True # check the Minimum Cost nodes of v are solved
for childNode in childNodeList:
self.parent[childNode]=v
if self.getStatus(childNode)!=-1:
solved=solved & False
if solved==True: # if the Minimum Cost nodes of v are solved, set the current node
status as solved(-1)
self.setStatus(v,-1)
self.solutionGraph[v]=childNodeList # update the solution graph with the solved nodes
which may be a part of solution

if v!=self.start: # check the current node is the start node for backtracking the current
node value
self.aoStar(self.parent[v], True) # backtracking the current node value with
backtracking status set to true

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 6
if backTracking==False: # check the current call is not for backtracking
for childNode in childNodeList: # for each Minimum Cost child node
self.setStatus(childNode,0) # set the status of child node to 0(needs exploration)
self.aoStar(childNode, False) # Minimum Cost child node is further explored with
backtracking status as false
h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1, 'T': 3}
graph1 = {
'A': [[('B', 1), ('C', 1)], [('D', 1)]],
'B': [[('G', 1)], [('H', 1)]],
'C': [[('J', 1)]],
'D': [[('E', 1), ('F', 1)]],
'G': [[('I', 1)]]
}
G1= Graph(graph1, h1, 'A')
G1.applyAOStar()
G1.printSolution()
h2 = {'A': 1, 'B': 6, 'C': 12, 'D': 10, 'E': 4, 'F': 4, 'G': 5, 'H': 7} # Heuristic values of Nodes
graph2 = { # Graph of Nodes and Edges
'A': [[('B', 1), ('C', 1)], [('D', 1)]], # Neighbors of Node 'A', B, C & D with repective weights
'B': [[('G', 1)], [('H', 1)]], # Neighbors are included in a list of lists
'D': [[('E', 1), ('F', 1)]] # Each sublist indicate a "OR" node or "AND" nodes
}
G2 = Graph(graph2, h2, 'A') # Instantiate Graph object with graph, heuristic values
and start Node
G2.applyAOStar() # Run the AO* algorithm
G2.printSolution() # Print the solution graph as output of the AO* algorithm
search




3)import csv
def g_0(n):
return ("?",)*n

def s_0(n):
return ('ɸ',)*n

def more_general(h1, h2):
more_general_parts = []
for x, y in zip(h1, h2):
mg = x == "?" or (x != "ɸ" and (x == y or y == "ɸ"))
more_general_parts.append(mg)
return all(more_general_parts)

def fulfills(example, hypothesis):
### the implementation is the same as for hypotheses:
return more_general(hypothesis, example)

def min_generalizations(h, x):
h_new = list(h)
for i in range(len(h)):
if not fulfills(x[i:i+1], h[i:i+1]):
h_new[i] = '?' if h[i] != 'ɸ' else x[i]
return [tuple(h_new)]

def min_specializations(h, domains, x):
results = []
for i in range(len(h)):
if h[i] == "?":
for val in domains[i]:
if x[i] != val:

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 10
h_new = h[:i] + (val,) + h[i+1:]
results.append(h_new)
elif h[i] != "ɸ":
h_new = h[:i] + ('ɸ',) + h[i+1:]
results.append(h_new)
return results

with open('trainingexamples.csv') as csvFile:
examples = [tuple(line) for line in csv.reader(csvFile)]

def get_domains(examples):
d = [set() for i in examples[0]]
for x in examples:
for i, xi in enumerate(x):
d[i].add(xi)
return [list(sorted(x)) for x in d]

def candidate_elimination(examples):
domains = get_domains(examples)[:-1]

G = set([g_0(len(domains))])
S = set([s_0(len(domains))])
i = 0
print("\n G[{0}]:".format(i), G)
print("\n S[{0}]:".format(i), S)
for xcx in examples:
i = i + 1
x, cx = xcx[:-1], xcx[-1] # Splitting data into attributes and decisions
if cx == 'Y': # x is positive example
G = {g for g in G if fulfills(x, g)}
S = generalize_S(x, G, S)
else: # x is negative example
S = {s for s in S if not fulfills(x, s)}
G = specialize_G(x, domains, G, S)

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 11
print("\n G[{0}]:".format(i), G)
print("\n S[{0}]:".format(i), S)
return

def generalize_S(x, G, S):
S_prev = list(S)
for s in S_prev:
if s not in S:
continue
if not fulfills(x, s):
S.remove(s)
Splus = min_generalizations(s, x)
## keep only generalizations that have a counterpart in G
S.update([h for h in Splus if any([more_general(g,h)
for g in G])])
## remove hypotheses less specific than any other in S
S.difference_update([h for h in S if
any([more_general(h, h1)
for h1 in S if h != h1])])
return S
def specialize_G(x, domains, G, S):
G_prev = list(G)
for g in G_prev:
if g not in G:
continue
if fulfills(x, g):
G.remove(g)
Gminus = min_specializations(g, domains, x)
## keep only specializations that have a conuterpart in S
G.update([h for h in Gminus if any([more_general(h, s)
for s in S])])
## remove hypotheses less general than any other in G
G.difference_update([h for h in G if
any([more_general(g1, h)

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 12
for g1 in G if h != g1])])
return G
candidate_elimination(examples)




4)def infoGain(P, N):
import math
return -P / (P + N) * math.log2(P / ( P + N)) - N / (P + N) * math.log2(N / (P + N))

def insertNode(tree, addTo, Node):
for k, v in tree.items():
if isinstance(v, dict):
tree[k] = insertNode(v, addTo, Node)
if addTo in tree:
if isinstance(tree[addTo], dict):
tree[addTo][Node] = 'None'
else:
tree[addTo] = {Node:'None'}
return tree
def insertConcept(tree, addTo, Node):
for k, v in tree.items():
if isinstance(v, dict):
tree[k] = insertConcept(v, addTo, Node)
if addTo in tree:
tree[addTo] = Node
return tree
def getNextNode(data, AttributeList, concept, conceptVals, tree, addTo):
Total = data.shape[0]
if Total == 0:
return tree
countC = {}
for cVal in conceptVals:
dataCC = data[data[concept] == cVal]
countC[cVal] = dataCC.shape[0]
if countC[conceptVals[0]] == 0:
tree = insertConcept(tree, addTo, conceptVals[1])
return tree
if countC[conceptVals[1]] == 0:
tree = insertConcept(tree, addTo, conceptVals[0])

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 14
return tree
ClassEntropy = infoGain(countC[conceptVals[1]],countC[conceptVals[0]])
Attr = {}
for a in AttributeList:
Attr[a] = list(set(data[a]))
AttrCount = {}
EntropyAttr = {}
for att in Attr:
for vals in Attr [att]:
for c in conceptVals:
iData = data[data[att] == vals]
dataAtt = iData[iData[concept] == c]
AttrCount[c] = dataAtt.shape[0]
TotalInfo = AttrCount[conceptVals[1]] + AttrCount[conceptVals[0]]
if AttrCount[conceptVals[1]] == 0 or AttrCount[conceptVals[0]] == 0:
InfoGain=0
else:
InfoGain = infoGain(AttrCount[conceptVals[1]], AttrCount[conceptVals[0]])
if att not in EntropyAttr:
EntropyAttr[att] = ( TotalInfo / Total ) * InfoGain
else:
EntropyAttr[att] = EntropyAttr[att] + ( TotalInfo / Total ) * InfoGain
Gain = {}
for g in EntropyAttr:
Gain[g] = ClassEntropy - EntropyAttr[g]
Node = max(Gain, key = Gain.get)
tree = insertNode(tree, addTo, Node)
for nD in Attr[Node]:
tree = insertNode(tree, Node, nD)
newData = data[data[Node] == nD].drop(Node, axis = 1)
AttributeList=list(newData)[:-1]
tree = getNextNode(newData, AttributeList, concept, conceptVals, tree, nD)
return tree

def main():
import pandas as pd
data = pd.read_csv('id3.csv')

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 15
AttributeList = list(data)[:-1]
concept = str(list(data)[-1])
conceptVals = list(set(data[concept]))
tree = getNextNode(data, AttributeList, concept, conceptVals, {'root':'None'}, 'root')
print(tree)
compute(tree)
main()
OUTPUT:
The Resultant Decision Tree is :
{'Outlook': {'Overcast': 'Yes',
'Rain': {'Wind': {'Strong': 'No', 'Weak': 'Yes'}},
'Sunny': {'Humidity': {'High': 'No', 'Normal': 'Yes'}}}}
Best Attribute :
Outlook
Tree Keys:
dict_keys(['Overcast', 'Rain', 'Sunny'])
Accuracy is : 0.75



5)import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X) # max of array
y = y/100
def sigmoid (x):
return 1/(1 + np.exp(-x))
def derivatives_sigmoid(x):
return x * (1 - x)
epoch=5000
lr=0.1
wh = np.random.uniform(size=(2,3))
bh = np.random.uniform(size=(1,3))
wout = np.random.uniform(size=(3,1))
bout = np.random.uniform(size=(1,1))
for i in range(epoch):
# forward prop
hinp=np.dot(X,wh) + bh
hlayer_act = sigmoid(hinp)
outinp=np.dot(hlayer_act,wout) + bout
output = sigmoid(outinp)
hiddengrad = derivatives_sigmoid(hlayer_act)
outgrad = derivatives_sigmoid(output)
EO = y-output
d_output = EO* outgrad
EH = d_output.dot(wout.T)
d_hiddenlayer = EH * hiddengrad
wout += hlayer_act.T.dot(d_output) *lr
wh += X.T.dot(d_hiddenlayer) *lr

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 17
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output: \n" ,output)
OUTPUT:
Input:
[[0.66666667 1. ]
[0.33333333 0.55555556]
[1. 0.66666667]]
Actual Output:
[[0.92]
[0.86]
[0.89]]
Predicted Output:
[[0.69734296]
[0.68194708]
[0.69700956]]



6)import csv
import random
import math
def loadcsv(filename):
lines = csv.reader(open(filename, "r"))
dataset = list(lines)
for i in range(len(dataset)):
dataset[i] = [float(x) for x in dataset[i]]
return dataset
def splitDataset(dataset, splitRatio):
trainSize = int(len(dataset) * splitRatio)
trainSet = []
trainSet,testSet = dataset[:trainSize],dataset[trainSize:]
return [trainSet, testSet]
def mean(numbers):
return sum(numbers)/(len(numbers))
def stdev(numbers):
avg = mean(numbers)
v = 0
for x in numbers:
v += (x-avg)**2
return math.sqrt(v/(len(numbers)-1))
def summarizeByClass(dataset):
separated = {}
for i in range(len(dataset)):
vector = dataset[i]
if (vector[-1] not in separated):
separated[vector[-1]] = []
separated[vector[-1]].append(vector)
summaries = {}
for classValue, instances in separated.items():
summaries[classValue] = [(mean(attribute), stdev(attribute)) for attribute in
zip(*instances)][:-1]
return summaries

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 19
def calculateProbability(x, mean, stdev):
exponent = math.exp((-(x-mean)**2)/(2*(stdev**2)))
return (1 / ((2*math.pi)**(1/2)*stdev)) * exponent
def predict(summaries, inputVector):
probabilities = {}
for classValue, classSummaries in summaries.items():
probabilities[classValue] = 1
for i in range(len(classSummaries)):
mean, stdev = classSummaries[i]
x = inputVector[i]
probabilities[classValue] *= calculateProbability(x, mean, stdev)
bestLabel, bestProb = None, -1
for classValue, probability in probabilities.items():
if bestLabel is None or probability > bestProb:
bestProb = probability
bestLabel = classValue
return bestLabel
def getPredictions(summaries, testSet):
predictions = []
for i in range(len(testSet)):
result = predict(summaries, testSet[i])
predictions.append(result)
return predictions
def getAccuracy(testSet, predictions):
correct = 0
for i in range(len(testSet)):
if testSet[i][-1] == predictions[i]:
correct += 1
return (correct/(len(testSet))) * 100.0
filename = 'pima-indians-diabetes.csv'
splitRatio = 0.67
dataset = loadcsv(filename)
trainingSet, testSet = splitDataset(dataset, splitRatio)
summaries = summarizeByClass(trainingSet)
predictions = getPredictions(summaries, testSet)
print("\nPredictions:\n",predictions)
accuracy = getAccuracy(testSet, predictions)
print('Accuracy ',accuracy)

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 20
OUTPUT:
Naive Bayes Classifier for concept learning problem
Split 14 rows into
Number of Training data: 12
Number of Test Data: 2
The values assumed for the concept learning attributes are
OUTLOOK=> Sunny=1 Overcast=2 Rain=3
TEMPERATURE=> Hot=1 Mild=2 Cool=3
HUMIDITY=> High=1 Normal=2
WIND=> Weak=1 Strong=2
TARGET CONCEPT:PLAY TENNIS=> Yes=10 No=5
The Training set are:
[1.0, 1.0, 1.0, 1.0, 5.0]
The Test data set are:
[1.0, 1.0, 1.0, 2.0, 5.0]
The Test data set are:
[2.0, 1.0, 1.0, 1.0, 10.0]
The Test data set are:
[3.0, 2.0, 1.0, 1.0, 10.0]
The Test data set are:
[3.0, 3.0, 2.0, 1.0, 10.0]
The Test data set are:
[3.0, 3.0, 2.0, 2.0, 5.0]
The Test data set are:
[2.0, 3.0, 2.0, 2.0, 10.0]
The Test data set are:
[1.0, 2.0, 1.0, 1.0, 5.0]
The Test data set are:
[1.0, 3.0, 2.0, 1.0, 10.0]
The Test data set are:
[3.0, 2.0, 2.0, 1.0, 10.0]
The Test data set are:
[1.0, 2.0, 2.0, 2.0, 10.0]
The Test data set are:
[2.0, 2.0, 1.0, 2.0, 10.0]
The Test data set are:

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 21
[2.0, 1.0, 2.0, 1.0, 10.0]
[3.0, 2.0, 1.0, 2.0, 5.0]

Summarize Attributes By Class
{5.0: [(1.5, 1.0), (1.75, 0.9574271077563381), (1.25, 0.5), (1.5, 0.5773502691896257)], 10.0:
[(2.125, 0.8345229603962802), (2.25, 0.7071067811865476), (1.625, 0.5175491695067657),
(1.375, 0.5175491695067657)]}
Actual values: [5.0]%
Predictions: [5.0, 5.0]%
Accuracy: 50.0%



7)import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
l1 = [0,1,2]
def rename(s):
l2 = []
for i in s:
if i not in l2:
l2.append(i)
for i in range(len(s)):
pos = l2.index(s[i])
s[i] = l1[pos]
return s
iris = datasets.load_iris()
X = pd.DataFrame(iris.data,columns
=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width'] )
y = pd.DataFrame(iris.target,columns = ['Targets'])
def graph_plot(l,title,s,target):
plt.subplot(l[0],l[1],l[2])
if s==1:
plt.scatter(X.Sepal_Length,X.Sepal_Width, c=colormap[target], s=40)
else:
plt.scatter(X.Petal_Length,X.Petal_Width, c=colormap[target], s=40)
plt.title(title)
plt.figure()
colormap = np.array(['red', 'lime', 'black'])
graph_plot([1, 2, 1],'sepal',1,y.Targets)

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 23
graph_plot([1, 2, 2],'petal',0,y.Targets)
plt.show()
def fit_model(modelName):
model = modelName(3)
model.fit(X)
plt.figure()
colormap = np.array(['red', 'lime', 'black'])
graph_plot([1, 2, 1],'Real Classification',0,y.Targets)
if modelName == KMeans:
m = 'Kmeans’
else:
m = 'Em'
y1 = model.predict(X)
graph_plot([1, 2, 2],m,0,y1)
plt.show()
km = rename(y1)
print("\nPredicted: \n", km)
print("Accuracy ",sm.accuracy_score(y, km))
print("Confusion Matrix ",sm.confusion_matrix(y, km))

fit_model(KMeans)
fit_model(GaussianMixture)



8)#import the dataset and library files
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import train_test_split
iris_dataset=load_iris()
#display the iris dataset
print("\n IRIS FEATURES \ TARGET NAMES: \n ", iris_dataset.target_names)
for i in range(len(iris_dataset.target_names)):
print("\n[{0}]:[{1}]".format(i,iris_dataset.target_names[i]))
print("\n IRIS DATA :\n",iris_dataset["data"])
#split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"]
, random_state=0)
print("\n Target :\n",iris_dataset["target"])
print("\n X TRAIN \n", X_train)
print("\n X TEST \n", X_test)
print("\n Y TRAIN \n", y_train)
print("\n Y TEST \n", y_test)
#train and fit the model
kn = KNeighborsClassifier(n_neighbors=5)
kn.fit(X_train, y_train)
#predicting from model
x_new = np.array([[5, 2.9, 1, 0.2]])
print("\n XNEW \n",x_new)
prediction = kn.predict(x_new)
print("\n Predicted target value: {}\n".format(prediction))
print("\n Predicted feature name: {}\n".format(iris_dataset["target_names"][prediction]))
i=1
x= X_test[i]
x_new = np.array([x])
print("\n XNEW \n",x_new)

VII Semester 18CSL76 – Artificial Intelligence & Machine Learning Laboratory

Department of ISE, RNSIT 30
for i in range(len(X_test)):
x = X_test[i]
x_new = np.array([x])
prediction = kn.predict(x_new)
print("\n Actual : {0} {1}, Predicted :{2}{3}".format(y_test[i],iris_dataset["target_names
"][y_test[i]],prediction,iris_dataset["target_names"][ prediction]))
print("\n TEST SCORE[ACCURACY]: {:.2f}\n".format(kn.score(X_test, y_test)))




8)import numpy as np
import matplotlib.pyplot as plt
def local_regression(x0, X, Y, tau):
x0 = [1, x0]
X = [[1, i] for i in X]
X = np.asarray(X)
xw = (X.T) * np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * tau))
beta = np.linalg.pinv(xw @ X) @ xw @ Y @ x0
return beta
def draw(tau):
prediction = [local_regression(x0, X, Y, tau) for x0 in domain]
plt.plot(X, Y, 'o', color='black')
plt.plot(domain, prediction, color='red')
plt.show()
X = np.linspace(-3, 3, num=1000)
domain = X
Y = np.log(np.abs(X ** 2 - 1) + .5)
draw(10)
draw(0.1)
