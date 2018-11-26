# <center> 作业二 </center>


#### Problem
探索iris(鸢尾花)数据集```pythonfrom sklearn import datasetsimport numpy as npiris = datasets.load_iris()```
一、探索iris的以下属性:
   - .DESCR:数据集总体描述  - .data:所有观测实例的特征数据(numpy.ndarray类型)  - .data.shape:维信息 - .feature_names:特征名(sepal萼片,petal花瓣)- .target:所有实例的类别值(numpy.ndarray类型),0/1/2- .target.shape:维信息- .target_names:类别名称(setosa/versicolor/virginica)例如,

```pythonprint iris.dataprint iris.target```

##### Solution

```python
# .DESCR :

Iris Plants Database
====================
Notes
-----
Data Set Characteristics:
    :Number of Instances: ...
    :Number of Attributes: ...
    :Attribute Information: ...
    :Summary Statistics:

    ============== ==== ==== ======= ===== ====================
                    Min  Max   Mean    SD   Class Correlation
    ============== ==== ==== ======= ===== ====================
    sepal length:   4.3  7.9   5.84   0.83    0.7826
    sepal width:    2.0  4.4   3.05   0.43   -0.4194
    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)
    petal width:    0.1  2.5   1.20  0.76     0.9565  (high!)
    ============== ==== ==== ======= ===== ====================

    :Missing Attribute Values: None
    ...

This is a copy of UCI ML iris datasets.
http://archive.ics.uci.edu/ml/datasets/Iris
...

References
----------
...

可以看出.DESCR给出了该数据集的描述，包括具体代码细节，数据的统计特征以及诸多数据集的相关信息。

# .data
[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]
 ...
 [ 5.9  3.   5.1  1.8]]
 
数据是4维数组，包含了对应的特征的值

# .data.shape
(150, 4)

数据为150x4的数组

# .feature_names
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

数据集包含这4种特征

# .target
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]

数据集中各个数据对应的鸢尾花的种类，分别对应.target_names的类型

# .target.shape
(150,)

鸢尾花的分类结果为150x1的数组

# .target_names
['setosa' 'versicolor' 'virginica']

鸢尾花的3种类型
```

##### Implementation

```python 
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

print(iris.DESCR)
print(iris.data)
print(iris.data.shape)
print(iris.feature_names)
print(iris.target)
print(iris.target.shape)
print(iris.target_names)
```


<p><br></br></p>



二、探索各特征的最小值,最大值,均值,中位数,标准差.  
##### Solution
直接利用numpy库中的统计函数可以求得各个统计特征：

```python 
sepal length (cm)
Min:	4.3
Max:	7.9
Mean:	5.84333333333
Median:	5.8
Std:	0.825301291785
-----------------------
sepal width (cm)
Min:	2.0
Max:	4.4
Mean:	3.054
Median:	3.0
Std:	0.432146580071
-----------------------
petal length (cm)
Min:	1.0
Max:	6.9
Mean:	3.75866666667
Median:	4.35
Std:	1.75852918341
-----------------------
petal width (cm)
Min:	0.1
Max:	2.5
Mean:	1.19866666667
Median:	1.3
Std:	0.760612618588
-----------------------
```

##### Implementation
```python
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

sta = np.zeros([4,5])

for i in range(0,4):
    print(iris.feature_names[i])
    datum = iris.data.T[i]
    print("Min:\t" + str(min(datum)))
    print("Max:\t" + str(max(datum)))
    print("Mean:\t" + str(np.mean(datum)))
    print("Median:\t" + str(np.median(datum)))
    print("Std:\t" + str(np.std(datum)))
```



三、探索各特征之间,以及特征与目标之间的相关性(相关系数).例如,

```pythonnp.corrcoef(iris.data.T[0],iris.target)
>> array([[1.        , 0.78256123],         [0.78256123, 1.        ]])```说明萼片长度与类别有一定相关度.##### Solution 

```python 
sepal length (cm)
Correlation with sepal width (cm):
[[ 1.         -0.10936925]
 [-0.10936925  1.        ]]
Correlation with petal length (cm):
[[ 1.          0.87175416]
 [ 0.87175416  1.        ]]
Correlation with petal width (cm):
[[ 1.          0.81795363]
 [ 0.81795363  1.        ]]
Correlation with target: 
[[ 1.          0.78256123]
 [ 0.78256123  1.        ]]
-----------------------
sepal width (cm)
Correlation with sepal length (cm):
[[ 1.         -0.10936925]
 [-0.10936925  1.        ]]
Correlation with petal length (cm):
[[ 1.        -0.4205161]
 [-0.4205161  1.       ]]
Correlation with petal width (cm):
[[ 1.         -0.35654409]
 [-0.35654409  1.        ]]
Correlation with target: 
[[ 1.        -0.4194462]
 [-0.4194462  1.       ]]
-----------------------
petal length (cm)
Correlation with sepal length (cm):
[[ 1.          0.87175416]
 [ 0.87175416  1.        ]]
Correlation with sepal width (cm):
[[ 1.        -0.4205161]
 [-0.4205161  1.       ]]
Correlation with petal width (cm):
[[ 1.         0.9627571]
 [ 0.9627571  1.       ]]
Correlation with target: 
[[ 1.          0.94904254]
 [ 0.94904254  1.        ]]
-----------------------
petal width (cm)
Correlation with sepal length (cm):
[[ 1.          0.81795363]
 [ 0.81795363  1.        ]]
Correlation with sepal width (cm):
[[ 1.         -0.35654409]
 [-0.35654409  1.        ]]
Correlation with petal length (cm):
[[ 1.         0.9627571]
 [ 0.9627571  1.       ]]
Correlation with target: 
[[ 1.          0.95646382]
 [ 0.95646382  1.        ]]
-----------------------

即：

			sepal length  sepal width  petal length  petal width  target
sepal length	0.0			-0.109		0.872		0.818		0.783		
sepal width		-0.109		0.0			-0.421		-0.357		-0.419		
petal length	0.872		-0.421		0.0			0.963		0.949		
petal width		0.818		-0.357		0.963		0.0			0.956	


```##### Analysis
从相关系数可以看出，萼片长度与类别之间的相关系数为 0.783 ，即存在一定的相关性。##### Implementation
```python
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

sta = np.zeros([4,5])

for i in range(0,4):
    datum = iris.data.T[i]
    for j in range(0,4):
        if (i != j):
            compare = iris.data.T[j]
            print("Correlation with " + iris.feature_names[j] + ":\n" + str(np.corrcoef(datum,compare)))
            sta[i][j] = (np.corrcoef(datum,compare))[0][1]
    print("Correlation with target: \n" + str(np.corrcoef(datum,iris.target)))
    sta[i][4] = (np.corrcoef(datum,iris.target))[0][1]
    print('-----------------------')

for i in range(0,4):
    s = ""
    for j in range(0,5):
        s += (str(round(sta[i][j],3)) + '\t\t\t')
    print(s)

```<!--<p><br></br></p>
<p><br></br></p>
<p><br></br></p>-->


