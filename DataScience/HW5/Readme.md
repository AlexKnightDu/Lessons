# <center> 作业五 </center>


### Problem 	
Python编程实现Apriori算法,能够从交易数据集发现频繁项集,并生成关联规则.另外要求生成一个交易数据集,验证算法实现的正确性.

### Solution 
交易数据集通过规定项目种类、范围以及每条记录中的数量范围来随机生成。程序整体上按照算法步骤进行逐一实现： 

```
L1 = {频繁1-项集};for (k = 1; Lk != empty_set; k++) do	Ck+1 = Lk自连接生成的候选项集	删除Ck+1中有非频繁k-项集的项集	for t in D do   		将被t 包含的Ck+1中项集的计数加1	Lk+1 = Ck+1中至少具有min_sup的项集	return	all Lk;
```

为了处理较多数据的情况，可以进一步优化Apriori算法：
  
- 基于hash表的项集计数  
将每个项集通过相应的hash函数映射到hash表中的不同的桶中，这样可以通过将桶中的项集技术跟最小支持计数相比较先淘汰一部分项集。  
- 事务压缩（压缩进一步迭代的事务数）  
不包含任何k-项集的事务不可能包含任何(k+1)-项集，这种事务在下一步的计算中可以加上标记或删除  

因为对某一个元素要成为K维项目集的一元素的话，该元素在k-1阶频繁项目集中的计数次数必须达到K-1个，否则不可能生成K维项目集。所以在逐层搜索循环过程的第k步中，根据k-1步生成的k-1维频繁项目集来产生k维候选项目集，由于在产生k-1维频繁项目集时，我们可以实现对该集中出现元素的个数进行计数处理，因此对某元素而言，若它的计数个数不到k-1的话，可以事先删除该元素，从而排除由该元素将引起的大规格所有组合。在得到了这个候选项目集后，可以对数据库D的每一个事务进行扫描，若该事务中至少含有候选项目集Ck中的一员则保留该项事务，否则把该事物记录与数据库末端没有作删除标记的事务记录对换，并对移到数据库末端的事务记录作删除标一记，整个数据库扫描完毕后为新的事务数据库D’ 中。  
	因此随着K 的增大，D’中事务记录量大大地减少，对于下一次事务扫描可以大大节约I/0 开销。由于每条记录中可能只有几项交易，因此这种虚拟删除的方法可以实现大量的交易记录在以后的挖掘中被剔除出来，在所剩余的不多的记录中再作更高维的数据挖掘是可以很大地节约时间的。  
  进一步优化可以通过FP-tree来减少扫描次数，也就是FP-Growth算法。

### Implementation
```python 
import numpy as np

def generate_data(more, less, num, category):
    data = []
    for i in range(num):
        record = set()
        n = np.random.randint(more, less)
        while(len(record) < more):
            record.add('A' + str(np.random.randint(1,category+1)))
        for j in range(n - more):
            record.add('A' + str(np.random.randint(1,category+1)))
        data += [list(record)]
    return data

def create_C1(data):
    C1 = set()
    for datum in data:
        for item in datum:
            items = frozenset([item])
            C1.add(items)
    return C1

def pruning(Ck_item, Lksub1):
    for item in Ck_item:
        sub_Ck = Ck_item - frozenset([item])
        if sub_Ck not in Lksub1:
            return False
    return True


def create_Ck(Lksub1, k):
    Ck = set()
    len_Lksub1 = len(Lksub1)
    list_Lksub1 = list(Lksub1)
    for i in range(len_Lksub1):
        for j in range(1, len_Lksub1):
            l1 = list(list_Lksub1[i])
            l2 = list(list_Lksub1[j])
            l1.sort()
            l2.sort()
            if l1[0:k-2] == l2[0:k-2]:
                Ck_item = list_Lksub1[i] | list_Lksub1[j]
                if pruning(Ck_item, Lksub1):
                    Ck.add(Ck_item)
    return Ck

def optimize(items, Lk, k):
    items_count = {}
    for item in items:
        items_count[item] = 0
        for item_set in Lk:
            if item in item_set:
                items_count[item] += 1
    for item in items:
        if items_count[item] < k:
            for item_set in Lk:
                if item in item_set:
                    # print('--->',item_set)
                    Lk.remove(item_set)


def get_Lk_by_Ck(data, items, Ck, min_support, support_data):
    Lk = set()
    item_count = {}
    for datum in data:
        for item in Ck:
            if item.issubset(datum):
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
    t_num = float(len(data))
    for item in item_count:
        if (item_count[item] / t_num) >= min_support:
            Lk.add(item)
            support_data[item] = item_count[item] / t_num
    optimize(items, Lk, 0)
    return Lk

def apriori(data, items, min_support):
    support_data = {}
    C1 = create_C1(data)
    L1 = get_Lk_by_Ck(data, items, C1, min_support, support_data)
    Lksub1 = L1.copy()

    L = []
    L.append(Lksub1)

    k = 2
    while True:
        Ck = create_Ck(Lksub1, k)
        Lk = get_Lk_by_Ck(data, items, Ck, min_support, support_data)
        if (len(Lk) == 0):
            break
        Lksub1 = Lk.copy()
        L.append(Lksub1)
        k = k + 1
    return L, support_data

def get_rules(L, support_data, min_conf):
    rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and rule not in rule_list:
                        rule_list.append(rule)
            sub_set_list.append(freq_set)
    return rule_list


def main():
    # The number of items in each records is more than 4
    # The number of items in each records is less than 9
    # The number of records is 10
    # The number of categories of item is 7
    data = generate_data(4,9,10,7)
    items = set()
    print('Data: ')
    for i in range(0, len(data)):
        data[i].sort()
        items |= set(data[i])
        print(data[i])
    L, support_data = apriori(data, items, min_support=0.5)
    rules = get_rules(L, support_data, min_conf=0.5)

    print('-' * 100)

    for Lk in L:
        for freq_set in Lk:
            print(str(freq_set)[10:-1], '=>' , support_data[freq_set])

    print('-' * 100)

    for item in rules:
        print(str(item[0])[10:-1], "=>", str(item[1])[10:-1], "conf: ", round(item[2],2))

main()
```


### Result
```
Data: 
['A1', 'A2', 'A6', 'A7']
['A2', 'A3', 'A4', 'A5', 'A6']
['A1', 'A4', 'A5', 'A6', 'A7']
['A1', 'A2', 'A3', 'A5']
['A3', 'A4', 'A5', 'A6']
['A1', 'A2', 'A3', 'A5', 'A6']
['A1', 'A2', 'A4', 'A5', 'A6', 'A7']
['A1', 'A2', 'A3', 'A4', 'A6', 'A7']
['A1', 'A2', 'A6', 'A7']
['A2', 'A4', 'A6', 'A7']
----------------------------------------------------------------------------------------------------
{'A6'} => 0.9
{'A2'} => 0.8
{'A4'} => 0.6
{'A1'} => 0.7
{'A3'} => 0.5
{'A7'} => 0.6
{'A5'} => 0.6
{'A1', 'A6'} => 0.6
{'A6', 'A5'} => 0.5
{'A7', 'A2'} => 0.5
{'A6', 'A7'} => 0.6
{'A6', 'A2'} => 0.7
{'A1', 'A7'} => 0.5
{'A1', 'A2'} => 0.6
{'A6', 'A4'} => 0.6
{'A1', 'A6', 'A2'} => 0.5
{'A1', 'A6', 'A7'} => 0.5
{'A6', 'A7', 'A2'} => 0.5
----------------------------------------------------------------------------------------------------
{'A1'} => {'A6'} conf:  0.86
{'A6'} => {'A1'} conf:  0.67
{'A5'} => {'A6'} conf:  0.83
{'A6'} => {'A5'} conf:  0.56
{'A7'} => {'A2'} conf:  0.83
{'A2'} => {'A7'} conf:  0.62
{'A7'} => {'A6'} conf:  1.0
{'A6'} => {'A7'} conf:  0.67
{'A2'} => {'A6'} conf:  0.87
{'A6'} => {'A2'} conf:  0.78
{'A7'} => {'A1'} conf:  0.83
{'A1'} => {'A7'} conf:  0.71
{'A1'} => {'A2'} conf:  0.86
{'A2'} => {'A1'} conf:  0.75
{'A4'} => {'A6'} conf:  1.0
{'A6'} => {'A4'} conf:  0.67
{'A1', 'A2'} => {'A6'} conf:  0.83
{'A1', 'A6'} => {'A2'} conf:  0.83
{'A6', 'A2'} => {'A1'} conf:  0.71
{'A2'} => {'A1', 'A6'} conf:  0.62
{'A1'} => {'A6', 'A2'} conf:  0.71
{'A6'} => {'A1', 'A2'} conf:  0.56
{'A1', 'A7'} => {'A6'} conf:  1.0
{'A6', 'A7'} => {'A1'} conf:  0.83
{'A1', 'A6'} => {'A7'} conf:  0.83
{'A7'} => {'A1', 'A6'} conf:  0.83
{'A1'} => {'A6', 'A7'} conf:  0.71
{'A6'} => {'A1', 'A7'} conf:  0.56
{'A7', 'A2'} => {'A6'} conf:  1.0
{'A6', 'A7'} => {'A2'} conf:  0.83
{'A6', 'A2'} => {'A7'} conf:  0.71
{'A6'} => {'A7', 'A2'} conf:  0.56
{'A2'} => {'A6', 'A7'} conf:  0.62
{'A7'} => {'A6', 'A2'} conf:  0.83
```