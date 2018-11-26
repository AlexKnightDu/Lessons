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
                    items.remove(item)


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
    # optimize(items, Lk, 0)
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




