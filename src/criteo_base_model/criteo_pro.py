# _*_ coding: utf-8 _*_

import numpy as np
import pandas as pd
import collections
from csv import DictReader
from datetime import datetime
import random


field = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'C13', 'C14', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C25', 'C26', 'C27', 'C29',
           'C30', 'C31', 'C32', 'C34', 'C35', 'C36', 'C37', 'C38']

table = collections.defaultdict(lambda: 0)
def txt2csv(rp, wp):
    f = open(rp, 'r')
    o = open(wp, 'w')
    #填写头部
    length = len(next(f).split('\t'))
    if length == 39:
        header = 'N0,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32,C33,C34,C35,C36,C37,C38\n'
    else:
        header = 'label,N0,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32,C33,C34,C35,C36,C37,C38\n'
    o.write(header)

    for i in f:
        i = i.replace('\t', ',')
        o.write(i)
    f.close()
    o.close()

def getDfMean(rp, columns):
    df = pd.read_csv(rp, usecols=columns)
    df_mean = df.mean()
    del df
    return round(df_mean, 3) #浮点数 保留三位

def getDfMode(rp, columns):
    df = pd.read_csv(rp, usecols=columns)
    df_mode = df.mode().iloc[0]  #mode() 取众数
    del df
    return df_mode

def handleMissingValue(rp, wp, missingList):
    df = pd.read_csv(rp, chunksize=1024)
    for index, chunk in enumerate(df, start=0):
        if index == 0:
            chunk = chunk.fillna(missingList)
            chunk.to_csv(wp, sep=',', header=True, index=False, mode='w')
        else:
            chunk = chunk.fillna(missingList)
            chunk.to_csv(wp, sep=',', header=False, index=False, mode='a+')
def getBins(rp, column):
    df = pd.read_csv(rp, usecols=[column])
    c = pd.qcut(df[column].rank(method='first').values, 5, labels=['one', 'two', 'three', 'four', 'five'], retbins=False)
    #print(type(c))
    return c

def field_index(x):
    index = field.index(x)
    return index


def getIndices(key):
    indices = table.get(key)
    if indices is None:
        indices = len(table)
        table[key] = indices
    return indices

if __name__ == '__main__':
    read_path = 'criteo/'
    write_path = 'criteo/'

    train_rp_0 = read_path + 'train.txt'
    train_wp_0 = write_path + '0.train_origin.csv'
    test_rp_0 = read_path + 'test.txt'
    test_wp_0 = write_path + 'test_no_label.csv'

    txt2csv(train_rp_0, train_wp_0)
    print('train completed !!!')
    txt2csv(test_rp_0, test_wp_0)
    print('test completed !!!')

    train_rp_1 = read_path + '0.train_origin.csv'
    train_wp_1 = write_path + '1.train_filled.csv'

    columns = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12', 'C13', 'C14', 'C15',
               'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29',
               'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38']

    # 按列逐步地找出各列的均值或众数，再把这些数值拼接起来，形成missinglist
    list_0 = getDfMean(train_rp_1, columns[0:4])
    list_1 = getDfMean(train_rp_1, columns[4:8])
    list_2 = getDfMean(train_rp_1, columns[8:13])
    list_3 = getDfMode(train_rp_1, columns[13:18])
    list_4 = getDfMode(train_rp_1, columns[18:23])
    list_5 = getDfMode(train_rp_1, columns[23:28])
    list_6 = getDfMode(train_rp_1, columns[28:33])
    list_7 = getDfMode(train_rp_1, columns[33:])
    missingList = pd.concat([list_0, list_1, list_2, list_3, list_4, list_5, list_6, list_7])

    handleMissingValue(train_rp_1, train_wp_1, missingList)

    print('Train has been filled !!!')

    rp = './criteo/1.train_filled.csv'
    wp0 = './criteo/2.train_filled_bin.csv'
    # wp1 = '../../data/criteo/n1.csv'

    columns = ['N0', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'N8', 'N9', 'N10', 'N11', 'N12']

    l0 = list(getBins(rp, columns[0]))
    print('l0', len(l0), 'is prepared !!!')
    l1 = list(getBins(rp, columns[1]))
    print('l1', len(l1), 'is prepared !!!')
    l2 = list(getBins(rp, columns[2]))
    print('l2 is prepared !!!')
    l3 = list(getBins(rp, columns[3]))
    print('l3 is prepared !!!')
    l4 = list(getBins(rp, columns[4]))
    print('l4 is prepared !!!')
    l5 = list(getBins(rp, columns[5]))
    print('l5 is prepared !!!')
    l6 = list(getBins(rp, columns[6]))
    print('l6 is prepared !!!')
    l7 = list(getBins(rp, columns[7]))
    print('l7 is prepared !!!')
    l8 = list(getBins(rp, columns[8]))
    print('l8 is prepared !!!')
    l9 = list(getBins(rp, columns[9]))
    print('l9 is prepared !!!')
    l10 = list(getBins(rp, columns[10]))
    print('l10 is prepared !!!')
    l11 = list(getBins(rp, columns[11]))
    print('l11 is prepared !!!')
    l12 = list(getBins(rp, columns[12]))
    print('l12 is prepared !!!')

    w = open(wp0, 'w')
    for i in zip(l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12):
        w.write(str(i[0]))
        for item in i[1:]:
            w.write(',' + str(item))
        w.write('\n')
    print('finished !!!')
    w.close()

    rp_0 = './criteo/2.train_filled_bin.csv'
    rp_1 = './criteo/1.train_filled.csv'
    wp = './criteo/3.train_done.csv'

    header = 'label,N0,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32,C33,C34,C35,C36,C37,C38\n'

    with open(rp_0, 'r') as r0, open(rp_1, 'r') as r1:
        next(r1)
        w = open(wp, 'w')
        w.write(header)
        counts = 0
        for item in zip(r0, r1):
            counts += 1
            l0 = item[0].replace('\n', '').split(',')
            l1 = item[1].replace('\n', '').split(',')
            w.write(l1[0])
            for i in l0:
                w.write(',' + i)
            for i in l1[14:]:
                w.write(',' + i)
            w.write('\n')
            # print(l1[0], l0, l1[14:])
        print('index:', counts)
        print('03 done is finished !!!')
        w.close()
    r0.close()
    r1.close()

    train_path = './criteo/3.train_done.csv'
    train_ffm = './criteo/criteo/train.bid.all.txt'
    test_ffm = './criteo/criteo/test.bid.all.txt'
    vali_path = './criteo/criteo/validation.csv'
    feature_index = './criteo/criteo/feat_index.txt'
    test_count = 0
    with open(train_ffm, 'w') as outfile, open(test_ffm, 'w') as f1, open(vali_path, 'w') as f2:
        f2.write('id,label' + '\n')
        for e, row in enumerate(DictReader(open(train_path)), start=1):
            features = []
            for k, v in row.items():
                if k in field:
                    if len(v) > 0:
                        idx = field_index(k)
                        kv = k + '_' + v
                        features.append('{}'.format(getIndices(kv)))

            if e % 100000 == 0:
                print(datetime.now(), 'creating train.ffm...', e)
                # break

            n = random.randint(0, 5)
            if n == 0:
                f1.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))
                f2.write(str(test_count) + ',' + row['label'] + '\n')
                test_count += 1
            else:
                outfile.write('{0} {1}\n'.format(row['label'], ' '.join('{0}'.format(val) for val in features)))

    fo = open(feature_index, 'w')
    fo.write(str(len(table)))
    fo.close()
