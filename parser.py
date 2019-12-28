import numpy as np


def parse_row(row):
    a = np.array(list(map(lambda x: np.array([el.replace(' ', '').replace(',', '.') for el
                                              in x.replace('(', '').replace(')', '').strip().split(', ')]),
                          row.split('\t'))))

    x = [float(el[0]) for el in a if el[0]]
    y = [float(el[1]) for el in a if len(el) > 1 and el[1]]

    return [x, y]


def parse(filename):
    file = open(filename, 'r', encoding='utf-16')
    data = file.readlines()
    data = [el for el in data if el.find('iteration') == -1]

    res_data = list(map(lambda x: parse_row(x), data))
    all_dt = [res_data[i: i + 6] for i in range(0, len(res_data), 6)]

    for iter, dt in enumerate(all_dt):
        for d in dt[0:5]:
            d[0] = [i + iter * 20 for i in d[0]]

    return [el[:-1] + [el[-1][0]] for el in all_dt]

df = parse('Data/2,2,2,T(x),norm,10,Reanim/Graphics0.txt')

print(len(df[500][1]))
