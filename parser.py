import numpy as np

def parse(iteration):
    f = open("bort_net", 'r')
    # 627 iterations count, 5 number of charts, 60 max number of
    # points, 2 point`s dimension
    arrs = np.zeros((5, 2, 60), dtype='float')
    content = f.read()
    arrays = content.split("\t\n")
    for index_array, arr in enumerate(arrays):
        raw_points = arr.split(" )	( ")
        for index_point, point in enumerate(raw_points):
            xy = point.split(", ")
            if index_array == 4:
                if index_point == 0:
                    arrs[index_array][0][index_point + 50] = 490
                    arrs[index_array][1][index_point + 50] = float(xy[1].replace(",", "."))
                else:
                    arrs[index_array][0][index_point + 50] = float(xy[0])
                    arrs[index_array][1][index_point + 50] = float(xy[1].replace(",", ".").replace(" )", ""))
            else:
                if index_point == 0:
                    arrs[index_array][0][index_point] = 0
                    arrs[index_array][1][index_point] = float(xy[1].replace(",", "."))
                else:
                    arrs[index_array][0][index_point] = float(xy[0])
                    arrs[index_array][1][index_point] = float(xy[1].replace(",", ".").replace(" )", ""))
            print(arrs)
    return arrs