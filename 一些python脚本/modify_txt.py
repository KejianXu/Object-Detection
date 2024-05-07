import codecs
import os

path = '../yolov5-6.2.0/runs/detect/exp2/labels/'  # 标签文件train路径
m = os.listdir(path)
# 读取路径下的txt文件
for n in range(0, len(m)):
    t = codecs.open('../yolov5-6.2.0/runs/detect/exp2/labels/' + m[n], mode='r', encoding='utf-8')
    line = t.readline()  # 以行的形式进行读取文件
    list1 = []
    while line:
        a = line.split()
        list1.append(a)
        line = t.readline()
    t.close()

    lt = open('../yolov5-6.2.0/runs/detect/exp2/labels/' + m[n], "w")
    for num in range(0, len(list1)):
        if list1[num][0] == str(96):  # 判断将txt文件中第一列是否为0
            list1[num][0] = str(0)  # 第一列为0时，将0改为1
        lt.writelines(' '.join(list1[num]) + '\n')  # 每个元素以空格间隔，一行元素写完并换行
    lt.close()
    print(m[n] + " 修改完成")