import matplotlib.pyplot as plt
# 设置字体格式
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.interpolate import make_interp_spline


size = 14

# 设置英文字体
config = {'family': 'Palatino Linotype',
          'weight': 'bold',
          'size': 13,
}
# rcParams.update(config)
# 设置中文宋体
fontcn = {'family': 'SimSun', 'size': 14, 'weight': 'bold'}
label_size = size
text_size = size

# 数据1
training2way_AP = [30.0, 31.0, 35.0, 38.0, 44.0, 43.0, 46.0,
                   46.5, 46.2, 46.7, 46.5, 48.3,
                   49.1, 47.9, 49.5, 49.3, 49.6, 49.3, 50.0, 50.3, 50.2, 50.4,
                   50.3, 50.3, 50.3, 50.4, 50.4, 50.4, 50.3, 50.4, 50.5, 50.5]
training2way_epoch = [3, 3.2, 4, 5, 14, 20, 30,
                     34, 36, 38, 40, 45.5,
                     48, 51, 54, 56, 58, 60, 63, 64, 66, 68,
                     70, 71, 72, 75, 77, 79, 80, 84, 85, 87]

m = make_interp_spline(training2way_epoch, training2way_AP)
training2way_epoch = np.linspace(3, 87, 150)
training2way_AP = m(training2way_epoch)
# # 数据2

training4way_AP = [29.0, 33.0, 37.0, 40.0, 41.0, 44.0, 46.0, 47.0, 48.0, 49.0,
                   50.0, 49.8, 49.6, 49.5, 50.8, 50.5, 51.5, 50.8, 51.7, 51.5,
                   52.0, 51.8, 52.6, 52.4, 52.5, 52.7, 52.8, 52.8, 52.7, 52.8,
                   52.7, 52.8]
training4way_epoch = [2.5, 3, 4, 5.5, 6, 7, 8, 9, 10, 14,
                     17, 20, 21, 23, 25, 27, 29, 33, 36,
                     38, 40, 41, 43, 45, 46, 49, 52, 55, 58, 62,
                     65, 70]

m = make_interp_spline(training4way_epoch, training4way_AP)
training4way_epoch = np.linspace(2.5, 70, 100)
training4way_AP = m(training4way_epoch)


# 参数设置
lw = 2
ms = 7
my_text = ['S', 'M', 'L', 'X']
# 绘制 mAP-Param
plt.figure(figsize=(5.5, 7))
plt.tick_params(axis='x', colors='white', direction='in', labelcolor='black')
plt.tick_params(axis='y', colors='white', direction='in', labelcolor='black')

plt.plot(training2way_epoch, training2way_AP, label='YOLOX',
         c='limegreen',
         lw=lw,
         markersize=10,
         ls='-')
plt.plot(training4way_epoch, training4way_AP, label='YOLOAX',
         c='r',
         lw=lw,
         markersize=10,
         ls='-')

plt.legend(loc='lower right', prop={'size':12})

plt.xlabel('epoch', fontdict={'family': 'Palatino Linotype', 'size': 16, 'color': 'black'})
plt.ylabel('AP', fontdict={'family': 'Palatino Linotype', 'size': 16, 'color': 'black'})
ax = plt.gca()
ax.patch.set_facecolor("gainsboro")
plt.xlim((0, 100))
plt.ylim((25, 55))
x_ticks = np.linspace(0, 100, 6)
y_ticks = np.linspace(25, 55, 7)
plt.xticks(x_ticks, fontsize=12)
plt.yticks(y_ticks, fontsize=12)
plt.grid(linestyle='-', color='w')
plt.show()
