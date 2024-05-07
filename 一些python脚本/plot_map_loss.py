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
training2way_Loss = [0.58, 0.52, 0.45, 0.40, 0.38, 0.37, 0.36, 0.358, 0.355]
training2way_epoch = [1, 3.5, 5, 7, 8, 9, 11, 12, 13.5]

# m = make_interp_spline(training2way_epoch, training2way_AP)
# training2way_epoch = np.linspace(1, 11.5, 50)
# training2way_AP = m(training2way_epoch)

# 数据2
training4way_Loss = [0.57, 0.50, 0.47, 0.43, 0.38, 0.36, 0.34, 0.338, 0.336]
training4way_epoch = [1, 3.5, 3.8, 4.5, 7, 8, 10, 12, 13.5]

# m = make_interp_spline(training4way_epoch, training4way_AP)
# training4way_epoch = np.linspace(1, 11.5, 50)
# training4way_AP = m(training4way_epoch)


# 参数设置
lw = 2
ms = 7
my_text = ['S', 'M', 'L', 'X']
# 绘制 mAP-Param
plt.figure(figsize=(5.5, 7))
plt.tick_params(axis='x', colors='white', direction='in', labelcolor='black')
plt.tick_params(axis='y', colors='white', direction='in', labelcolor='black')

plt.plot(training2way_epoch, training2way_Loss, label='YOLOX',
         c='limegreen',
         lw=lw,
         markersize=10,
         ls='-')
plt.plot(training4way_epoch, training4way_Loss, label='YOLOAX',
         c='r',
         lw=lw,
         markersize=10,
         ls='-')

plt.legend(loc='upper right', prop={'size':12})

plt.xlabel('epoch', fontdict={'family': 'Palatino Linotype', 'size': 16, 'color': 'black'})
plt.ylabel('Loss', fontdict={'family': 'Palatino Linotype', 'size': 16, 'color': 'black'})
ax = plt.gca()
ax.patch.set_facecolor("gainsboro")
plt.xlim((0, 12))
plt.ylim((0.30, 0.60))
x_ticks = np.linspace(0, 14, 8)
y_ticks = np.linspace(0.30, 0.60, 7)
plt.xticks(x_ticks, fontsize=12)
plt.yticks(y_ticks, fontsize=12)
plt.grid(linestyle='-', color='w')
plt.show()
