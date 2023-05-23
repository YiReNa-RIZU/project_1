import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as pls

pls.use('ggplot')

x = np.array([0.0, 0.3, 0.7])
nus_wide = np.array([0.8706, 0.8613, 0.8737])
cifar10 = np.array([0.8586, 0.8550, 0.8495, 0.8301, 0.8380, 0.8067])
nus_wide_down = np.array([0.6270, 0.7463, 0.7521])
cifar10_down = np.array([0.9104, 0.9044, 0.8981, 0.8870, 0.8751, 0.8689])
flickr25k_supervised_down = np.array([0.8414, 0.8573, 0.8566])


y = nus_wide

photo1 = plt.subplot(111)
photo1.plot(x, y, 'r-s')
photo1.set_xticks(np.linspace(0, 1, 11))
photo1.set_title('map of nuswide_supervised_all')
photo1.set_xlabel('ita parameter')
photo1.set_ylabel('mAP')

for a, b in zip(x, y):
    photo1.text(a, b, b, weight='light', verticalalignment='top', horizontalalignment='right', fontsize=15)

plt.savefig('nuswide_supervised_all_map.jpg')
plt.show()
