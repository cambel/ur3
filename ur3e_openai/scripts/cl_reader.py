import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def read_cl_from_file(filename):
    data = []
    fo = open(filename, "r")
    lines = fo.readlines()
    for line in lines:
        if "difficulty:" in line:
            d = np.array(line.split(':')[4].split(',')[:2], dtype=float)
            data.append(d)
            # print(d)
    return np.array(data)
    
def plot_cl(filename):
    data = read_cl_from_file(filename)
    x = np.linspace(0, len(data), len(data))
    plt.plot(x, data[:,0])

filename = '/home/cambel/dev/results/20220202T054738.787818_SAC_pih_m24/console.log'
plot_cl(filename)
filename = '/home/cambel/dev/results/20220203T003145.002918_SAC_pih_m24/console.log'
plot_cl(filename)
filename = '/home/cambel/dev/results/20220202T034345.787792_SAC_pih_m24/console.log'
plot_cl(filename)
filename = '/home/cambel/dev/results/20220203T063559.350251_SAC_pih_m24/console.log'
plot_cl(filename)

plt.grid()
plt.show()