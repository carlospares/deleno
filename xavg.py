import data_handler as dh
import sys
import numpy as np
import matplotlib.pyplot as plt

XAVG_PREFIX = "xavg_"

# Average input file along axis 0, output to npy file prefixed with XAVG_PREFIX
if len(sys.argv) != 2:
    print("Usage: python xavg.py filename")
    sys.exit()

data = dh.load_data(sys.argv[1])
print(data.shape)
avgs = np.average(data, 2)

out = np.zeros(data.shape)
for c in range(data.shape[0]):
    for i in range(data.shape[1]):
        out[c, :, i] = avgs[c,:]

# plt.subplot(221)
# plt.contourf(data[0])
# plt.colorbar()
# plt.subplot(222)
# plt.contourf(data[1])
# plt.colorbar()
# plt.subplot(223)
# plt.contourf(out[0])
# plt.colorbar()
# plt.subplot(224)
# plt.contourf(out[1])
# plt.colorbar()
# plt.show()

out = np.reshape(out, (data.shape[0],data.shape[1]*data.shape[1]))

idx = sys.argv[1].rfind("/")
if idx == -1:
    outname = XAVG_PREFIX+sys.argv[1]
else:
    outname = sys.argv[1][:idx+1] + XAVG_PREFIX + sys.argv[1][idx+1:]
print(outname)
print(out.shape)
np.save(outname, out)
