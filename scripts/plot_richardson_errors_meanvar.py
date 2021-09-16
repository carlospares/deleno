import matplotlib.pyplot as plt
import numpy as np

# I ran Richardson extrapolation, using ground truth 1024x1024, for smooth and discontinuous
# shear layer, mean and variance. Each experiment I ran computing the extrapolation using
# resolutions in the triplets (64, 128, 256) and (128, 256, 512) as input.

smooth_mean_RE = np.array([0.81, 0.36])
smooth_var_RE = np.array([9.17, 8.33])
discont_mean_RE = np.array([1.67, 0.97])
discont_var_RE = np.array([11.4, 7.97])

smooth_mean_RE_w5 = np.array([0.48, 0.25])
smooth_var_RE_w5 = np.array([9.12, 8.32])
discont_mean_RE_w5 = np.array([1.58, 0.95])
discont_var_RE_w5 = np.array([11.36, 7.96])

smooth_mean_data = np.array([4.13, 1.99, 0.9, 0.37])
smooth_var_data = np.array([17.46, 13.65, 9.17, 8.33])
discont_mean_data = np.array([4.33, 2.39, 1.67, 0.97])
discont_var_data = np.array([34.19, 22.29, 11.65, 7.97])

smooth_mean_data_w5 = np.array([4.08, 1.92, 0.82, 0.3])
smooth_var_data_w5 = np.array([17.17, 13.49, 9.12, 8.32])
discont_mean_data_w5 = np.array([4.1, 2.21, 1.58, 0.95])
discont_var_data_w5 = np.array([34.05, 22.21, 11.6, 7.96])

N = [256, 512]
Nplus = [64,128,256,512]


# plt.loglog(Nplus, smooth_mean_data, '.', linestyle='solid', c='red', label="smooth, mean (d)", base=2)
# plt.loglog(Nplus, smooth_mean_data_w5, '.', linestyle='dashed', c='red', label="smooth, mean (d+W5)", base=2)
# plt.loglog(N, smooth_mean_RE, '.', linestyle='dotted', c='red', label="smooth, mean (RE)", base=2)
# plt.loglog(N, smooth_mean_RE_w5, '.', linestyle='dashdot', c='red', label="smooth, mean (RE+W5)", base=2)

# plt.loglog(Nplus, smooth_var_data, '.', linestyle='solid', c='green', label="smooth, var (d)", base=2)
# plt.loglog(Nplus, smooth_var_data_w5, '.', linestyle='dashed', c='green', label="smooth, var (d+W5)", base=2)
# plt.loglog(N, smooth_var_RE, '.', linestyle='dotted', c='green', label="smooth, var (RE)", base=2)
# plt.loglog(N, smooth_var_RE_w5, '.', linestyle='dashdot', c='green', label="smooth, var (RE+W5)", base=2)

plt.loglog(Nplus, discont_mean_data, '.', linestyle='solid', c='blue', label="discont., mean (d)", base=2)
plt.loglog(Nplus, discont_mean_data_w5, '.', linestyle='dashed', c='blue', label="discont., mean (d+W5)", base=2)
plt.loglog(N, discont_mean_RE, '.', linestyle='dotted', c='blue', label="discont., mean (RE)", base=2)
plt.loglog(N, discont_mean_RE_w5, '.', linestyle='dashdot', c='blue', label="discont., mean (RE+W5)", base=2)

plt.loglog(Nplus, discont_var_data, '.', linestyle='solid', c='black', label="discont., var (d)", base=2)
plt.loglog(Nplus, discont_var_data_w5, '.', linestyle='dashed', c='black', label="discont., var (d+W5)", base=2)
plt.loglog(N, discont_var_RE, '.', linestyle='dotted', c='black', label="discont., var (RE)", base=2)
plt.loglog(N, discont_var_RE_w5, '.', linestyle='dashdot', c='black', label="discont., var (RE+W5)", base=2)



plt.legend()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--', linewidth='0.2')
plt.xlabel("N")
plt.ylabel("L1 error (% of GT)")
plt.title("Rich. extr. based on N/4, N/2, N. Discontinous SL, stats.")

plt.show()
