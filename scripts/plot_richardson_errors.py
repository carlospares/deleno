import matplotlib.pyplot as plt
import numpy as np

# I ran Richardson extrapolation, using ground truth 2048x2048, for smooth and discontinuous
# shear layer, at times 0 and 0.4. Each experiment I ran computing the extrapolation using
# resolutions in the triplets (64, 128, 256), (128, 256, 512) and (256, 512, 1024) as input.


# Using trivial FV upscaling
smooth_f0 = np.array([0.79, 0.35, 0.15])
smooth_f1 = np.array([0.81, 0.35, 0.14])
discont_f0 = np.array([1.71, 0.85, 0.44])
discont_f1 = np.array([7.87, 5.05, 5.65])

# Using 5th order WENO FV upscaling
smooth_f0_w5 = np.array([0.22, 0.08, 0.13])
smooth_f1_w5 = np.array([0.12, 0.1, 0.13])
discont_f0_w5 = np.array([1.68, 0.85, 0.43])
discont_f1_w5 = np.array([7.85, 5.04, 5.65])

# Compare data directly
smooth_f0_data = np.array([4.44, 2.07, 0.96, 0.42, 0.15])
smooth_f1_data = np.array([4.23, 2.05, 0.96, 0.42, 0.14])
discont_f0_data = np.array([6.26, 3.39, 1.71, 0.85, 0.44])
discont_f1_data = np.array([10.95, 7.89, 7.87, 5.05, 5.65])

# ...and data directly, upscaled with 5th order WENO
smooth_f0_data_w5 = np.array([4.2, 2.02, 0.94, 0.4, 0.14])
smooth_f1_data_w5 = np.array([4.18, 2.03, 0.95, 0.41, 0.14])
discont_f0_data_w5 = np.array([6.35, 3.38, 1.68, 0.85, 0.43])
discont_f1_data_w5 = np.array([10.73, 7.81, 7.85, 5.04, 5.65])

N = [256, 512, 1024]
Nplus = [64,128,256,512,1024]


# plt.loglog(Nplus, smooth_f0_data, '.', linestyle='solid', c='red', label="smooth, t=0 (d)", base=2)
# plt.loglog(Nplus, smooth_f0_data_w5, '.', linestyle='dashed', c='red', label="smooth, t=0 (d+W5)", base=2)
# plt.loglog(N, smooth_f0, '.', linestyle='dotted', c='red', label="smooth, t=0 (RE)", base=2)
# plt.loglog(N, smooth_f0_w5, '.', linestyle='dashdot', c='red', label="smooth, t=0 (RE+W5)", base=2)

# plt.loglog(Nplus, smooth_f1_data, '.', linestyle='solid', c='green', label="smooth, t=0.4 (d)", base=2)
# plt.loglog(Nplus, smooth_f1_data_w5, '.', linestyle='dashed', c='green', label="smooth, t=0.4 (d+W5)", base=2)
# plt.loglog(N, smooth_f1, '.', linestyle='dotted', c='green', label="smooth, t=0.4 (RE)", base=2)
# plt.loglog(N, smooth_f1_w5, '.', linestyle='dashdot', c='green', label="smooth, t=0.4 (RE+W5)", base=2)

plt.loglog(Nplus, discont_f0_data, '.', linestyle='solid', c='blue', label="discont., t=0 (d)", base=2)
plt.loglog(Nplus, discont_f0_data_w5, '.', linestyle='dashed', c='blue', label="discont., t=0 (d+W5)", base=2)
plt.loglog(N, discont_f0, '.', linestyle='dotted', c='blue', label="discont., t=0 (RE)", base=2)
plt.loglog(N, discont_f0_w5, '.', linestyle='dashdot', c='blue', label="discont., t=0 (RE+W5)", base=2)

plt.loglog(Nplus, discont_f1_data, '.', linestyle='solid', c='black', label="discont., t=0.4 (d)", base=2)
plt.loglog(Nplus, discont_f1_data_w5, '.', linestyle='dashed', c='black', label="discont., t=0.4 (d+W5)", base=2)
plt.loglog(N, discont_f1, '.', linestyle='dotted', c='black', label="discont., t=0.4 (RE)", base=2)
plt.loglog(N, discont_f1_w5, '.', linestyle='dashdot',c='black', label="discont., t=0.4 (RE+W5)", base=2)


plt.legend()
plt.grid(which='major', linestyle='-')
plt.grid(which='minor', linestyle='--', linewidth='0.2')
plt.xlabel("N")
plt.ylabel("L1 error (% of GT)")
plt.title("Rich. extr. based on N/4, N/2, N. Discontinuous SL, sample")

plt.show()
