import numpy as np
import sys
import matplotlib.pyplot as plt

UR = 0             # right state. EXPRESSIONS BELOW HARDCODE UR=0 FOR NOW.
A = -2.5           # interval left
B = 2.5            # interval right
LAMBDA = 2         # amplitude of random jump
SEED = 112358      # numpy.random seed for reproducibility

def random_left_state():
    return UR + LAMBDA*np.random.rand()


def make_init_state(x, t, cellavgs, uL):
    jump = 0.5*(uL+UR)*t
    for i in range(len(cellavgs)):
        xL = x[i]
        xR = x[i+1]
        dx = xR - xL
        if xR <= jump:
            cellavgs[i] = uL
        elif xL >= jump:
            cellavgs[i] = UR
        else:
            cellavgs[i] = ((jump-xL)*uL + (xR-jump)*UR)/dx
    return

# def Xi(x,t):
#     xi = 2*x/t
#     if xi <= 0:
#         return 0
#     elif xi <= LAMBDA:
#         return xi/LAMBDA
#     else:
#         return 1

def exact_mean(x,t):
    return (x<=0)*(LAMBDA/2) + (x>0)*(2*x/t < LAMBDA)*(0.5*LAMBDA - 2*x*x/LAMBDA/t/t)

def exact_mean_squared(x,t):
    L3 = LAMBDA*LAMBDA*LAMBDA
    return (x<=0)*(LAMBDA*LAMBDA/3) + (x>0)*(2*x/t < LAMBDA)*(L3 - 8*x*x*x/t/t/t)/(3*LAMBDA)

N = 200
t = 1
M = 1000
np.random.seed(SEED)

x = np.linspace(A,B,N+1)
xmid = (x[1:]+x[:-1])/2
realization_fv = np.zeros(N)
mean = np.zeros(N)
M2 = np.zeros(N)
delta = np.zeros(N)
for m in range(M):
    make_init_state(x, t, realization_fv, random_left_state())
    delta = realization_fv - mean
    mean += delta/(m+1)
    delta2 = realization_fv - mean
    M2 += delta*delta2
M2 /= (M-1)
plt.plot(xmid, exact_mean(xmid, t), 'b--')
plt.plot(xmid, mean, 'b')
plt.plot(xmid, exact_mean_squared(xmid,t) - exact_mean(xmid,t)**2, 'r--')
plt.plot(xmid, M2, 'r')
plt.show()
