# Import libraries
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import time
import scipy.io
import pandas as pd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from DeepFlow import *
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(123456)
np.random.seed(123456)
device = torch.device('cuda:0')


# Generate Neural Network
class DNN(pl.LightningModule):

    def __init__(self):
        super(DNN, self).__init__()
        self.net = nn.Sequential()  # Define neural network
        self.net.add_module('Linear_layer_1', nn.Linear(2, 30))  # First linear layer
        self.net.add_module('Tanh_layer_1', nn.Tanh())  # First activation Layer

        for num in range(2, 7):  # Number of layers (2 through 7)
            self.net.add_module('Linear_layer_%d' % (num), nn.Linear(30, 30))  # Linear layer
            self.net.add_module('Tanh_layer_%d' % (num), nn.Tanh())  # Activation Layer
        self.net.add_module('Linear_layer_final', nn.Linear(30, 4))  # Output Layer

    # Forward Feed
    def forward(self, x):
        return self.net(x)

    # Loss function for PDE
    def loss_FVM(self, x, q, nx, ny, idf):
        y = self.net(x.double())

        q_1 = torch.tensor(q[1:ny - 1, 1:nx - 1, 0].flatten()[:, None], dtype=torch.float64).to(device)
        q_2 = torch.tensor(q[1:ny - 1, 1:nx - 1, 1].flatten()[:, None], dtype=torch.float64).to(device)
        q_3 = torch.tensor(q[1:ny - 1, 1:nx - 1, 2].flatten()[:, None], dtype=torch.float64).to(device)
        q_4 = torch.tensor(q[1:ny - 1, 1:nx - 1, 3].flatten()[:, None], dtype=torch.float64).to(device)

        f = (((y[:, 0:1] - q_1)) ** 2).mean() + (((y[:, 1:2] - q_2)) ** 2).mean() + (
                    ((y[:, 2:3] - q_3)) ** 2).mean() + (((y[:, 3:] - q_4)) ** 2).mean()
        return f

# Convert torch tensor into np.array
def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))


def Euler_IC(x, y, IC):
    # Different Test Cases
    if IC == 1:
        p = [1.0, 0.4, 0.0439, 0.15];
        r = [1.0, 0.5197, 0.1072, 0.2579];
        u = [0.0, -0.7259, -0.7259, 0.0];
        v = [0.0, -0.0, -1.4045, -1.4045];

    if IC == 2:
        p = [1.0, 0.4, 1.0, 0.4];
        r = [1.0, 0.5197, 1.0, 0.5197];
        u = [0.0, -0.7259, -0.7259, 0.0];
        v = [0.0, 0.0, -0.7259, -0.7259];

    if IC == 3:
        p = [1.5, 0.3, 0.029, 0.3];
        r = [1.5, 0.5323, 0.138, 0.5323];
        u = [0.0, 1.206, 1.206, 0.0];
        v = [0.0, 0.0, 1.206, 1.206];

    if IC == 4:
        p = [1.1, 0.35, 1.1, 0.35];
        r = [1.1, 0.5065, 1.1, 0.5065];
        u = [0.0, 0.8939, 0.8939, 0.0];
        v = [0.0, 0.0, 0.8939, 0.8939];

    if IC == 5:
        p = [1.0, 1.0, 1.0, 1.0];
        r = [1.0, 2.0, 1.0, 3.0];
        u = [-0.75, -0.75, 0.75, 0.75];
        v = [-0.5, 0.5, 0.5, -0.5];

    r0, u0, v0, p0 = np.zeros((len(y), len(x))), np.zeros((len(y), len(x))), np.zeros((len(y), len(x))), np.zeros(
        (len(y), len(x)))

    reg1 = np.where((x >= 0.5) & (0.5 <= y))
    reg2 = np.where((x < 0.5) & (0.5 <= y))
    reg3 = np.where((x < 0.5) & (y < 0.5))
    reg4 = np.where((x >= 0.5) & (y <= 0.5))

    r0[reg1], r0[reg2], r0[reg3], r0[reg4] = r[0], r[1], r[2], r[3]
    u0[reg1], u0[reg2], u0[reg3], u0[reg4] = u[0], u[1], u[2], u[3]
    v0[reg1], v0[reg2], v0[reg3], v0[reg4] = v[0], v[1], v[2], v[3]
    p0[reg1], p0[reg2], p0[reg3], p0[reg4] = p[0], p[1], p[2], p[3]

    return r0, u0, v0, p0


lr = 0.0005
CFL = 0.5  # CFL number
tEnd = 0.3  # Final time
nx = 240  # Number of cells/Elements in x
ny = 240  # Number of cells/Elements in y
n = 5  # Degrees of freedom: ideal air=5, monoatomic gas=3
IC = 3
fluxMth = 'HLLE'
limiter = 'MC'

gamma = 1.4

Lx = 1
dx = Lx / nx
xc = np.arange(dx / 2, Lx, dx)

Ly = 1
dy = Ly / ny
yc = np.arange(dy / 2, Ly, dy)

x, y = np.meshgrid(xc, yc)

r0, u0, v0, p0 = Euler_IC(x, y, IC)

E0 = p0 / ((gamma - 1) * r0) + 0.5 * (u0 ** 2 + v0 ** 2)
c0 = np.sqrt(gamma * p0 / r0)
Q0 = np.zeros((ny, nx, 4))

Q0[:, :, 0] = r0
Q0[:, :, 1] = r0 * u0
Q0[:, :, 2] = r0 * v0
Q0[:, :, 3] = r0 * E0

nx += 2
ny += 2

q0 = np.zeros((ny, nx, 4))
q0[1:ny - 1, 1:nx - 1, 0:4] = Q0

q0[:, 0, :] = q0[:, 1, :]
q0[:, nx - 1, :] = q0[:, nx - 2, :]

q0[0, :, :] = q0[1, :, :]
q0[ny - 1, :, :] = q0[ny - 1, :, :]

vn = np.sqrt(u0 ** 2 + v0 ** 2)
lambda1 = vn + c0
lambda2 = vn - c0
a0 = np.max(np.abs([lambda1, lambda2]))
dt0 = 0.0011

q = q0
t = dt0
it = 0
dt = 0.0011
a = a0

Y = y.flatten()[:, None]  # Vectorized t_grid
X = x.flatten()[:, None]  # Vectorized x_grid

# Sample grid (Batch training)
num_f_train = 5000
id_f = np.random.choice((nx - 2) * (ny - 2), num_f_train, replace=False)
x_test = np.hstack((Y, X))  # Vectorized whole domain

X_int = X[:, 0][:, None]  # Random x - interior
Y_int = Y[:, 0][:, None]  # Random t - interior

x_int_train = np.hstack((Y_int, X_int))  # Random (x,t) - vectorized

x_int_train = torch.tensor(x_int_train, requires_grad=True, dtype=torch.float64).to(device)
x_test = torch.tensor(x_test, requires_grad=True, dtype=torch.float64).to(device)

with plt.style.context(['science','ieee']):
    fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(nrows=2, ncols=2)
    ax1.contourf(x, y, r0,100, cmap='jet',extend='both')
    ax2.contourf(x, y, p0,100,cmap='jet',extend='both')
    ax3.contourf(x, y, u0,100,cmap='jet',extend='both')
    ax4.contourf(x, y, v0,100,cmap='jet',extend='both')


    ax1.set_xlabel(r'$x(m)$', fontsize=4)
    ax1.set_ylabel(r'$y(m)$', fontsize=4)
    ax1.set_title(r'$\rho(x,y,0)$', fontsize=7)

    ax2.set_xlabel(r'$x(m)$', fontsize=4)
    ax2.set_ylabel(r'$y(m)$', fontsize=4)
    ax2.set_title(r'$p(x,y,0)$', fontsize=7)

    ax3.set_xlabel(r'$x(m)$', fontsize=4)
    ax3.set_ylabel(r'$y(m)$', fontsize=4)
    ax3.set_title('$u(x,y,0)$', fontsize=7)

    ax4.set_xlabel(r'$x(m)$', fontsize=4)
    ax4.set_ylabel(r'$y(m)$', fontsize=4)
    ax4.set_title(r'$v(x,y,0)$', fontsize=7)
plt.tight_layout()

parameterize = True
tic_total = time.time()
while (t < tEnd):
    DF = autodiff()
    DF_nn = DF.MUSCL_Euler_2D(q, a, gamma, dx, dy, nx, ny, limiter, fluxMth)
    if (it % 2 == 0 or t == 0.3):

        print(
            '--------------------------------------------------------------------------------------------------------------------')
        print('Parametrize solve for t = %f' % t)
        print('Time iteration: %d' % it)
        if (parameterize == True):

            model = DNN().to(device).double()
            # Loss and optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)


            # Train PINNs
            def train(epoch):
                model.train()

                def closure():
                    optimizer.zero_grad()
                    loss_FVM = model.loss_FVM(x_int_train,  q - dt * DF_nn, nx, ny, id_f)
                    loss = loss_FVM
                    loss.backward()
                    return loss

                # Optimize loss function
                loss = optimizer.step(closure)
                loss_value = loss.item() if not isinstance(loss, float) else loss

                # Print total loss
                if epoch % 10000 == 0:
                     print(f'epoch {epoch}: loss {loss_value:.8f}')
                return loss_value


            tic = time.time()
            epoch = 0
            loss_value = np.inf

            if it < 70:
                while loss_value > 0.0000009:
                    loss_value = train(epoch)
                    epoch += 1
            else:
                while loss_value > 0.000009:
                    loss_value = train(epoch)
                    epoch += 1

            toc = time.time()
            total_time = toc - tic
            print('Total training time: %f' % total_time)

            q_nn = to_numpy(model(x_int_train))
            q_nn = np.reshape(q_nn.to_numpy(), (ny - 2, nx - 2, 4))

            q = np.zeros((ny, nx, 4))
            q_nn = pd.DataFrame(q_nn).interpolate()
            q[1:ny - 1, 1:nx - 1, :] = q_nn
            q[:, 0, :] = q[:, 1, :]
            q[:, nx - 1, :] = q[:, nx - 2, :]

            q[0, :, :] = q[1, :, :]
            q[ny - 1, :, :] = q[ny - 2, :, :]

            r = q[:, :, 0]
            u = q[:, :, 1] / r
            v = q[:, :, 2] / r
            E = q[:, :, 3] / r
            p = (gamma - 1) * r * (E - 0.5 * (u ** 2 + v ** 2))
            c = np.sqrt(np.abs(gamma * p / r))

            vn = np.sqrt(np.abs(u ** 2 + v ** 2));
            lambda1 = vn + c
            lambda2 = vn - c
            a = np.max(np.abs([lambda1, lambda2]))

            t = t + dt
            it = it + 1
    else:
        print(
            '--------------------------------------------------------------------------------------------------------------------')
        print('Predictor solve for t = %f' % t)
        print('Time iteration: %d' % it)
        f = q - dt * DF_nn
        q = f
        q[:, 0, :] = q[:, 1, :]
        q[:, nx - 1, :] = q[:, nx - 2, :]

        q[0, :, :] = q[1, :, :]
        q[ny - 1, :, :] = q[ny - 2, :, :]

        r = q[:, :, 0]
        u = q[:, :, 1] / r
        v = q[:, :, 2] / r
        E = q[:, :, 3] / r
        p = (gamma - 1) * r * (E - 0.5 * (u ** 2 + v ** 2))
        c = np.sqrt(np.abs(gamma * p / r))
        vn = np.sqrt(np.abs(u ** 2 + v ** 2));
        lambda1 = vn + c
        lambda2 = vn - c
        a = np.max(np.abs([lambda1, lambda2]))
        t = t + dt
        it = it + 1
tic_total_final = time.time() - tic_total
print('Total training time: ', tic_total_final)
print(
    '--------------------------------------------------------------------------------------------------------------------')
q = q[1:ny - 1, 1:nx - 1, 0:4]
r = q[:, :, 0]
u = q[:, :, 1] / r
v = q[:, :, 2] / r
E = q[:, :, 3] / r
p = (gamma - 1) * r * (E - 0.5 * (u ** 2 + v ** 2))

with plt.style.context(['science','ieee']):
    fig, ((ax1, ax2), (ax3, ax4))= plt.subplots(nrows=2, ncols=2)
    ax1.contourf(x, y, r, 100,cmap='jet')
    ax2.contourf(x, y, p, 100,cmap='jet',extend='both')
    ax3.contourf(x, y, u, 100,cmap='jet',extend='both')
    ax4.contourf(x, y, v, 100,cmap='jet',extend='both')


    ax1.set_xlabel(r'$x(m)$', fontsize=4)
    ax1.set_ylabel(r'$y(m)$', fontsize=4)
    ax1.set_title(r'$\rho(x,y,0.3)$', fontsize=5)

    ax2.set_xlabel(r'$x(m)$', fontsize=4)
    ax2.set_ylabel(r'$y(m)$', fontsize=4)
    ax2.set_title(r'$p(x,y,0.3)$', fontsize=5)

    ax3.set_xlabel(r'$x(m)$', fontsize=4)
    ax3.set_ylabel(r'$y(m)$', fontsize=4)
    ax3.set_title(r'$u(x,y,0.3)$', fontsize=5)

    ax4.set_xlabel(r'$x(m)$', fontsize=4)
    ax4.set_ylabel(r'$y(m)$', fontsize=4)
    ax4.set_title(r'$v(x,y,0.3)$', fontsize=5)
fig.suptitle(r'FV-PINNs - 2D Shocktube (Liska, 2003)', fontsize=7)
plt.tight_layout()
plt.savefig('2-D.png')
