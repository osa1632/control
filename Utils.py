import numpy as np
from numpy import cos, sin

from ExternalParams import useSecondThanFirstA, useSecondThanFirstU_c


def f(t, x, u):
    x1, x2, x3 = x
    dx1 = 1 + cos(t) - x2
    dx2 = a(x, t) - (3 + 2 * cos(x3)) * u_d(u)
    dx3 = sin(x1 - x2 + x3 - 5 * t) - x3 + 0.5 * u_d(u)
    dx = np.array([dx1, dx2, dx3])
    return dx


def power_sign(a, b):
    return np.abs(a) ** b * np.sign(a)


def U_2(z):
    p = lambda x,x1,*_: power_sign(x1, 2)+x
    return p(*z)/p(*np.abs(z))


def U_3(z):
    p = lambda x,x1,x2: x**3+2*power_sign(x1, 3/2)+x2
    return p(*z)/p(*np.abs(z))

# def U(z, r=3):
#     def p(z, r=r):
#         ret = np.sum(np.apply_along_axis(z**(1/r), lambda power_sign))
#     return p(*z)/p(*np.abs(z))

def dv(t, x, dx):
    return dx[0]


def d2v(t, x, dx):
    return (-1) * sin(t)-dx[1]


def v(t, x):
    return x[0]


def D2(L, z_f, z, f, filter=True):
    if filter:
        w1, w2, w3 = z_f
        z0, z1, z2 = z
        dw1 = -7*L**(1/6)*power_sign(w1, 5/6)+w2
        dw2 = -23.72*L**(2/6)*power_sign(w1, 4/6)+z0 - f
        dw3 = -32.24*L**(3/6)*power_sign(w1, 3/6)+z0 - f

        dz0 = -20.26*L**(4/6)*power_sign(w1, 2/6)+z1
        dz1 = -6.75*L**(5/6)*power_sign(w1, 1/6)+z2
        dz2 = -1.1*L*np.sign(w1)
        dz_f = np.array([dw1, dw2, dw3])
        dz = np.array([dz0,dz1,dz2])
    elif filter==2:
        w1, w2 = z_f
        z0, z1, z2 = z
        dw1 = -5*L**(1/5)*power_sign(w1, 4/5)+w2
        dw2 = -10.03*L**(2/5)*power_sign(w1, 3/5)+z0 - f
        dz0 = -9.30*L**(3/5)*power_sign(w1, 2/5)+z1
        dz1 = -4.75*L**(4/5)*power_sign(w1, 1/5)+z2
        dz2 = -1.1*L*np.sign(w1)
        dz_f = np.array([dw1, dw2])
        dz = np.array([dz0,dz1,dz2])
    else:
        z0, z1, z2 = z
        dz0 = -2*L**(1/3)*power_sign(z0-f, 2/3)+z1
        dz1 = -2.12*L**(2/3)*power_sign(z0-f, 1/3)+z2
        dz2 = -1.1*L*np.sign(z0-f)
        dz_f = np.array([])
        dz = np.array([dz0,dz1,dz2])
    return dz_f, dz


def a(x, t):
    if useSecondThanFirstA:
        a = cos(3 + x[2])
    else:
        a = cos(x[1] - 2 * x[2])
    return a


def u_d(u):
    return u


def v_c(t):
    if useSecondThanFirstU_c:
        v_c = 2 * sin(2 * t + 1) - 1.5 * cos(1.3 * t)
    else:
        v_c = 3 * cos(2 * t) - cos(1.2 * t - 1) + 0.01 * cos(10 * t)
    return v_c


def dv_c(t, u):
    if useSecondThanFirstU_c:
        dv_c = 2 * cos(2 * t + 1) * 2 - (-1) * 1.5 * sin(1.3 * t) * 1.3
    else:
        dv_c = (-1) * 3 * sin(2 * t) * 2 - (-1) * sin(1.2 * t - 1) * 1.2 + (-1) * 0.01 * sin(10 * t) * 10
    return dv_c


def d2v_c(t, u):
    if useSecondThanFirstU_c:
        dv_c = 2 *(-1)* sin(2 * t + 1) * 2*2 - (-1) * 1.5 * cos(1.3 * t) * 1.3*1.3
    else:
        dv_c = (-1) * 3 *cos(2 * t) * 2*2 - (-1) * cos(1.2 * t - 1) * 1.2*1.2 + (-1) * 0.01 * cos(10 * t) * 10*10

    # df_1 = lambda t: (-1) * 3 * np.sin(2 * t) * 2 - (-1) * np.sin(1.2 * t - 1) * 1.2 + (-1) * 0.01 * np.sin(10 * t) * 10
    # df = lambda t: (f(t + .5e-5) - f(t - .5e-5)) / 1e-4
    # plt.plot(t, df_1(t))
    # plt.plot(t, df(t))
    # plt.show()

    # d2f = lambda t: (df(t + 5e-5) - df(t - 5e-5)) / 1e-4
    # df2_ = lambda t: (-1) * 3 * np.cos(2 * t) * 2 * 2 - (-1) * np.cos(1.2 * t - 1) * 1.2 * 1.2 + (-1) * 0.01 * np.cos(
    #     10 * t) * 10 * 10
    # plt.plot(t, d2f(t), '--o')
    # plt.plot(t, df2_(t))
    # plt.show()
    return dv_c