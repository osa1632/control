import numpy as np
import matplotlib.pyplot as plt

from ExternalParams import use_differentiator, x0, alpha, L, use_secondOrThird, simulateSignalThanDiffrientiator, \
    use_filter, add_noise
from Utils import f, U_2, U_3, dv, d2v, v, D2, v_c, dv_c, d2v_c

t0 = 0
t1 = 10.

t = t0
x = x0
dt = 1e-5

u = 0
xs = []
ts = []
v_cs = []
us = []
us_dot = []

if simulateSignalThanDiffrientiator:
    sigma = v(t,x) - v_c(t)
    z = np.array([sigma, 0, 0])
    if use_filter:
        z_f = np.array([0., 0., 0.])
    else:
        z_f = np.array([])

    while t < t1:
        dx = f(t, x, u)
        sigma = v(t, x) - v_c(t)
        if add_noise:
            noise_gain = 2.
            sigma+= noise_gain*(np.random.normal(np.sin(10000*t), noise_gain))

        if use_differentiator:
            [dz_f, dz] = D2(L, z_f, z, sigma, use_filter)
        else:
            sigma_dot = dv(t, x, dx) - dv_c(t, u)
            sigma_dotaiim = d2v(t, x, dx) - d2v_c(t, u)
            z = np.array([sigma, sigma_dot, sigma_dotaiim])

        if use_secondOrThird:
            u = alpha * U_3(z)
        else:
            u = alpha*U_2(z)

        if use_differentiator:
            z_f += dz_f * dt
            z += dz * dt

        x += dx * dt
        t += dt

        ts.append(t)
        xs.append(x[0])
        us.append(z.copy())

    plt.subplot(211)
    plt.plot(ts, xs, 'r')
    plt.plot(ts, v_c(np.array(ts)), 'g--')
    plt.title(f'$x_1\leftrightarrow v_c$ {"$3^{rd}$" if use_secondOrThird else "$2^{nd}$"} Order'
              f' With{"" if use_differentiator else "out"} Diffrentiator '
              f'{"and filter" if use_filter else ""}')
    plt.grid()
    for ii in range(3):
        plt.subplot(2,3,4+ii)
        plt.plot(ts, [_[ii] for _ in us])
        title='$\\'
        if ii == 2:
            title += 'd'
        if ii >= 1:
            title += 'dot\\'
        title+='sigma$'
        plt.title(title)
        plt.grid()
    # plt.subplot(224)
    # plt.plot(ts, us_dot)
    plt.show()

else:
    sigma = v_c(t)

    z = np.array([sigma, 0, 0])
    z_f = np.array([0., 0.])

    while t < t1:
        sigma = v_c(t)
        [dz_f, dz] = D2(L, z_f, z, sigma)

        ts.append(t)
        xs.append(dz)
        us.append(dv_c(t, u))
        v_cs.append(d2v_c(t, u))

        z_f += dz_f * dt
        z += dz * dt
        t += dt

    plt.subplot(121)
    plt.plot(ts, [_[0] for _ in xs], 'r')
    plt.plot(ts, us, 'g--')
    plt.subplot(122)
    plt.plot(ts, [_[1] for _ in xs], 'r.')
    plt.plot(ts, v_cs, 'g--')
    plt.show()