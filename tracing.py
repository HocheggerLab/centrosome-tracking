import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode


def f(s, b, B, Tz):
    return [Tz / B * np.sin(b[1]), b[0], np.sin(b[1]), np.cos(b[1])]


def onclick(event):
    mevent = event.mouseevent
    print('pickevent: x=%d, y=%d, xdata=%f, ydata=%f' %
          (mevent.x, mevent.y, mevent.xdata, mevent.ydata))


class heavy_planar_elastica():
    def __init__(self):
        self.L = 10.0
        self.B = 1.0
        self.Tz = 5.0
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        # self.ax1.xlabel('s')
        # self.ax1.ylabel('b')
        # self.ax2.xlabel('x')
        # self.ax2.ylabel('y')

    def update_ode(self):
        self.r = ode(f).set_integrator('dop853', method='bdf')
        self.r.set_initial_value([0, self.L,0,0], 0).set_f_params(self.B, self.Tz)

    def plot(self):
        dt = .01
        r = self.r
        b, s, x, y = [], [], [], []
        while r.successful() and r.t < self.L:
            r.integrate(r.t + dt)
            # print('%g %s' % (r.t, r.y))
            b.append(r.t)
            s.append(r.y[1])
            x.append(r.y[2])
            y.append(r.y[3])
        self.ax1.plot(b, s)
        self.ax2.plot(x, y)
        self.ax2.plot(x[-1], y[-1], 'ro', picker=5)  # 5 points tolerance
        self.fig.canvas.mpl_connect('pick_event', onclick)


if __name__ == '__main__':
    fiber = heavy_planar_elastica()
    fiber.update_ode()
    fiber.plot()

    plt.show()
