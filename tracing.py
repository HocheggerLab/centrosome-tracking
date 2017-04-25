import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_bvp

np.set_printoptions(1)


class heavy_planar_elastica():
    def __init__(self):
        self.L = 3.0
        self.E = 10.0
        self.F = 0.1
        self.gamma = 0.1
        self.Ye = 1

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        self.picking = False
        self.connect()

        self.ax1.set_xlabel('s')
        self.ax1.set_ylabel('M(s)')
        self.ax2.set_xlabel('x [um]')
        self.ax2.set_ylabel('y [um]')

    def connect(self):
        self.cidpress = self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.cidrelease = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fig.figure.canvas.mpl_disconnect(self.cidpress)
        self.fig.figure.canvas.mpl_disconnect(self.cidrelease)
        self.fig.figure.canvas.mpl_disconnect(self.cidmotion)

    def on_pick(self, event):
        self.picking = True
        mevent = event.mouseevent
        # print('pickevent: x=%d, y=%d, xdata=%f, ydata=%f' % (mevent.x, mevent.y, mevent.xdata, mevent.ydata))
        self.Xe = mevent.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if not self.picking: return
        # print('motionevent: picking=%f, event.xdata=%f, event.ydata=%f' % (self.picking, event.xdata, event.ydata))

        self.Xe = self.x_plot[0]
        self.Ye = event.ydata

        self.update_plot()
        self.fig.canvas.draw()

    def on_release(self, event):
        if not self.picking: return
        self.picking = False
        self.Ye *= -1
        self.update_ode()
        self.update_ode_plot()
        self.update_plot()
        self.fig.canvas.draw()

    def f(self, s, x, m0):
        k = x[3] / self.E
        return [-cos(x[2]), -sin(x[2]), -k, self.F * sin(x[2] + self.gamma)]

    # Implement evaluation of the boundary condition residuals:
    def bc(self, xa, xb, m0):
        # p = m0
        out = np.array([xb[0], xb[1], xb[2], xa[3] - m0, xa[1] - self.Ye])
        # print  np.array(xa), np.array(xb), m0, '->', out
        return out

    def update_ode(self):
        s = np.linspace(0, self.L, 4)
        y_a = np.zeros((4, s.size))

        # Now we are ready to run the solver.
        self.res = solve_bvp(self.f, self.bc, s, y_a, p=[1], verbose=0)
        self.s_plot = np.linspace(0, self.L, 100)
        self.x_plot = self.res.sol(self.s_plot)[0]
        self.y_plot = -self.res.sol(self.s_plot)[1]
        self.M_plot = self.res.sol(self.s_plot)[3]

        self.Xe = self.x_plot[0]
        self.Ye = self.y_plot[0]
        print 'm0=%0.2f' % self.res.p[0]

    def update_ode_plot(self):
        self.ax1.plot(self.s_plot, self.M_plot, label='b_a')
        self.ax2.plot(self.x_plot, self.y_plot, label='%0.1f' % self.res.p[0])
        self.ax2.legend()

    def update_plot(self):
        self.pick_point.set_ydata([self.Ye, self.Ye])
        self.pick_point.set_xdata([self.Xe, self.Xe])
        self.ax2.draw_artist(self.pick_point)

    def initial_plot(self):
        px = (self.Xe, self.Xe)
        py = (self.Ye, self.Ye)
        self.pick_point = plt.Line2D(px, py, marker='.', markersize=10, markerfacecolor='r', picker=5)
        self.ax2.add_line(self.pick_point)

        self.update_ode_plot()
        plt.show()


if __name__ == '__main__':
    fiber = heavy_planar_elastica()
    fiber.update_ode()
    fiber.initial_plot()
