import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp


class heavy_planar_elastica():
    def __init__(self):
        self.L = 10.0
        self.B = 1.0
        self.Tz = 5.0
        self.Xe = 1
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(121)
        plt.xlabel('s')
        plt.ylabel('b')
        self.ax2 = self.fig.add_subplot(122)
        plt.xlabel('x')
        plt.ylabel('y')
        self.picking = False
        self.connect()

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
        print('pickevent: x=%d, y=%d, xdata=%f, ydata=%f' % (mevent.x, mevent.y, mevent.xdata, mevent.ydata))
        self.Xe = mevent.ydata

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if not self.picking: return
        print('motionevent: picking=%f, event.xdata=%f,event.ydata=%f' % (self.picking, event.xdata, event.ydata))
        self.Xe = event.ydata
        self.update_plot()

    def on_release(self, event):
        'on release we reset the press data'
        if not self.picking: return
        self.picking = False
        self.update_ode()
        self.update_ode_plot()

    def f(self, s, b):
        return [self.Tz / self.B * np.sin(b[1]), b[0], np.sin(b[1]), np.cos(b[1])]

    # Implement evaluation of the boundary condition residuals:

    def bc(self, ya, yb):
        return np.array([ya[0], yb[0] - self.Xe])

    def fun(self, x, y):
        return np.vstack((y[1], -np.exp(y[0])))

    def update_ode(self):
        x = np.linspace(0, 1, 5)
        y_a = np.zeros((2, x.size))
        y_b = np.zeros((2, x.size))
        y_b[0] = 3

        # Now we are ready to run the solver.

        self.res_a = solve_bvp(self.fun, self.bc, x, y_a)
        self.res_b = solve_bvp(self.fun, self.bc, x, y_b)
        self.x_plot = np.linspace(0, 1, 100)
        self.y_plot_a = self.res_a.sol(self.x_plot)[0]
        self.y_plot_b = self.res_b.sol(self.x_plot)[0]

    def update_plot(self):
        self.pick_point.set_ydata([self.Xe, self.Xe])
        self.ax2.draw_artist(self.pick_point)
        self.fig.canvas.draw()

    def update_ode_plot(self):
        self.ax2.plot(self.x_plot, self.y_plot_a, label='y_a')
        self.ax2.plot(self.x_plot, self.y_plot_b, label='y_b')
        self.ax2.legend()

        self.fig.canvas.draw()

    def initial_plot(self):
        self.Xe = self.y_plot_a[-1]
        self.Ze = self.x_plot[-1]

        self.pick_point = plt.Line2D((self.Ze, self.Xe), (self.Ze, self.Xe), marker='.', markersize=10,
                                     markerfacecolor='r', picker=5)
        self.ax2.add_line(self.pick_point)

        self.update_ode_plot()
        plt.show()
        self.update_plot()


if __name__ == '__main__':
    fiber = heavy_planar_elastica()
    fiber.update_ode()
    fiber.initial_plot()
