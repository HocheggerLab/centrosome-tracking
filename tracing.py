import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_bvp

np.set_printoptions(1)


class HeavyPlanarElastica():
    def __init__(self):
        self.L = 3.0
        self.E = 10.0
        self.F = 0.1
        self.gamma = 0.1
        self.endY = 1

    def f(self, s, x, m0):
        k = x[3] / self.E
        return [-cos(x[2]), -sin(x[2]), -k, self.F * sin(x[2] + self.gamma)]

    # Implement evaluation of the boundary condition residuals:
    def bc(self, xa, xb, m0):
        # p = m0
        out = np.array([xb[0], xb[1], xb[2], xa[3] - m0, xa[1] - self.endY])
        # print  np.array(xa), np.array(xb), m0, '->', out
        return out

    def update_ode(self):
        s = np.linspace(0, self.L, 4)
        y_a = np.zeros((4, s.size))

        # Now we are ready to run the solver.
        self.res = solve_bvp(self.f, self.bc, s, y_a, p=[1], verbose=0)
        print 'm0 = %0.2f' % self.res.p[0]


class HeavyPlanarElasticaDrawObject_xy(plt.Artist):
    def __init__(self, axes, elasticaModel):
        plt.Artist.__init__(self)
        self.picking = False
        self.fiber = elasticaModel
        self.Xe = 1
        self.Ye = 1
        self.ax = axes

        self.ax.set_xlabel('x [um]')
        self.ax.set_ylabel('y [um]')

    def on_pick(self, event):
        self.picking = True
        mevent = event.mouseevent
        # print('pickevent: x=%d, y=%d, xdata=%f, ydata=%f' % (mevent.x, mevent.y, mevent.xdata, mevent.ydata))
        self.Ye = mevent.ydata

    def on_motion(self, event):
        if not self.picking: return
        # print('motionevent: picking=%f, event.xdata=%f, event.ydata=%f' % (picking, event.xdata, event.ydata))
        self.Ye = event.ydata
        self._update_picker_point()
        self.ax.figure.canvas.draw()

    def on_release(self, event):
        if not self.picking: return
        self.picking = False
        self.Ye *= -1

        self.fiber.endY = self.Ye

        self.update_plot()

    def _update_ode_plot(self):
        self.fiber.update_ode()

        s_plot = np.linspace(0, self.fiber.L, 100)
        x_plot = self.fiber.res.sol(s_plot)[0]
        y_plot = -self.fiber.res.sol(s_plot)[1]

        self.Xe = x_plot[0]
        self.Ye = y_plot[0]

        self.ax.plot(x_plot, y_plot, label='%0.1f' % self.fiber.res.p[0])
        self.ax.legend()

    def _update_picker_point(self):
        self.pick_point.set_ydata([self.Ye, self.Ye])
        self.pick_point.set_xdata([self.Xe, self.Xe])
        self.ax.draw_artist(self.pick_point)

    def update_plot(self):
        self._update_ode_plot()
        self._update_picker_point()
        self.ax.figure.canvas.draw()

    def initial_plot(self):
        self._update_ode_plot()

        px = (self.Xe, self.Xe)
        py = (self.Ye, self.Ye)
        self.pick_point = plt.Line2D(px, py, marker='.', markersize=10, markerfacecolor='r', picker=5)
        self.ax.add_line(self.pick_point)


if __name__ == '__main__':
    fiber = HeavyPlanarElastica()
    fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # ax2 = fig.add_subplot(122)
    ax = fig.gca()

    fiber_draw = HeavyPlanarElasticaDrawObject_xy(ax, fiber)
    fiber_draw.initial_plot()

    # connect all
    cidpress = fig.canvas.mpl_connect('pick_event', fiber_draw.on_pick)
    cidrelease = fig.canvas.mpl_connect('button_release_event', fiber_draw.on_release)
    cidmotion = fig.canvas.mpl_connect('motion_notify_event', fiber_draw.on_motion)

    plt.show()

    # disconnect all the stored connection ids
    fig.canvas.mpl_disconnect(cidpress)
    fig.canvas.mpl_disconnect(cidrelease)
    fig.canvas.mpl_disconnect(cidmotion)
