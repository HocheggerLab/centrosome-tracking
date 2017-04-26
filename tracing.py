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




class hpe_artist(plt.Artist):
    def __init__(self, elasticaModel):
        plt.Artist.__init__(self)
        self.model = elasticaModel

global picking, fiber
picking = False
fiber = heavy_planar_elastica()


def on_pick(event):
    global picking
    picking = True
    mevent = event.mouseevent
    # print('pickevent: x=%d, y=%d, xdata=%f, ydata=%f' % (mevent.x, mevent.y, mevent.xdata, mevent.ydata))
    fiber.Xe = mevent.ydata


def on_motion(event):
    global picking
    'on motion we will move the rect if the mouse is over us'
    if not picking: return
    # print('motionevent: picking=%f, event.xdata=%f, event.ydata=%f' % (picking, event.xdata, event.ydata))
    fiber.Xe = fiber.x_plot[0]
    fiber.Ye = event.ydata

    update_plot(ax1,ax2)
    fig.canvas.draw()


def on_release(event):
    global picking
    if not picking: return
    picking = False
    fiber.Ye *= -1
    fiber.update_ode()
    update_ode_plot(ax1,ax2)
    update_plot(ax1,ax2)
    fig.canvas.draw()

def update_ode_plot(ax1,ax2):
    global fiber
    ax1.plot(fiber.s_plot, fiber.M_plot, label='b_a')
    ax2.plot(fiber.x_plot, fiber.y_plot, label='%0.1f' % fiber.res.p[0])
    ax2.legend()

def update_plot(ax1,ax2):
    global fiber
    fiber.pick_point.set_ydata([fiber.Ye, fiber.Ye])
    fiber.pick_point.set_xdata([fiber.Xe, fiber.Xe])
    ax2.draw_artist(fiber.pick_point)

def initial_plot():
    global fiber
    px = (fiber.Xe, fiber.Xe)
    py = (fiber.Ye, fiber.Ye)
    fiber.pick_point = plt.Line2D(px, py, marker='.', markersize=10, markerfacecolor='r', picker=5)

if __name__ == '__main__':
    global picking, fiber
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fiber.update_ode()
    initial_plot()
    ax2.add_line(fiber.pick_point)
    update_ode_plot(ax1,ax2)

    ax1.set_xlabel('s')
    ax1.set_ylabel('M(s)')
    ax2.set_xlabel('x [um]')
    ax2.set_ylabel('y [um]')

    # connect all
    cidpress = fig.canvas.mpl_connect('pick_event', on_pick)
    cidrelease = fig.canvas.mpl_connect('button_release_event', on_release)
    cidmotion = fig.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.show()

    # disconnect all the stored connection ids
    # fig.figure.canvas.mpl_disconnect(cidpress)
    # fig.figure.canvas.mpl_disconnect(cidrelease)
    # fig.figure.canvas.mpl_disconnect(cidmotion)
