import matplotlib.patches


class DraggableCircle:
    def __init__(self, circle, on_pick_callback=None, on_release_callback=None):
        if type(circle) != matplotlib.patches.Circle: raise Exception('not a circle')
        self.circle = circle
        self.press = None
        self._on_pick_call = on_pick_callback
        self._on_rel_call = on_release_callback
        self._hidden = False

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.circle.figure.canvas.mpl_connect('button_press_event', lambda s: self.on_press(s))
        self.cidrelease = self.circle.figure.canvas.mpl_connect('button_release_event', lambda s: self.on_release(s))
        self.cidmotion = self.circle.figure.canvas.mpl_connect('motion_notify_event', lambda s: self.on_motion(s))

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if self._hidden: return
        if self.press is not None: return
        if event.inaxes != self.circle.axes: return

        contains, attrd = self.circle.contains(event)
        if not contains: return
        print('event contains', self.circle.center)
        x0, y0 = self.circle.center
        self.press = x0, y0, self.circle.radius, event.xdata, event.ydata

        if self._on_pick_call is not None:
            self._on_pick_call()

    def on_motion(self, event):
        'on motion we will move the circle if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.circle.axes: return
        x0, y0, r, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.circle.center = (x0 + dx, y0 + dy)
        self.circle.figure.canvas.draw()

        # print('x0=%f, xpress=%f, event.xdata=%f, dx=%f, x0+dx=%f' %
        #       (x0, xpress, event.xdata, dx, x0 + dx))

    def on_release(self, event):
        'on release we reset the press data'
        if self.press is None: return
        self.press = None
        self.circle.figure.canvas.draw()

        self.picked = False

        if self._on_rel_call is not None:
            self._on_rel_call()

    def hide(self):
        self.press = None
        self._hidden = True
        self.circle.set_visible(False)

    def show(self):
        self.press = None
        self._hidden = False
        self.circle.set_visible(True)
