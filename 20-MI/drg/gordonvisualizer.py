from mpl_toolkits.axes_grid1 import make_axes_locatable
from pictor import Pictor
from pictor.plotters.plotter_base import Plotter

import matplotlib.pyplot as plt
import numpy as np



class GordonVisualizer(Pictor):

  class Keys(Pictor.Keys):
    LAYER = 'LaYeR'
    CHANNELS = 'ChAnNeL'
    DEPTHS = 'DePtHs'

  def __init__(self, raw_data, layer_name, title='Gordon Visualizer'):
    super(GordonVisualizer, self).__init__(title)

    self.add_plotter(SliceView(layer_name=layer_name, pictor=self))

    # [bs, layer, channel, depth, H, W]
    # Create dimensions for different channels and nets
    self.create_dimension(self.Keys.LAYER)
    self.create_dimension(self.Keys.CHANNELS)
    self.create_dimension(self.Keys.DEPTHS)

    self.set_data(raw_data)


  # region: Overwriting

  def _register_default_key_events(self):
    super(GordonVisualizer, self)._register_default_key_events()

    self.shortcuts.register_key_event(
      ['J'], lambda: self._set_cursor(self.Keys.LAYER, 1),
      description='Next Channel', color='yellow')
    self.shortcuts.register_key_event(
      ['K'], lambda: self._set_cursor(self.Keys.LAYER, -1),
      description='Previous Channel', color='yellow')

    self.shortcuts.register_key_event(
      ['H'], lambda: self._set_cursor(self.Keys.CHANNELS, 1),
      description='Next Channel', color='yellow')
    self.shortcuts.register_key_event(
      ['L'], lambda: self._set_cursor(self.Keys.CHANNELS, -1),
      description='Previous Channel', color='yellow')

    self.shortcuts.register_key_event(
      ['N'], lambda: self._set_cursor(self.Keys.DEPTHS, 1),
      description='Next depth', color='green')
    self.shortcuts.register_key_event(
      ['P'], lambda: self._set_cursor(self.Keys.DEPTHS, -1),
      description='Previous depth', color='green')

  # endregion: Overwriting

  def _set_cursor(self, key, step):
    self.set_cursor(key, step)
    index = list(self.cursors.keys()).index(key)

    self.refresh_dimension(index)
    self.refresh()


  def set_data(self, raw_data):
    self.objects = raw_data
    self.refresh_dimension()


  def refresh_dimension(self, index=0):
    keys = list(self.cursors.keys())
    x = self.objects

    for i in range(0, len(keys)):
      key = keys[i]
      if key == self.Keys.PLOTTERS: continue

      if i > index:
        self.cursors[key] = 0
        self.set_to_axis(key, list(range(len(x))), overwrite=True)

      x = x[self.cursors[key]]



class SliceView(Plotter):

  def __init__(self, layer_name, pictor=None):
    # Call parent's constructor
    super(SliceView, self).__init__(self.view_slices, pictor)
    self.layer_name = layer_name

    # Settable attributes
    self.new_settable_attr('color_bar', False, bool, 'Color bar')
    self.new_settable_attr('cmap', 'gray', float, 'Color map')
    self.new_settable_attr('vmin', 0., float, 'Min value')
    self.new_settable_attr('vmax', 1., float, 'Max value')


  def view_slices(self, fig: plt.Figure, ax: plt.Axes, x):
    layer_cursor = self.pictor.cursors[self.pictor.Keys.LAYER]
    channel_cursor = self.pictor.cursors[self.pictor.Keys.CHANNELS]
    depth_cursor = self.pictor.cursors[self.pictor.Keys.DEPTHS]

    data = x[layer_cursor][channel_cursor][depth_cursor]

    # Show slice
    im = ax.imshow(data, cmap=self.get('cmap'),
                   vmin=self.get('vmin'), vmax=self.get('vmax'))
    ax.set_title(f'{self.layer_name[layer_cursor]} {data.shape[0]}x{data.shape[1]}')

    # Set style
    ax.set_axis_off()

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)


  def register_shortcuts(self):
    self.register_a_shortcut(
      'C', lambda: self.flip('color_bar'), 'Turn on/off color bar')



if __name__ == '__main__':
  pass
