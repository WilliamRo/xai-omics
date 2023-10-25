from mpl_toolkits.axes_grid1 import make_axes_locatable
from xomics.gui.dr_gordon import DrGordon, SliceView
from xomics import MedicalImage

import matplotlib.pyplot as plt
import numpy as np



class RLDExplorer(DrGordon):

  def __init__(self, medical_images, title='RLD Explorer'):
    super().__init__(medical_images, title=title)

    self.sv = RLDViewer(self)
    self.set_to_axis(self.Keys.PLOTTERS, [self.sv], overwrite=True)

    self.shortcuts.register_key_event(
      ['l'], lambda: self.set_cursor(self.Keys.LAYERS, 1, refresh=True),
      description='Next layer', color='yellow')
    self.shortcuts.register_key_event(
      ['h'], lambda: self.set_cursor(self.Keys.LAYERS, -1, refresh=True),
      description='Previous layer', color='yellow')


class RLDViewer(SliceView):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(pictor=dr_gordon)

    self.new_settable_attr('view_point', 'other', str,
                           'Change the methods to view(cor, sag, other)')


  def view_slice(self, fig: plt.Figure, ax: plt.Axes, x: int):
    mi: MedicalImage = self.selected_medical_image
    selected_vol = mi.images[self.displayed_layer_key]

    if self.get('view_point') == 'cor':
      selected_vol = selected_vol.swapaxes(0, 1)
    elif self.get('view_point') == 'sag':
      selected_vol = selected_vol.swapaxes(0, 2)

    image: np.ndarray = selected_vol[x]
    im = ax.imshow(image, cmap=self.get('cmap'), vmin=self.get('vmin'),
                   vmax=self.get('vmax'))

    # Set title
    title = mi.key

    # Show title if necessary
    if title != '': ax.set_title(title, fontsize=10)

    # Set style
    ax.set_axis_off()

    # Show color bar if required
    if self.get('color_bar'):
      divider = make_axes_locatable(ax)
      cax = divider.append_axes('right', size='5%', pad=0.05)
      fig.colorbar(im, cax=cax)

  # region: Shortcuts

  def register_shortcuts(self):
    super().register_shortcuts()

    def modify_view():
      vlist = ['other', 'cor', 'sag']
      self.set('view_point', vlist[(vlist.index(self.get('view_point'))+1) % 3])
      self.refresh()

    self.register_a_shortcut(
      'v', modify_view, 'Change the Viewpoints')
  # endregion: Shortcuts



if __name__ == '__main__':
  pass
