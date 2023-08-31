import matplotlib.pyplot as plt

from xomics.gui.dr_gordon import DrGordon, SliceView, Plotter
from xomics import MedicalImage

import os
import numpy as np



class ULDExplorer(DrGordon):
  def __init__(self, medical_images, title='ULDose Explorer'):
    super().__init__(medical_images, title=title)

    self.dv = DoseViewer(self)
    self.set_to_axis(self.Keys.PLOTTERS, [self.dv], overwrite=True)

  # region: Data IO

  @staticmethod
  def load_subject(data_dir, subject, doses):
    data_dict = {}
    for dose in doses:
      file_path = os.path.join(data_dir, f'subject{subject}',
                              f'subject{subject}_{dose}.npy')
      data_dict[f'{dose}'] = np.load(file_path)[0]
    return MedicalImage(f'Subject-{subject}', data_dict)

  # endregion: Data IO



class DoseViewer(SliceView):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(pictor=dr_gordon)



class HistogramViewer(Plotter):

  def __init__(self, dr_gordon=None):
    # Call parent's constructor
    super().__init__(self.view_hist, pictor=dr_gordon)

    self.new_settable_attr('xmin', None, float, 'Min x')
    self.new_settable_attr('xmax', None, float, 'Max x')
    self.new_settable_attr('ymin', None, float, 'Min y')
    self.new_settable_attr('ymax', None, float, 'Max y')


  def view_hist(self, ax: plt.Axes):
    mi: MedicalImage = self.pictor.get_element(self.pictor.Keys.PATIENTS)
    i_layer = self.pictor.get_element(self.pictor.Keys.LAYERS)
    key = list(mi.images.keys())[i_layer]
    x = np.ravel(mi.images[key])
    ax.hist(x=x, bins=50, log=True)

    ax.set_xlim(self.get('xmin'), self.get('xmax'))
    ax.set_ylim(self.get('ymin'), self.get('ymax'))



if __name__ == '__main__':
  data_dir = r'../../../data/01-ULD/'

  subjects = [1]
  doses = [
    'Full',
    # '1-2',
    '1-4',
    # '1-10',
    '1-20',
    # '1-50',
    '1-100',
  ]
  mi_list = [ULDExplorer.load_subject(data_dir, s, doses) for s in subjects]

  ue = ULDExplorer(mi_list)
  ue.add_plotter(HistogramViewer())
  ue.dv.set('vmin', auto_refresh=False)
  ue.dv.set('vmax', auto_refresh=False)
  ue.show()
