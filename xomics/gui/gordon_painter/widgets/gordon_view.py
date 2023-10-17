from xomics.gui.dr_gordon import SliceView

import numpy as np



class GordonView(SliceView):

  def __init__(self, gordon_painter=None):
    # Call parent's constructor
    super().__init__(pictor=gordon_painter)

    self.percentile = None
    self.click_address = None
    self.new_label_name = 'label-10'

    # Settable attributes
    self.new_settable_attr('painter', False, bool, 'Painter')
    self.pictor.canvas._canvas.mpl_connect(
      'button_press_event', self.on_mouse_click)
    self.pictor.canvas._canvas.mpl_connect(
      'scroll_event', self.on_mouse_scroll)


  def register_shortcuts(self):
    super().register_shortcuts()
    self.register_a_shortcut(
      'Q', lambda: self.flip('painter'), 'Turn on/off painter')


  def on_mouse_click(self, event):
    if event.xdata is not None and event.ydata is not None and self.get('painter'):
      # Coordinates of the mouse click
      y, x = int(event.xdata), int(event.ydata)
      num_slice = self.pictor.cursors[self.pictor.Keys.OBJECTS]
      image = self.selected_medical_image.images[self.displayed_layer_key]
      data = image[num_slice, x, y]

      # get the percentile
      flatten_image = np.sort(image.flatten())
      percentile = round(
        ((np.searchsorted(flatten_image, data) + 1) / flatten_image.size * 100), 2)

      # calculate the percentile and the threshold
      percentile = max(percentile - 0.5, 0)
      self.percentile = percentile
      self.click_address = tuple([num_slice, x, y])
      threshold = np.percentile(image, q=percentile)

      # get the mask and add to mi
      mask = (image >= threshold).astype(np.uint8)
      mask = self.amend_mask(mask, tuple([num_slice, x, y]))
      self.selected_medical_image.labels[self.new_label_name] = mask

      # visualiza the new label
      self.annotations_to_show.append(self.new_label_name)
      self.refresh()

      print(f"Mouse clicked at ({x:.2f}, {y:.2f})")
      print(f"percentile: {percentile}")
      print(f"threshold: {threshold}")


  def on_mouse_scroll(self, event):
    if event.button == 'up':
      self.pictor.set_cursor(self.pictor.Keys.OBJECTS, 1, refresh=True)
    elif event.button == 'down':
      self.pictor.set_cursor(self.pictor.Keys.OBJECTS, -1, refresh=True)


  def amend_mask(self, mask: np.ndarray, chosen_data_address: tuple):
    from scipy.ndimage import label, generate_binary_structure

    structure = generate_binary_structure(3, 1)
    # Get connected region for denoise
    labeled_image, num_features = label(mask, structure)

    # Gets the label number of the connected area where the mouse clicks
    num_label = labeled_image[chosen_data_address]
    labeled_image[labeled_image != num_label] = 0
    labeled_image[labeled_image == num_label] = 1

    return labeled_image


  def adjust_mask(self, percentile):
    image = self.selected_medical_image.images[self.displayed_layer_key]
    threshold = np.percentile(image, q=percentile)

    new_mask = (image >= threshold).astype(np.uint8)
    new_mask = self.amend_mask(new_mask, self.click_address)
    self.selected_medical_image.labels[self.new_label_name] = new_mask



if __name__ ==  '__main__':
  pass
