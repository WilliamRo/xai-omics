import numpy as np



class SplCom:
  def __init__(self, arr: np.ndarray, kernel_size, dim: list, processor=None):
    assert len(dim) == len(arr.shape)
    assert kernel_size % 2 == 1

    self.slices = []
    self.iter_record = []
    self.ids = []
    self.shape = arr.shape
    self.pad = (kernel_size - 1)//2
    self.step = self.pad * 2
    self.arr = arr
    self.dim = dim
    self.length = 1

    for i in range(len(self.shape)):
      if dim[i] != 0:
        self.length *= (self.shape[i] - self.step)//(self.dim[i] - self.step)

    if processor is None:
      self.processor = lambda x: x
    else:
      self.processor = processor

    for i in range(len(self.shape)):
      if dim[i] != 0:
        self.slices.append(slice(0, dim[i]))
        self.ids.append(i)
        self.iter_record.append(0)
      else:
        self.slices.append(slice(0, self.shape[i]))

    self.sub_arr = None
    self.com_arr = None
    self.com_slices = self.slices.copy()
    self.off_slices = self.slices.copy()

  def __len__(self):
    return self.length

  def __iter__(self):
    return self

  def __next__(self):
    if self.sub_arr is None:
      self.sub_arr = self.arr[tuple(self.slices)]
      return self.sub_arr
    else:
      result = self.update()
      if result is StopIteration:
        raise result
      else:
        return result

  def update(self):
    # iter_record = self.iter_record.copy()
    self.iter_record[0] = 1
    for rid, i in enumerate(self.ids):
      if self.iter_record[rid]:
        start = self.slices[i].stop - self.step
        end = start + self.dim[i]
        com_start = self.com_slices[i].stop - self.pad
        com_end = com_start + self.dim[i] - self.pad
        off_start = self.pad
        off_end = self.dim[i]

        self.slices[i] = slice(start, end)
        self.com_slices[i] = slice(com_start, com_end)
        self.off_slices[i] = slice(off_start, off_end)

        if self.slices[i].start + self.step >= self.shape[i]:
          if len(self.iter_record) <= rid+1:
            return StopIteration
          self.iter_record[rid+1] = 1

          self.slices[i] = slice(0, self.dim[i])
          self.com_slices[i] = slice(0, self.dim[i])
          self.off_slices[i] = slice(0, self.dim[i])

        self.iter_record[rid] = 0
    self.sub_arr = self.arr[tuple(self.slices)]
    return self.sub_arr

  def combine(self, arr):
    if self.com_arr is None:
      self.com_arr = np.zeros(self.shape)
    self.com_arr[tuple(self.com_slices)] = arr[tuple(self.off_slices)]

  def execute(self):
    for i in self:
      self.combine(self.processor(i))
    return self.com_arr





if __name__ == '__main__':
  data = np.ones((1, 684, 440, 440, 1))
  a = SplCom(data, 3, [0, 24, 0, 0, 0])
  b = a.execute()
  print(1 in b)
  pass

