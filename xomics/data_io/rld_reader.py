from xomics.data_io.npy_reader import NpyReader


class RLDReader(NpyReader):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

