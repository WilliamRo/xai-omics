from tframe.utils.note import Note



fp = r'P:\xai-sleep\09-S2S\01_cnn_v1\0917_s1_cnn.sum'

notes = Note.load(fp)

ks, bn = 3, True

interested_notes = [
  n for n in notes
  if n.configs['kernel_size'] == ks and n.configs['use_batchnorm']]

for n in interested_notes:
  print(n.criteria['Best Accuracy'])


