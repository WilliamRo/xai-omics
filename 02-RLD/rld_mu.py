from tframe import mu
from rld.layers.layers import *
from rld.loss.custom_loss import *

custom_loss = {
  'psnr': get_psnr(),
  'ssim': get_ssim_3D(),
  'nrmse': get_nrmse(),
}

def get_initial_model():
  from rld_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def finalize(model):
  from rld_core import th

  assert isinstance(model, mu.Predictor)
  if th.output_conv:
    model.add(mu.HyperConv3D(filters=1, kernel_size=1))

  if th.use_sigmoid:
    model.add(mu.Activation('sigmoid'))
  elif th.clip_off:
    model.add(Clip(0, 1.2))

  # Build model
  metrics = list(custom_loss.values())
  if th.loss_string not in list(custom_loss.keys()):
    model.build(loss=th.loss_string, metric=metrics + ['loss'])
  else:
    model.build(loss=custom_loss[th.loss_string], metric=metrics)
  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()
  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_unet_list(arc_string='8-3-4-2-relu-mp', **kwargs):
  unet = mu.UNet(3, arc_string=arc_string, **kwargs)
  return unet._get_layers()

