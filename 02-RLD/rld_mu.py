from rld.models.rld_gan import *
from tframe import mu, pedia
from rld.layers.layers import *
from rld.loss.custom_loss import *

custom_loss = {
  'ssim': get_ssim_3D(),
  'psnr': get_psnr(),
  'nrmse': get_nrmse(),
  'rela': get_relative_loss(),
}

def get_initial_model():
  from rld_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model

def get_container(flatten_D_input=False) -> mu.GAN:
  gan = PETGAN(
    mark=th.mark, G_input_shape=th.input_shape,
    D_input_shape=th.input_shape)

  if flatten_D_input: gan.D.add(mu.Flatten())

  return gan

def finalize(model):
  from rld_core import th

  assert isinstance(model, mu.Predictor)
  if th.output_conv:
    model.add(mu.HyperConv3D(filters=1, kernel_size=1))

  if th.use_res:
    model.input_.abbreviation = 'input'
    model.add(mu.ShortCut(model.input_, mode=mu.ShortCut.Mode.SUM))

  if th.use_sigmoid:
    model.add(mu.Activation('sigmoid'))
  elif th.clip_off:
    model.add(Clip(0, 1.2))

  if th.internal_loss:
    context.customized_loss_f_net = get_total_loss
    th.probe_cycle = th.updates_per_round

  # Build model
  metrics = list(custom_loss.values())
  if th.loss_string not in list(custom_loss.keys()):
    model.build(loss=th.loss_string, metric=metrics + ['loss'])
  else:
    model.build(loss=custom_loss[th.loss_string], metric=metrics)
  return model


def gan_finalize(gan):
  assert isinstance(gan, mu.GAN)

  gan.G.add(mu.HyperConv3D(filters=1, kernel_size=1))
  gan.D.add(mu.HyperDense(1, activation='lrelu'))
  gan.D.add(mu.Activation('sigmoid', set_logits=True))

  gan.build(loss=pedia.cross_entropy,
            metric=list(custom_loss.values()))

  return gan


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()
  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_unet_list(arc_string='8-3-4-2-relu-mp', **kwargs):
  unet = mu.UNet(3, arc_string=arc_string, **kwargs)
  return unet._get_layers()



