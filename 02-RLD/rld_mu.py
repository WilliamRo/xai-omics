from rld.models.models import *
from tframe import mu, pedia
from rld.layers.layers import *
from rld.loss.custom_loss import *

custom_loss = {
  'ssim': get_ssim_3D(),
  'psnr': get_psnr(),
  'nrmse': get_nrmse(),
  'rela': get_relative_loss(),
}

def get_conv(dimension):
  if dimension == 1:
    return mu.HyperConv1D
  elif dimension == 2:
    return mu.HyperConv2D
  elif dimension == 3:
    return mu.HyperConv3D
  else:
    raise ValueError('!! Unknown dimension `{}`'.format(dimension))


def get_initial_model():
  from rld_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model


def get_gan_container(flatten_D_input=False) -> mu.GAN:
  from rld_core import th
  gan = PETGAN(
    mark=th.mark, G_input_shape=th.input_shape,
    D_input_shape=th.input_shape, learning_rate=th.learning_rate)

  if flatten_D_input: gan.D.add(mu.Flatten())

  return gan


def get_ddpm_container(time_steps, time_dim) -> mu.GaussianDiffusion:
  from rld_core import th
  model = DDPM(
    mark=th.mark, x_shape=th.input_shape, time_steps=time_steps,
    beta_schedule=th.beta_schedule, time_dim=time_dim)
  return model


def finalize(model):
  from rld_core import th

  assert isinstance(model, mu.Predictor)
  if th.output_conv:
    model.add(get_conv(th.dimension)(filters=1, kernel_size=1))

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
  from rld_core import th

  gan.G.add(get_conv(th.dimension)(filters=1, kernel_size=1))
  gan.D.add(mu.HyperDense(1, activation='lrelu'))
  gan.D.add(mu.Activation('sigmoid', set_logits=True))

  gan.build(loss=pedia.cross_entropy,
            metric=list(custom_loss.values()))

  return gan


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  from rld_core import th
  model = get_initial_model()
  mu.UNet(th.dimension, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)


def get_unet_list(arc_string='8-3-4-2-relu-mp', **kwargs):
  from rld_core import th
  unet = mu.UNet(th.dimension, arc_string=arc_string, **kwargs)
  return unet._get_layers()



