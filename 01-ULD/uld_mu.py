from tframe import mu
from uld.operators.custom_layers import Tanh_k, Atanh_k, Clip
from uld.operators.custom_loss import get_ssim_3D, get_nrmse, get_psnr, \
  get_pw_rmse

EPSILON = 0.001

custom_loss = {
  'ssim': get_ssim_3D(),
  'nrmse': get_nrmse(),
  'psnr': get_psnr(),
  'pw_rmse': get_pw_rmse(EPSILON),
}

def get_initial_model():
  from uld_core import th

  model = mu.Predictor(th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if th.use_tanh != 0:
    model.add(Tanh_k(k=th.use_tanh))
  return model


def finalize(model):
  from uld_core import th
  from tframe import tf

  assert isinstance(model, mu.Predictor)
  model.add(mu.HyperConv3D(filters=1, kernel_size=1))

  if th.use_tanh != 0:
    model.add(Atanh_k(k=th.use_tanh))

  if th.use_sigmoid:
    model.add(mu.Activation('sigmoid'))
  else:
    model.add(Clip(0, 1.2))

  if th.learn_delta:
    model.input_.abbreviation = 'input'
    model.add(mu.ShortCut(model.input_, mode=mu.ShortCut.Mode.SUM))

  # Build model
  if th.loss_string not in custom_loss:
    model.build(loss=th.loss_string, metric=[
      get_ssim_3D(), get_nrmse(), get_psnr(), get_pw_rmse(), 'loss'])
  else:
    model.build(loss=custom_loss[th.loss_string], metric=custom_loss.items())
  return model


def get_unet(arc_string='8-3-4-2-relu-mp', **kwargs):
  model = get_initial_model()
  mu.UNet(3, arc_string=arc_string, **kwargs).add_to(model)

  return finalize(model)



