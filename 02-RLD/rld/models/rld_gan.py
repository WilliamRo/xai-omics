from rld.loss.custom_loss import nrmse
from tframe.core import TensorSlot, with_graph
from tframe.core.quantity import Quantity
from tframe import DataSet, pedia, tf, context, hub
from tframe.models import GAN
from tframe.utils import checker



class PETGAN(GAN):

  def __init__(self, mark, G_input_shape, D_input_shape, learning_rate=0.0001):
    super().__init__(mark, G_input_shape, D_input_shape)

    self.lr = learning_rate
    self._targets = TensorSlot(self, 'targets')

  @with_graph
  def predict(self, data, batch_size=None, **kwargs):
    # self.launch_model()
    return self.evaluate(self.val_outputs.tensor, data,
                         batch_size=batch_size, **kwargs)

  # region : Overwriting

  def _evaluate_batch(self, fetch_list, data_batch, **kwargs):
    # Sanity check
    assert isinstance(fetch_list, list)
    checker.check_fetchable(fetch_list)
    assert isinstance(data_batch, DataSet)

    # Run session
    feed_dict = self._get_default_feed_dict(data_batch, is_training=False)
    batch_outputs = self.session.run(fetch_list, feed_dict)

    return batch_outputs

  def generate(self, z=None, **kwargs):
    # Check model and session
    if not self.G.linked: raise AssertionError('!! Model not built yet.')
    if not self.launched: self.launch_model(overwrite=False)

    # Generate samples
    feed_dict = {self.G.input_tensor: z}
    feed_dict.update(self.agent.get_status_feed_dict(is_training=False))
    samples = self.outputs.run(feed_dict)
    return samples

  def _build(self, optimizer=None, loss=pedia.cross_entropy,
             G_optimizer=None, D_optimizer=None, metric=None, **kwargs):
    # Link generator
    self._G_output = self.Generator()

    # Initiate targets and add it to collection
    shape = self._G_output.shape.as_list()
    target_tensor = tf.placeholder(hub.dtype, shape, name='targets')
    self._targets.plug(target_tensor, collection=pedia.default_feed_dict)
    self._val_targets = self._targets

    # Create val_output tensor
    val_output = self._G_output
    self.val_outputs.plug(val_output)

    # Plug self._G_output to GAN.output slot
    self.outputs.plug(self._G_output)

    # Link discriminator
    logits_dict = context.logits_tensor_dict
    self._Dr = self.Discriminator()
    self._logits_Dr = logits_dict.pop(list(logits_dict.keys())[0])
    self._Df = self.Discriminator(self._G_output)
    self._logits_Df = logits_dict.pop(list(logits_dict.keys())[0])

    # Define loss (extra losses are not supported yet)
    with tf.name_scope('Losses'):
      self._define_losses(loss, kwargs.get('smooth_factor', 0.9))

    # Define train steps
    if G_optimizer is None:
      G_optimizer = tf.train.AdamOptimizer(self.lr)
    if D_optimizer is None:
      D_optimizer = tf.train.AdamOptimizer(self.lr)

    with tf.name_scope('Train_Steps'):
      with tf.name_scope('G_train_step'):
        self._train_step_G.plug(G_optimizer.minimize(
          loss=self._loss_G.tensor, var_list=self.G.parameters))
      with tf.name_scope('D_train_step'):
        self._train_step_D.plug(D_optimizer.minimize(
          loss=self._loss_D.tensor, var_list=self.D.parameters))

    if metric is not None:
      checker.check_type_v2(metric, (str, Quantity))

      with tf.name_scope('Metric'):
        self._metrics_manager.initialize(
          metric, False, self._val_targets.tensor,
          self.val_outputs.tensor, **kwargs)

  def _define_losses(self, loss, alpha):
    """To add extra losses, e.g., regularization losses, this method should be
    overwritten"""
    from tframe import hub as th

    if callable(loss):
      self._loss_G, self._loss_D = loss(self)
      assert False
      return
    elif not isinstance(loss, str):
      raise TypeError('loss must be callable or a string')

    loss = loss.lower()
    if loss == pedia.default:
      loss_Dr_raw = -tf.log(self._Dr, name='loss_D_real_raw')
      loss_Df_raw = -tf.log(1. - self._Df, name='loss_D_fake_raw')
      loss_G_raw = -tf.log(self._Df, name='loss_G_raw')
    elif loss == pedia.cross_entropy:
      loss_Dr_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._logits_Dr, labels=tf.ones_like(self._logits_Dr) * alpha)
      loss_Df_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._logits_Df, labels=tf.zeros_like(self._logits_Df))
      loss_G_raw = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self._logits_Df, labels=tf.ones_like(self._logits_Df))
    else:
      raise ValueError('Can not resolve "{}"'.format(loss))

    loss_G_pixel = nrmse(self.D.input_tensor, self._G_output)


    with tf.name_scope('D_losses'):
      loss_Dr = tf.reduce_mean(loss_Dr_raw, name='loss_D_real')
      loss_Df = tf.reduce_mean(loss_Df_raw, name='loss_D_fake')
      loss_D = tf.add(loss_Dr, loss_Df, name='loss_D')
      self._loss_D.plug(loss_D)
    with tf.name_scope('G_loss'):
      self._loss_G.plug(tf.reduce_mean(loss_G_raw+loss_G_pixel, name='loss_G'))

  def update_model(self, data_batch, **kwargs):
    assert isinstance(data_batch, DataSet)

    result = None

    # (1) Update D
    for _ in range(1):
      feed_dict_D = {self.D.input_tensor: data_batch.targets,
                     self.G.input_tensor: data_batch.features}
      feed_dict_D.update(self.agent.get_status_feed_dict(is_training=True))
      if result is None:
        results = self._update_group_D.run(feed_dict_D)
      else:
        results.update(self._update_group_D.run(feed_dict_D))

    # (2) Update G
    for _ in range(1):
      feed_dict_G = {self.G.input_tensor: data_batch.features,
                     self.D.input_tensor: data_batch.targets}
      feed_dict_G.update(self.agent.get_status_feed_dict(is_training=True))
      results.update(self._update_group_G.run(feed_dict_G))

    return results

  # def validate_model(self, **kwargs):
  #   return

  # endregion : Overwriting



