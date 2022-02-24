# coding=utf-8
# 
# Copyright 2022 by Jiaxin Shi, Yuhao Zhou, Jessica Hwang, Michalis Titsias, Lester Mackey
# https://arxiv.org/abs/2202.09497
# 
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.stats.leave_one_out import log_soomean_exp
import numpy as np

keras = tf.keras
tfd = tfp.distributions

EPS = 1e-6


def safe_log_prob(p):
  return tf.math.log(tf.clip_by_value(p, EPS, 1.0))


def logit_func(prob_tensor):
  """Calculate logits."""
  return safe_log_prob(prob_tensor) - safe_log_prob(1. - prob_tensor)


def _sample_uniform_variables(sample_shape, nfold=1):
  if nfold > 1:
    sample_shape = tf.concat(
        [sample_shape[0:1] * nfold, sample_shape[1:]],
        axis=0)
  return tf.random.uniform(shape=sample_shape, maxval=1.0)


class BinaryNetwork(tf.keras.Model):
  """Network generating binary samples."""

  def __init__(self,
               hidden_sizes,
               activations,
               mean_xs=None,
               demean_input=False,
               final_layer_bias_initializer='zeros',
               name='binarynet'):

    super().__init__(name=name)
    assert len(activations) == len(hidden_sizes)

    num_layers = len(hidden_sizes)
    self.hidden_sizes = hidden_sizes
    self.output_event_shape = hidden_sizes[-1]
    self.activations = activations
    self.networks = keras.Sequential()

    if demean_input:
      if mean_xs is not None:
        self.networks.add(
            tf.keras.layers.Lambda(lambda x: x - mean_xs))
      else:
        self.networks.add(
            tf.keras.layers.Lambda(lambda x: 2.*tf.cast(x, tf.float32) - 1.))
    for i in range(num_layers-1):
      self.networks.add(
          keras.layers.Dense(
              units=hidden_sizes[i],
              activation=activations[i]))

    self.networks.add(
        keras.layers.Dense(
            units=hidden_sizes[-1],
            activation=activations[-1],
            bias_initializer=final_layer_bias_initializer))

  def __call__(self,
               input_tensor,
               samples=None,
               num_samples=(),
               u_noise=None):
    logits = self.get_logits(input_tensor)
    dist = tfd.Bernoulli(logits=logits)
    if u_noise is not None:
      samples = tf.cast(u_noise < tf.math.sigmoid(logits), tf.float32)
    elif samples is None:
      samples = dist.sample(num_samples)
    samples = tf.cast(samples, tf.float32)
    likelihood = dist.log_prob(samples)
    return samples, likelihood, logits

  def get_logits(self, input_tensor):
    logits = self.networks(input_tensor)
    return logits


class DiscreteVAE(tf.keras.Model):
  """Discrete VAE as described in ARM, (Yin and Zhou (2019))."""

  def __init__(self,
               encoder,
               decoder,
               prior_logits,
               grad_type='arm',
               control_nn=None,
               shared_randomness=False,
               name='dvae'):
    super().__init__(name)

    self.num_layers = len(encoder)
    assert len(encoder) == len(decoder)
    if self.num_layers > 1:
      # for multi-layer discrete VAE
      self.encoder_list = encoder
      self.decoder_list = decoder
    else:
      raise NotImplementedError()

    self.shared_randomness = shared_randomness
    self.prior_logits = prior_logits
    self.prior_dist = tfd.Bernoulli(logits=self.prior_logits)
    self.grad_type = grad_type.lower()

    # used for variance of gradients estiamations.
    self.ema = tf.train.ExponentialMovingAverage(0.999)

    if self.grad_type == 'relax':
      self.log_temperature_variable = tf.Variable(
          initial_value=tf.math.log(0.1),  # Reasonable init
          dtype=tf.float32)

      # the scaling_factor is a trainable ~1.
      self.scaling_variable = tf.Variable(
          initial_value=1.,
          dtype=tf.float32)

      # neural network for control variates lambda * r(z)
      self.control_nn = control_nn
    elif self.grad_type == 'double_cv':
      self.alpha = tf.Variable(
        np.zeros([self.num_layers], dtype=np.float32),
        trainable=True,
        name="alpha",
        dtype=tf.float32)
    elif self.grad_type.startswith('discrete_stein'):
      self.alpha = tf.Variable(
        np.zeros([self.num_layers], dtype=np.float32),
        trainable=True,
        name="alpha",
        dtype=tf.float32)
      self.beta = tf.Variable(
        np.zeros([self.num_layers], dtype=np.float32),
        trainable=True,
        name="beta",
        dtype=tf.float32)
      self.control_nn = control_nn

  def call(self, input_tensor, hidden_samples=None, num_samples=()):
    if self.num_layers == 1:
      raise NotImplementedError()
    elif self.num_layers > 1:
      return self.multilayer_call(input_tensor, sample_list=hidden_samples)

  def multilayer_call(self, input_tensor, sample_list=None):
    """Returns ELBO for multi-layer discrete VAE.

    Args:
      input_tensor: a `float` Tensor for input observations.
        The tensor is of the shape [batch_size, observation_dims].
      sample_list: contains the samples of hidden layers:
        [b[1], b[2], ..., b[l]], where l is the number of hidden layers.

    Returns:
      elbo: the ELBo with shape `[batch_size]`.
      encoder_sample_list: the samples from each stochastic encoder layers.
        The length of the list is the same of number of layers. Each of the
        shape `[batch_size, b[i]]` for the i-th hidden layer.
      encoder_logits: the concatenated encoder logits with the shape
        `[batch_size, b[1] + b[2] + ... + b[l]]`, where l is the number
        of hidden layers.
      encoder_llk: the encoder likelihood with shape `[batch_size]`.
    """
    encoder_llk_list = []
    encoder_logits_list = []
    decoder_llk_list = []

    # The `encoder_sample_list` contains `[x, b[1], b[2], ..., b[l]]`,
    # where `l` is the number of layers, `self.num_layers`.
    encoder_sample_list = [input_tensor]

    if sample_list is not None:
      num_fixed_layers = len(sample_list)
      encoder_sample_list.extend(sample_list)
      for i in range(num_fixed_layers):
        _, encoder_llk_i, encoder_logits_i = self.encoder_list[i](
            encoder_sample_list[i], encoder_sample_list[i+1])
        encoder_llk_list.append(tf.reduce_sum(encoder_llk_i, axis=-1))
        encoder_logits_list.append(encoder_logits_i)
    else:
      num_fixed_layers = 0

    current_sample = encoder_sample_list[-1]
    for encoder_i in self.encoder_list[num_fixed_layers:]:
      current_sample, encoder_llk_i, encoder_logits_i = encoder_i(
          current_sample)
      encoder_sample_list.append(current_sample)
      encoder_llk_list.append(tf.reduce_sum(encoder_llk_i, axis=-1))
      encoder_logits_list.append(encoder_logits_i)

    log_pb = tf.reduce_sum(
        self.prior_dist.log_prob(encoder_sample_list[-1]),
        axis=-1)

    # decoder_sample_list is `[b[l], ..., b[2], b[1], x]`
    decoder_sample_list = encoder_sample_list[::-1]
    for i, decoder_i in enumerate(self.decoder_list):
      decoder_llk_i = decoder_i(decoder_sample_list[i],
                                decoder_sample_list[i+1])[1]
      decoder_llk_list.append(tf.reduce_sum(decoder_llk_i, axis=-1))

    # After `tf.stack`, the `Tensor` is of the shape `[num_layers, batch_size]`.
    # After `tf.reduce_sum`, the `Tensor` is of the shape `[batch_size]`.
    encoder_llk = tf.reduce_sum(encoder_llk_list, axis=0)
    decoder_llk = tf.reduce_sum(decoder_llk_list, axis=0)
    elbo = decoder_llk + log_pb - encoder_llk

    encoder_logits = tf.concat(encoder_logits_list, axis=-1)

    return elbo, encoder_sample_list, encoder_logits, encoder_llk

  def get_elbo(self, input_tensor, hidden_tensor):
    """Returns ELBO.

    Args:
      input_tensor: a `float` Tensor for input observations.
        The tensor is of the shape [batch_size, observation_dims].
      hidden_tensor: a discrete Tensor for hidden states.
        The tensor is of the shape [batch_size, hidden_dims].

    Returns:
      elbo: the ELBO with shape [batch_size].
    """
    elbo = self.call(input_tensor, hidden_samples=hidden_tensor)[0]
    return elbo

  def get_multilayer_uniform_sample(self, batch_shape, nfold=1):
    batch_shape = tf.TensorShape(batch_shape)
    u_noise_list = []
    for l in range(self.num_layers):
      # full_sample_shape = [batch_size, hidden_unit_size]
      full_sample_shape = batch_shape.concatenate(
          self.encoder_list[l].output_event_shape)
      u_noise_list.append(_sample_uniform_variables(full_sample_shape, nfold=nfold))
    return u_noise_list

  def get_multilayer_bernoulli_sample(self, sample_list, u_noise_list=None,
                                      resampled_layer=0):
    # The `sample_list` contains `[x, b[1], b[2], ..., b[l]]`,
    # where `l` is the number of layers, `self.num_layers`.
    resampled = sample_list[:resampled_layer+1]
    prev_sample = resampled[resampled_layer]

    if self.shared_randomness and u_noise_list:
      for l in range(resampled_layer, self.num_layers):
        prev_sample = self.encoder_list[l](prev_sample,
                                           u_noise=u_noise_list[l])[0]
        resampled.append(prev_sample)
    else:
      for l in range(resampled_layer, self.num_layers):
        prev_sample = self.encoder_list[l](prev_sample)[0]
        resampled.append(prev_sample)
    return resampled

  def get_multilayer_grad_estimation(
      self, sample_list, u_noise_list, grad_type=None, start_layer=0):
    if grad_type is None:
      grad_type = self.grad_type

    # The `sample_list` contains `[x, b[1], b[2], ..., b[l]]`,
    # where `l` is the number of layers, `self.num_layers`.

    encoder_logits = self.encoder_list[start_layer].get_logits(
        input_tensor=sample_list[start_layer])
    sigma_phi = tf.math.sigmoid(encoder_logits)
    u_noise = u_noise_list[start_layer]

    def sample_from_u(u, tape=None):
      b = tf.cast(u < sigma_phi, tf.float32)
      if tape is not None:
        tape.watch(b)
      sample_list_b = sample_list[:start_layer+1]
      sample_list_b.append(b)
      sample_list_b = self.get_multilayer_bernoulli_sample(
          sample_list_b, u_noise_list, resampled_layer=start_layer+1)

      f = self.get_elbo(sample_list_b[0], sample_list_b[1:])[:, tf.newaxis]
      return b, f

    if grad_type == 'arm':
      # Augment-REINFORCE-merge estimator
      b1 = tf.cast(u_noise > 1. - sigma_phi, tf.float32)
      b2 = tf.cast(u_noise < sigma_phi, tf.float32)

      sample_list_b1 = sample_list[:start_layer+1]
      sample_list_b1.append(b1)
      sample_list_b1 = self.get_multilayer_bernoulli_sample(
          sample_list_b1, u_noise_list, resampled_layer=start_layer+1)
      sample_list_b2 = sample_list[:start_layer+1]
      sample_list_b2.append(b2)
      sample_list_b2 = self.get_multilayer_bernoulli_sample(
          sample_list_b2, u_noise_list, resampled_layer=start_layer+1)

      f1 = self.get_elbo(sample_list_b1[0], sample_list_b1[1:])[:, tf.newaxis]
      f2 = self.get_elbo(sample_list_b2[0], sample_list_b2[1:])[:, tf.newaxis]
      layer_grad = (f1 - f2) * (u_noise - 0.5)

    elif grad_type == 'disarm':
      sigma_abs_phi = tf.math.sigmoid(tf.math.abs(encoder_logits))
      b1 = tf.cast(u_noise > 1. - sigma_phi, tf.float32)
      b2 = tf.cast(u_noise < sigma_phi, tf.float32)
      sample_list_b1 = sample_list[:start_layer+1]
      sample_list_b1.append(b1)
      sample_list_b1 = self.get_multilayer_bernoulli_sample(
          sample_list_b1, u_noise_list, resampled_layer=start_layer+1)
      sample_list_b2 = sample_list[:start_layer+1]
      sample_list_b2.append(b2)
      sample_list_b2 = self.get_multilayer_bernoulli_sample(
          sample_list_b2, u_noise_list, resampled_layer=start_layer+1)

      f1 = self.get_elbo(sample_list_b1[0], sample_list_b1[1:])[:, tf.newaxis]
      f2 = self.get_elbo(sample_list_b2[0], sample_list_b2[1:])[:, tf.newaxis]
      # the factor is I(b1+b2=1) * (-1)**b2 * sigma(|phi|)
      disarm_factor = ((1. - b1) * (b2) + b1 * (1. - b2)) * (-1.)**b2
      disarm_factor *= sigma_abs_phi
      layer_grad = 0.5 * (f1 - f2) * disarm_factor

    elif grad_type == 'reinforce_loo':
      # 2-sample REINFORCE with leave-one-out baseline.
      u1 = u_noise
      u2 = _sample_uniform_variables(sample_shape=tf.shape(encoder_logits))

      b1, f1 = sample_from_u(u1)
      b2, f2 = sample_from_u(u2)
      layer_grad = 0.5 * ((f1 - f2) * (b1 - sigma_phi)
                          + (f2 - f1) * (b2 - sigma_phi))

    elif grad_type == 'double_cv':
      u1 = u_noise
      u2 = _sample_uniform_variables(sample_shape=tf.shape(encoder_logits))

      with tf.GradientTape(persistent=True) as tape:
        b1, f1 = sample_from_u(u1, tape)
        b2, f2 = sample_from_u(u2, tape)
        fb_sum = 0.5 * tf.reduce_sum(f1 + f2)

      grad_b1 = tape.gradient(fb_sum, b1)
      grad_b2 = tape.gradient(fb_sum, b2)
      del tape

      alpha = self.alpha[start_layer]

      c1 = alpha * tf.reduce_sum(grad_b2 * (b1 - sigma_phi), axis=-1, keepdims=True) 
      c2 = alpha * tf.reduce_sum(grad_b1 * (b2 - sigma_phi), axis=-1, keepdims=True) 
      dlog_q1 = b1 - sigma_phi
      dlog_q2 = b2 - sigma_phi

      grad_avg = 0.5 * (grad_b1 + grad_b2)
      global_corr = alpha * (grad_avg * sigma_phi * (1. - sigma_phi)) 
      diffs = f1 + c1 - (f2 + c2)
      layer_grad = 0.5 * ( diffs * dlog_q1 - diffs * dlog_q2 ) - global_corr


    elif grad_type.startswith('discrete_stein'):
      u1 = u_noise
      u2 = _sample_uniform_variables(sample_shape=tf.shape(encoder_logits))
      K = 2

      with tf.GradientTape(persistent=True) as tape:
        b1, f1 = sample_from_u(u1, tape)
        b2, f2 = sample_from_u(u2, tape)
        f_b = tf.squeeze(tf.concat([f1[None, ...], f2[None, ...]], axis=0), axis=-1)
        fb_sum = tf.reduce_sum(tf.reduce_mean(f_b, axis=0))
        b = tf.concat([b1[None, ...], b2[None, ...]], axis=0)

      grad_b1 = tape.gradient(fb_sum, b1)
      grad_b2 = tape.gradient(fb_sum, b2)
      grad_b = tf.concat([grad_b1[None, ...], grad_b2[None, ...]], axis=0)
      del tape
      theta = sigma_phi

      # b: [K, bs, D]
      nbr_mask = tf.eye(tf.shape(theta)[-1], dtype=theta.dtype)
      # nbrs: [K, bs, D, D]
      nbrs = nbr_mask * (1. - b[..., None, :]) + (1. - nbr_mask) * b[..., None, :]

      if 'output' in grad_type:
        def get_avg(center):
          f_theta = tf.repeat(f_b[None, ...], repeats=K, axis=0)   #(tf.reduce_sum(f_b, axis=0) - f_b) / (K - 1)
          # prod_b_expand_grad_b: [K, K, bs]
          prod_b_expand_grad_b = tf.reduce_sum(grad_b * b[:, None, ...], axis=-1)
          # prod_nbrs_expand_grad_b: [K, K, bs, D]
          prod_nbrs_expand_grad_b = tf.reduce_sum(grad_b[..., None, :] * nbrs[:, None, ...], axis=-1)
          # prod_b_grad_b: [K, 1, bs]
          prod_b_grad_b = tf.reduce_sum(grad_b * center, axis=-1)
          # f_nbrs_lin: [K, K, bs, D, 1]
          f_nbrs_lin = (prod_nbrs_expand_grad_b - prod_b_grad_b[..., None])[..., None]
          # f_x_lin: [K, K, bs, 1, 1]
          f_x_lin = (prod_b_expand_grad_b - prod_b_grad_b)[..., None, None]
          mask = tf.math.logical_not(tf.eye(K, dtype=tf.bool))
          f_nbrs_lin = tf.reshape(tf.boolean_mask(f_nbrs_lin, mask), tf.concat([[K, K - 1], tf.shape(f_nbrs_lin)[2:]], axis=0))
          f_x_lin = tf.reshape(tf.boolean_mask(f_x_lin, mask), tf.concat([[K, K - 1], tf.shape(f_x_lin)[2:]], axis=0))
          f_theta = tf.reshape(tf.boolean_mask(f_theta, mask), tf.concat([[K, K - 1], tf.shape(f_theta)[2:]], axis=0))
          # tf.print(tf.shape(f_nbrs_lin))
          return f_theta, f_nbrs_lin, f_x_lin

        if 'mu' in grad_type:
          raise NotImplementedError()
        elif 'oldavg' in grad_type:
          # f_theta: [K, 1, bs]
          f_theta, f_nbrs_lin, f_x_lin = get_avg(theta)
        elif 'jointavg' in grad_type:
          _, f_nbrs_lin_old, f_x_lin_old = get_avg(theta)
          f_theta, f_nbrs_lin_new, f_x_lin_new = get_avg(b)

          f_nbrs_lin = tf.concat([f_nbrs_lin_old, f_nbrs_lin_new], axis=-1)
          f_x_lin = tf.concat([f_x_lin_old, f_x_lin_new], axis=-1)
        elif 'avg' in grad_type:
          f_theta, f_nbrs_lin, f_x_lin = get_avg(b)
        else:
          raise NotImplementedError()

        f_nbrs_lin = tf.stop_gradient(f_nbrs_lin)
        g_nbrs = self.control_nn[start_layer](f_nbrs_lin, tf.stop_gradient(f_theta[..., None, None]))
        g_nbrs = tf.reduce_mean(g_nbrs, axis=1)

        f_x_lin = tf.stop_gradient(f_x_lin)

        g_x = self.control_nn[start_layer](f_x_lin, tf.stop_gradient(f_theta[..., None, None]))
        g_x = tf.squeeze(tf.reduce_mean(g_x, axis=1), axis=-2)
      else:
        if 'mu' in grad_type:
          raise NotImplementedError()
        elif 'oldavg' in grad_type:
          f_theta = (tf.reduce_sum(f_b, axis=0) - f_b) / (K - 1)
          # grad_theta: [1, bs, D]
          grad_theta = (tf.reduce_sum(grad_b, axis=0) - grad_b) / (K - 1)
          # f_nbrs_lin: [K, bs, D, 1]
          f_nbrs_lin = tf.reduce_sum(grad_theta[..., None, :] * (nbrs - theta[:, None, :]), axis=-1)[..., None]
          # f_x_lin: [K, bs, 1, 1]
          f_x_lin = tf.reduce_sum(grad_theta * (b - theta), axis=-1)[..., None, None]
        elif 'jointavg' in grad_type:
          f_theta = (tf.reduce_sum(f_b, axis=0) - f_b) / (K - 1)
          # grad_theta: [1, bs, D]
          grad_theta = (tf.reduce_sum(grad_b, axis=0) - grad_b) / (K - 1)
          # f_nbrs_lin: [K, bs, D, 1]
          f_nbrs_lin_old = tf.reduce_sum(grad_theta[..., None, :] * (nbrs - theta[:, None, :]), axis=-1)[..., None]
          # f_x_lin: [K, bs, 1, 1]
          f_x_lin_old = tf.reduce_sum(grad_theta * (b - theta), axis=-1)[..., None, None]

          loo_avg_grad_b = (tf.reduce_sum(grad_b, axis=0) - grad_b) / (K - 1)
          prod_b_grad_b = tf.reduce_sum(grad_b * b, axis=-1)
          loo_avg_prod_b_grad_b = tf.reduce_sum(prod_b_grad_b, axis=0) - prod_b_grad_b
          # f_nbrs_lin: [K, bs, D, 1]
          f_nbrs_lin_new = (tf.reduce_sum(loo_avg_grad_b[..., None, :] * nbrs, axis=-1) - loo_avg_prod_b_grad_b[..., None])[..., None]
          # f_x_lin: [K, bs, 1, 1]
          f_x_lin_new = (tf.reduce_sum(loo_avg_grad_b * b, axis=-1) - loo_avg_prod_b_grad_b)[..., None, None]

          f_nbrs_lin = tf.concat([f_nbrs_lin_old, f_nbrs_lin_new], axis=-1)
          f_x_lin = tf.concat([f_x_lin_old, f_x_lin_new], axis=-1)
        elif 'avg' in grad_type:
          # grad_theta: [K, bs, D]
          loo_avg_grad_b = (tf.reduce_sum(grad_b, axis=0) - grad_b) / (K - 1)
          prod_b_grad_b = tf.reduce_sum(grad_b * b, axis=-1)
          loo_avg_prod_b_grad_b = tf.reduce_sum(prod_b_grad_b, axis=0) - prod_b_grad_b
          # f_theta: [K, bs]
          f_theta = (tf.reduce_sum(f_b, axis=0) - f_b) / (K - 1)
          # f_nbrs_lin: [K, bs, D, 1]
          f_nbrs_lin = (tf.reduce_sum(loo_avg_grad_b[..., None, :] * nbrs, axis=-1) - loo_avg_prod_b_grad_b[..., None])[..., None]
          # f_x_lin: [K, bs, 1, 1]
          f_x_lin = (tf.reduce_sum(loo_avg_grad_b * b, axis=-1) - loo_avg_prod_b_grad_b)[..., None, None]
          #tf.reduce_sum(grad_b[None, ...] * (b[:, None, ...] - b[None, ...]), axis=(1, -1))[..., None, None] / (K - 1)
        else:
          raise NotImplementedError()

        f_nbrs_lin = tf.stop_gradient(f_nbrs_lin)
        g_nbrs = self.control_nn[start_layer](f_nbrs_lin, tf.stop_gradient(f_theta[..., None, None]))
        f_x_lin = tf.stop_gradient(f_x_lin)

        g_x = tf.squeeze(self.control_nn[start_layer](f_x_lin, tf.stop_gradient(f_theta[..., None, None])), axis=-2)
      # b: [K, bs, D]
      # theta: [bs, D]
      # dlog_qx: [K, bs, D]
      dlog_qx = b - theta
      # dlog_q_nbrs: [K, bs, D, D]
      dlog_q_nbrs = nbrs - theta[:, None, :]
      # double_cv: [K]
      # w1: [K, bs, D]
      w1 = theta * (1. - b) + (1. - theta) * b
      w2 = theta * b + (1. - theta) * (1. - b) - 1.
      double_cv = tf.reduce_mean(w1 * g_nbrs[..., 0] + w2 * g_x[..., :1], axis=-1)
      # cv: [K, bs]
      fb_vr = f_b + self.alpha[start_layer] * double_cv
      # fb_vr_not_k: [K, bs]
      fb_vr_not_k = (tf.reduce_sum(fb_vr, axis=0) - fb_vr) / (K - 1)
      # cv: [K, bs, D]
      cv = tf.reduce_mean(w1[..., None] * g_nbrs[..., 1:] * dlog_q_nbrs + w2[..., None] * (g_x[..., 1:] * dlog_qx)[..., None, :], axis=-2)
      layer_grad = tf.reduce_mean((f_b - fb_vr_not_k)[..., None] * dlog_qx + self.beta[start_layer] * cv, axis=0)

    else:
      raise NotImplementedError(f'Gradient type {grad_type} is not supported.')

    return layer_grad

  def get_multilayer_relax_parameters(
      self,
      sample_list,
      start_layer=0,
      temperature=None,
      scaling_factor=None):

    # the sample list contains the input and samples of hidden states
    # [x, b[1], b[2], ..., b[l]] where l is num_layers.
    if temperature is None:
      temperature = tf.math.exp(self.log_temperature_variable)
    if scaling_factor is None:
      scaling_factor = self.scaling_variable
    # [batch, hidden_units]
    encoder_logits = self.encoder_list[start_layer].get_logits(
        input_tensor=sample_list[start_layer])

    # returned uniform_noise would be of the shape
    # [batch x 2, event_dim].
    uniform_noise = _sample_uniform_variables(
        sample_shape=tf.shape(encoder_logits),
        nfold=2)
    # u_noise and v_noise are both of [batch, event_dim].
    u_noise, v_noise = tf.split(uniform_noise, num_or_size_splits=2, axis=0)

    theta = tf.math.sigmoid(encoder_logits)
    z = encoder_logits + logit_func(u_noise)
    b_sample = tf.cast(z > 0, tf.float32)

    v_prime = (b_sample * (v_noise * theta + 1 - theta)
               + (1 - b_sample) * v_noise * (1 - theta))
    # z_tilde ~ p(z | b)
    z_tilde = encoder_logits + logit_func(v_prime)

    sample_list_b = sample_list[:start_layer+1]
    sample_list_b.append(b_sample)
    sample_list_b = self.get_multilayer_bernoulli_sample(
        sample_list_b, resampled_layer=start_layer+1)

    elbo = self.get_elbo(sample_list_b[0], sample_list_b[1:])
    control_variate = self.get_multilayer_relax_control_variate(
        sample_list_b, z,
        temperature=temperature,
        scaling_factor=scaling_factor,
        resampled_layer_idx=start_layer+1)
    conditional_control = self.get_multilayer_relax_control_variate(
        sample_list_b, z_tilde,
        temperature=temperature,
        scaling_factor=scaling_factor,
        resampled_layer_idx=start_layer+1)

    log_q = tfd.Bernoulli(logits=encoder_logits).log_prob(b_sample)
    return elbo, control_variate, conditional_control, log_q

  def get_multilayer_relax_control_variate(
      self,
      sample_list,
      z_sample,
      temperature,
      scaling_factor,
      resampled_layer_idx):
    temp_sample_list = sample_list[:resampled_layer_idx]
    temp_sample_list.append(tf.math.sigmoid(z_sample/temperature))
    temp_sample_list = self.get_multilayer_bernoulli_sample(
        temp_sample_list, resampled_layer=resampled_layer_idx)
    control_value = (
        scaling_factor *
        self.get_elbo(temp_sample_list[0], temp_sample_list[1:]))
    if self.control_nn is not None:
      # concatenate the input tensor of the ith layer, the z generated
      # from the logits output by the ith layer. Here ith layer is labeled
      # by resampled_layer_idx.
      control_nn_input = tf.concat(
          (temp_sample_list[resampled_layer_idx-1], z_sample), axis=-1)
      control_value += (
          scaling_factor
          * tf.squeeze(self.control_nn[resampled_layer_idx-1](control_nn_input),
                       axis=-1))
    return control_value

  def get_multilayer_relax_loss(
      self,
      input_batch,
      temperature=None,
      scaling_factor=None):
    sample_list = self.multilayer_call(input_tensor=input_batch)[1]
    # elbo, control_variate, conditional_control should be of [batch_size]
    # log_q is of [batch_size, event_dim]
    reparam_loss = []
    learning_signal = []
    log_q = []

    for layer_idx in range(self.num_layers):
      elbo_i, control_variate_i, conditional_control_i, log_q_i = (
          self.get_multilayer_relax_parameters(
              sample_list,
              start_layer=layer_idx,
              temperature=temperature,
              scaling_factor=scaling_factor))
      reparam_loss_i = -1. * (control_variate_i - conditional_control_i)
      learning_signal_i = -1. * (elbo_i - conditional_control_i)
      # [batch_size, hidden_size]
      learning_signal_i = tf.tile(
          tf.expand_dims(learning_signal_i, axis=-1),
          [1, tf.shape(log_q_i)[-1]])

      reparam_loss.append(reparam_loss_i)
      learning_signal.append(learning_signal_i)
      log_q.append(log_q_i)

    # Define losses
    genmo_loss = -1. * elbo_i

    self.mean_learning_signal = tf.reduce_mean(
        tf.concat(learning_signal, axis=0))

    return genmo_loss, reparam_loss, learning_signal, log_q

  def _get_grad_variance(self, grad_variable, grad_sq_variable, grad_tensor):
    grad_variable.assign(grad_tensor)
    grad_sq_variable.assign(tf.square(grad_tensor))
    self.ema.apply([grad_variable, grad_sq_variable])

    # mean per component variance
    grad_var = (
        self.ema.average(grad_sq_variable)
        - tf.square(self.ema.average(grad_variable)))
    return grad_var

  def compute_grad_variance(
      self,
      grad_variables,
      grad_sq_variables,
      grad_tensors):
    # In order to use `tf.train.ExponentialMovingAverage`, one has to
    # use `tf.Variable`.
    grad_var = [
        tf.reshape(self._get_grad_variance(*g), [-1])
        for g in zip(grad_variables, grad_sq_variables, grad_tensors)]
    return tf.reduce_mean(tf.concat(grad_var, axis=0))

  @property
  def encoder_vars(self):
    if self.num_layers == 1:
      return self.encoder.trainable_variables
    elif self.num_layers > 1:
      enc_vars = []
      for enc in self.encoder_list:
        enc_vars.extend(enc.trainable_variables)
      return enc_vars

  @property
  def decoder_vars(self):
    if self.num_layers == 1:
      return self.decoder.trainable_variables
    elif self.num_layers > 1:
      dec_vars = []
      for dec in self.decoder_list:
        dec_vars.extend(dec.trainable_variables)
      return dec_vars

  @property
  def prior_vars(self):
    return self.prior_dist.trainable_variables
