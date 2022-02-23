# coding=utf-8
#
# Modifications Copyright 2022 by Jiaxin Shi, Yuhao Zhou, Jessica Hwang, Michalis Titsias, Lester Mackey
# https://arxiv.org/abs/2202.09497
# 
# Modifications Copyright 2021 by Michalis Titsias, Jiaxin Shi
# from https://github.com/alekdimi/arms
# and https://github.com/google-research/google-research/tree/master/disarm/binary
#
# Copyright 2021 The Google Research Authors.
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
               name='binarynet'):

    super().__init__(name=name)
    assert len(activations) == len(hidden_sizes)

    num_layers = len(hidden_sizes)
    self.hidden_sizes = hidden_sizes
    self.activations = activations
    self.networks = keras.Sequential()

    for i in range(num_layers-1):
      self.networks.add(
          keras.layers.Dense(units=hidden_sizes[i], activation=activations[i])) #kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234)))

    self.networks.add(
        keras.layers.Dense(units=hidden_sizes[-1], activation=activations[-1])) #kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1234)))

  def call(self, input_tensor):
    return self.networks(input_tensor)


class DiscreteVAE(tf.keras.Model):
  """Discrete VAE as described in ARM, (Yin and Zhou (2019))."""

  def __init__(self,
               encoder,
               decoder,
               prior_logits,
               likelihood='gaussian',
               grad_type='disarm',
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
      obs_dim = decoder[-1].hidden_sizes[-1]
    else:
      # for single layer discrete VAE
      self.encoder = encoder[0]
      self.decoder = decoder[0]
      obs_dim = self.decoder.hidden_sizes[-1]

    self.shared_randomness = shared_randomness
    self.prior_logits = prior_logits
    self.likelihood = likelihood
    self.prior_dist = tfd.Bernoulli(logits=self.prior_logits)
    self.grad_type = grad_type.lower()

    if self.likelihood == "gaussian":
      self.obs_var_base = tf.Variable(
        [-0.5 for i in range(obs_dim)],
        trainable=True,
        name='obs_var_base',
        dtype=tf.float32
      )

    # used for variance of gradients estiamations.
    self.ema = tf.train.ExponentialMovingAverage(0.999)
    self.eta_ema = tf.train.ExponentialMovingAverage(0.999)

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
        0.,
        trainable=True,
        name="alpha",
        dtype=tf.float32)
    elif self.grad_type.startswith('discrete_stein'):
      self.alpha = tf.Variable(
        0.,
        trainable=True,
        name="alpha",
        dtype=tf.float32)
      self.beta = tf.Variable(
        0.,
        trainable=True,
        name="beta",
        dtype=tf.float32)
      self.gamma = tf.Variable(
        0.,
        trainable=True,
        name="gamma",
        dtype=tf.float32)
      self.control_nn = control_nn

  def call(self, input_tensor, hidden_samples=None):
    if self.num_layers > 1:
      return self.multilayer_call(input_tensor, sample_list=hidden_samples)

  def sample_latent(self, input_tensor, K):
    eta = self.encoder(input_tensor)
    theta = tf.sigmoid(eta)
    bs = tf.shape(eta)[0]
    D = tf.shape(eta)[1]
    # u = tf.random.uniform([K, bs, D], dtype=tf.float32, seed=1234)
    u = tf.random.uniform([K, bs, D], dtype=tf.float32)
    b = tf.cast(u > 1. - theta, tf.float32)
    return b, eta, theta, u

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

  def get_loss(self, b, eta, x):
    # b: [K, bs, z_dim]
    # eta: [bs, z_dim]
    # x: [bs, x_dim]
    # x_var: [x_dim]
    # x_mean: [K, bs, x_dim]
    x_mean = self.decoder(b)
    # log_px_given_b: [K, bs]
    if self.likelihood == "gaussian":
      obs_var = tf.nn.softplus(self.obs_var_base)
      obs_var = tf.maximum(obs_var, 0.001)
      log_px_given_b = tf.reduce_sum(-0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(obs_var) - (x - x_mean)**2 / (2 * obs_var), axis=-1)
    else:
      dist = tfd.Bernoulli(logits=x_mean)
      log_px_given_b = tf.reduce_sum(dist.log_prob(x), axis=-1)
    # log_pb = -np.log(2) * b.shape[-1]
    log_pb = tf.reduce_sum(self.prior_dist.log_prob(b), axis=-1)
    q_dist = tfd.Bernoulli(logits=eta)
    # log_qb: [K, bs]
    log_qb = tf.reduce_sum(q_dist.log_prob(b), axis=-1)
    # log_qb = tf.reduce_sum(eta * b - tf.nn.softplus(eta), axis=-1)
    elbo = log_px_given_b + log_pb - log_qb
    return -1. * elbo

  def get_grad_estimation(
      self, input_tensor, f_b, b, eta, theta, u, grad_type=None, grad_b=None): #, extra_params='none'):
    K = b.shape[0]
    if grad_type is None:
      grad_type = self.grad_type

    if grad_type == 'disarm':
      # u_half: [K // 2, bs, D]
      u_half = u[:(K // 2), ...]
      # b_half: [K // 2, bs, D]
      b_half = b[:(K // 2), ...]
      # b_anti: [K // 2, D]
      b_anti = tf.cast(u_half < theta, tf.float32)
      # f_b_half: [K // 2, bs]
      f_b_half = f_b[:(K // 2), :]
      # f_b_anti: [K // 2, bs]
      f_b_anti = self.get_loss(b_anti, eta, input_tensor)
      # dlog_q: [K // 2, bs, D]
      dlog_q = (-1.)**b_anti * tf.cast(b_half != b_anti, tf.float32) * tf.sigmoid(tf.abs(eta))
      eta_grads = 0.5 * tf.reduce_mean((f_b_half - f_b_anti)[..., None] * dlog_q, axis=0)
      # u_noise = _sample_uniform_variables(
      #     sample_shape=tf.shape(eta),
      #     nfold=1)
      # sigma_abs_phi = tf.math.sigmoid(tf.math.abs(eta))
      # b1 = tf.cast(u_noise > 1. - theta, tf.float32)
      # b2 = tf.cast(u_noise < theta, tf.float32)
      # f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      # f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      # # the factor is I(b1+b2=1) * (-1)**b2 * sigma(|phi|)
      # disarm_factor = ((1. - b1) * (b2) + b1 * (1. - b2)) * (-1.)**b2
      # disarm_factor *= sigma_abs_phi
      # layer_grad = 0.5 * (f1 - f2) * disarm_factor

    elif grad_type == 'arms':
      encoder_shape = tf.shape(eta)
      batch_size, num_logits = encoder_shape[0], encoder_shape[1]

      p = tf.reshape(theta, [1, batch_size, num_logits])

      e = -tf.math.log(u)
      d = e / tf.reduce_sum(e, axis=0, keepdims=True)
      u_copula = tf.pow(1 - d, K - 1)

      p05 = tf.cast(p < 0.5, tf.float32)
      u_corr = u_copula * p05 + (1 - u_copula) * (1 - p05)

      def bivariate(p):
        term = 2 * tf.pow(p, 1 / (K - 1)) - 1
        return tf.pow(tf.maximum(term, 0), K - 1)

      j1 = bivariate(p)
      j2 = 2 * p - 1 + bivariate(1 - p)
      joint = j1 * p05 + j2 * (1 - p05)
      debias = p * (1 - p) / (p - joint + 1e-6)

      b = tf.cast(u_corr < p, tf.float32)
      # b_flat = tf.reshape(b, [K * batch_size, num_logits])
      # tiled_input_tensor = tf.tile(tf.reshape(input_tensor, [1, batch_size, -1]), [K, 1, 1,])
      # flat_input_tensor = tf.reshape(tiled_input_tensor, [K * batch_size, -1])
      # f = tf.reshape(self.get_elbo(flat_input_tensor, b_flat), [K, batch_size, 1])
      f = self.get_loss(b, eta, input_tensor)
      fmean = tf.reduce_mean(f, axis=0, keepdims=True)
      eta_grads = tf.reduce_mean((f - fmean)[..., None] * (b - p) * K / (K - 1) * debias, axis=0)

    elif grad_type == 'reinforce':
      # f_b: [K, bs]
      # b: [K, bs, D]
      # eta: [bs, D]
      # dlog_q: [K, bs, D]
      dlog_q = b - theta
      eta_grads = tf.reduce_mean(f_b[..., None] * dlog_q, axis=0)
      # u_noise = _sample_uniform_variables(
      #     sample_shape=tf.shape(eta),
      #     nfold=2)
      # u1, u2 = tf.split(u_noise, num_or_size_splits=2, axis=0)
      # b1 = tf.cast(u1 < theta, tf.float32)
      # b2 = tf.cast(u2 < theta, tf.float32)
      # f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      # f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      # layer_grad = 0.5 * (f1 * (b1 - theta) + f2 * (b2 - theta))

    elif grad_type == 'reinforce_loo':
      # f_not_k: [K, bs]
      f_not_k = tf.reduce_sum(f_b, axis=0) - f_b
      fk_minus_avg_f_not_k = f_b - f_not_k / (K - 1)
      # b: [K, bs, D]
      # eta: [bs, D]
      # dlog_q: [K, bs, D]
      dlog_q = b - theta
      eta_grads = tf.reduce_mean(fk_minus_avg_f_not_k[..., None] * dlog_q, axis=0)
      # u_noise = _sample_uniform_variables(
      #     sample_shape=tf.shape(eta),
      #     nfold=2)
      # u1, u2 = tf.split(u_noise, num_or_size_splits=2, axis=0)
      # b1 = tf.cast(u1 < theta, tf.float32)
      # b2 = tf.cast(u2 < theta, tf.float32)
      # f1 = self.get_elbo(input_tensor, b1)[:, tf.newaxis]
      # f2 = self.get_elbo(input_tensor, b2)[:, tf.newaxis]
      # layer_grad = 0.5 * ((f1 - f2) * (b1 - theta)
      #                     + (f2 - f1) * (b2 - theta))

    elif grad_type == "double_cv":
      if K == 4:
        g0 = (1.0 / 3.0 )*(grad_b[1, ...] + grad_b[2, ...] + grad_b[3, ...])
        g1 = (1.0 / 3.0 )*(grad_b[0, ...] + grad_b[2, ...] + grad_b[3, ...])
        g2 = (1.0 / 3.0 )*(grad_b[0, ...] + grad_b[1, ...] + grad_b[3, ...])
        g3 = (1.0 / 3.0 )*(grad_b[0, ...] + grad_b[1, ...] + grad_b[2, ...])

        b0 = self.alpha * tf.reduce_sum(g0 * (b[0, ...] - theta), axis=-1, keepdims=True)
        b1 = self.alpha * tf.reduce_sum(g1 * (b[1, ...] - theta), axis=-1, keepdims=True)
        b2 = self.alpha * tf.reduce_sum(g2 * (b[2, ...] - theta), axis=-1, keepdims=True)
        b3 = self.alpha * tf.reduce_sum(g3 * (b[3, ...] - theta), axis=-1, keepdims=True)

        dlog_q = b - theta
        grad_avg = 0.25 * (grad_b[0, ...] + grad_b[1, ...] + grad_b[2, ...] + grad_b[3, ...])
        global_corr = self.alpha * (grad_avg * theta * (1. - theta))
        diff0 = f_b[0, :, None] + b0 - (1.0 / 3.0 )*(f_b[1, :, None] + f_b[2, :, None] + f_b[3, :, None] + b1 + b2 + b3)
        diff1 = f_b[1, :, None] + b1 - (1.0 / 3.0 )*(f_b[0, :, None] + f_b[2, :, None] + f_b[3, :, None] + b0 + b2 + b3)
        diff2 = f_b[2, :, None] + b2 - (1.0 / 3.0 )*(f_b[0, :, None] + f_b[1, :, None] + f_b[3, :, None] + b0 + b1 + b3)
        diff3 = f_b[3, :, None] + b3 - (1.0 / 3.0 )*(f_b[0, :, None] + f_b[1, :, None] + f_b[2, :, None] + b0 + b1 + b2)

        eta_grads = 0.25*(diff0 * dlog_q[0, :] + diff1 * dlog_q[1, :] + diff2 * dlog_q[2, :] + diff3 * dlog_q[3, :]) - global_corr
      elif K == 2:
        b1 = self.alpha * tf.reduce_sum(grad_b[1, ...] * (b[0, ...] - theta), axis=-1, keepdims=True) 
        # b2 = self.alpha * tf.reduce_sum(grad_b[0, ...] * (b[1, ...] - theta), axis=-1, keepdims=True)
        c1 = self.alpha * tf.reduce_sum(grad_b[0, ...] * (b[1, ...] - theta), axis=-1, keepdims=True) 
        # c2 = self.alpha * tf.reduce_sum(grad_b[1, ...] * (b[0, ...] - theta), axis=-1, keepdims=True)
        dlog_q = b - theta
        grad_avg = 0.5 * (grad_b[1, ...] + grad_b[0, ...])
        global_corr = self.alpha * (grad_avg * theta * (1. - theta)) 
        diffs = f_b[0, :, None] + b1 - (f_b[1, :, None] + c1)
        eta_grads = 0.5 * ( diffs * dlog_q[0,:] - diffs * dlog_q[1,:] ) - global_corr
      else:
        # grad_b: [K, bs, D]
        avg_grad_b = tf.reduce_mean(grad_b, axis=0)
        loo_avg_grad_b = (tf.reduce_sum(grad_b, axis=0) - grad_b) / (K - 1)

        loo_vr = f_b + self.alpha * tf.reduce_sum((b - theta) * loo_avg_grad_b, axis=-1)
        loo_vr = (tf.reduce_sum(loo_vr, axis=0) - loo_vr) / (K - 1)
        # cv: [K, bs]
        cv = self.alpha * tf.reduce_sum(loo_avg_grad_b * (b - theta), axis=-1)
        dlog_q = b - theta
        eta_grads = tf.reduce_mean((f_b - loo_vr + cv)[..., None] * dlog_q, axis=0) - self.alpha * avg_grad_b * theta * (1. - theta)

    elif grad_type.startswith('discrete_stein'):
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
          if 'fast' in grad_type:
            indices = list(range(1, K + 1))
            indices[-1] = 0
            mask = tf.gather(tf.eye(K, dtype=tf.bool), indices, axis=0)
            mask_size = 1
          else:
            mask = tf.math.logical_not(tf.eye(K, dtype=tf.bool))
            mask_size = K - 1
          f_nbrs_lin = tf.reshape(tf.boolean_mask(f_nbrs_lin, mask), tf.concat([[K, mask_size], tf.shape(f_nbrs_lin)[2:]], axis=0))
          f_x_lin = tf.reshape(tf.boolean_mask(f_x_lin, mask), tf.concat([[K, mask_size], tf.shape(f_x_lin)[2:]], axis=0))
          f_theta = tf.reshape(tf.boolean_mask(f_theta, mask), tf.concat([[K, mask_size], tf.shape(f_theta)[2:]], axis=0))
          # tf.print(tf.shape(f_nbrs_lin))
          return f_theta, f_nbrs_lin, f_x_lin

        if 'mu' in grad_type:
          with tf.GradientTape() as tape:
            # theta: [bs, D]
            tape.watch(theta)
            # f_theta: [1, bs]
            f_theta = self.get_loss(theta[None, ...], eta, input_tensor)
            f_theta_sum = tf.reduce_sum(f_theta)

          # grad_theta: [1, bs, D]
          grad_theta = tape.gradient(f_theta_sum, theta)[None, ...]
          # f_nbrs_lin: [K, bs, D, 1]
          f_nbrs_lin = tf.reduce_sum(grad_theta[..., None, :] * (nbrs - theta[:, None, :]), axis=-1)[..., None]
          # f_x_lin: [K, bs, 1, 1]
          f_x_lin = tf.reduce_sum(grad_theta * (b - theta), axis=-1)[..., None, None]
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
        g_nbrs = self.control_nn(f_nbrs_lin, tf.stop_gradient(f_theta[..., None, None]))
        g_nbrs = tf.reduce_mean(g_nbrs, axis=1)
        f_x_lin = tf.stop_gradient(f_x_lin)
        g_x = self.control_nn(f_x_lin, tf.stop_gradient(f_theta[..., None, None]))
        g_x = tf.squeeze(tf.reduce_mean(g_x, axis=1), axis=-2)
      else:
        if 'mu' in grad_type:
          with tf.GradientTape() as tape:
            # theta: [bs, D]
            tape.watch(theta)
            # f_theta: [1, bs]
            f_theta = self.get_loss(theta[None, ...], eta, input_tensor)
            f_theta_sum = tf.reduce_sum(f_theta)

          # grad_theta: [1, bs, D]
          grad_theta = tape.gradient(f_theta_sum, theta)[None, ...]
          # f_nbrs_lin: [K, bs, D, 1]
          f_nbrs_lin = tf.reduce_sum(grad_theta[..., None, :] * (nbrs - theta[:, None, :]), axis=-1)[..., None]
          # f_x_lin: [K, bs, 1, 1]
          f_x_lin = tf.reduce_sum(grad_theta * (b - theta), axis=-1)[..., None, None]
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
        # g_nbrs: [K, bs, D, 2]
        g_nbrs = self.control_nn(f_nbrs_lin, tf.stop_gradient(f_theta[..., None, None]))
        f_x_lin = tf.stop_gradient(f_x_lin)
        # g_x: [K, bs, 2]
        g_x = tf.squeeze(self.control_nn(f_x_lin, tf.stop_gradient(f_theta[..., None, None])), axis=-2)
      # b: [K, bs, D]
      # theta: [bs, D]
      # dlog_qx: [K, bs, D]
      dlog_qx = b - theta
      # dlog_q_nbrs: [K, bs, D, D]
      dlog_q_nbrs = nbrs - theta[:, None, :]
      # double_cv: [K]
      # w1: [K, bs, D]
      if 'MPF' in grad_type:
        w1 = theta * (1. - b) + (1. - theta) * b   # q(y | x_{-i})
        w_ratio = tf.sqrt(w1 / (1. - w1 + 1.0e-3))   # q(y) / q(x)
        double_cv = tf.reduce_sum(w_ratio * (g_nbrs[..., 0] - g_x[..., :1]), axis=-1)
        # cv: [K, bs, D]
        cv = tf.reduce_sum(w_ratio[..., None] * (g_nbrs[..., 1:] * dlog_q_nbrs - (g_x[..., 1:] * dlog_qx)[..., None, :]), axis=-2)
      elif 'DIFF' in grad_type:
        # TODO: CHECK THIS CODE
        w1 = theta * (1. - b) + (1. - theta) * b   # q(y | x_{-i})
        w_ratio = w1 / (1. - w1 + 1.0e-3)   # q(y) / q(x)
        double_cv = tf.reduce_mean(g_nbrs[..., 0] - w_ratio * g_x[..., :1], axis=-1)
        # cv: [K, bs, D]
        cv = tf.reduce_mean(g_nbrs[..., 1:] * dlog_q_nbrs - w_ratio[..., None] * (g_x[..., 1:] * dlog_qx)[..., None, :], axis=-2)
      else:
        w1 = theta * (1. - b) + (1. - theta) * b
        w2 = theta * b + (1. - theta) * (1. - b) - 1.
        double_cv = tf.reduce_mean(w1 * g_nbrs[..., 0] + w2 * g_x[..., :1], axis=-1)
        # cv: [K, bs, D]
        cv = tf.reduce_mean(w1[..., None] * g_nbrs[..., 1:] * dlog_q_nbrs + w2[..., None] * (g_x[..., 1:] * dlog_qx)[..., None, :], axis=-2)
      # cv: [K, bs]
      fb_vr = f_b + self.alpha * double_cv
      # fb_vr_not_k: [K, bs]
      fb_vr_not_k = (tf.reduce_sum(fb_vr, axis=0) - fb_vr) / (K - 1)
      eta_grads = tf.reduce_mean((f_b - fb_vr_not_k)[..., None] * dlog_qx + self.beta * cv, axis=0)

    else:
      raise NotImplementedError(f'Gradient type {grad_type} is not supported.')
    return eta_grads

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

  def get_relax_parameters(
      self,
      input_tensor,
      temperature=None,
      scaling_factor=None):
    if temperature is None:
      temperature = tf.math.exp(self.log_temperature_variable)
    if scaling_factor is None:
      scaling_factor = self.scaling_variable

    # [batch, hidden_units]
    eta = self.encoder(input_tensor)
    theta = tf.sigmoid(eta)

    # returned uniform_noise would be of the shape
    # [batch x 2, event_dim].
    uniform_noise = _sample_uniform_variables(
        sample_shape=tf.shape(eta), nfold=2)
    # u_noise and v_noise are both of [batch, event_dim].
    u_noise, v_noise = tf.split(uniform_noise, num_or_size_splits=2, axis=0)

    z = eta + logit_func(u_noise)
    b_sample = tf.cast(z > 0, tf.float32)

    v_prime = (b_sample * (v_noise * theta + 1 - theta)
               + (1 - b_sample) * v_noise * (1 - theta))
    # z_tilde ~ p(z | b)
    z_tilde = eta + logit_func(v_prime)

    genmo_loss = tf.squeeze(self.get_loss(b_sample[None, ...], eta, input_tensor), axis=0)
    control_variate = self.get_relax_control_variate(
        input_tensor, z, eta,
        temperature=temperature, scaling_factor=scaling_factor)
    conditional_control = self.get_relax_control_variate(
        input_tensor, z_tilde, eta,
        temperature=temperature, scaling_factor=scaling_factor)

    log_q = tfd.Bernoulli(logits=eta).log_prob(b_sample)
    return genmo_loss, control_variate, conditional_control, log_q

  def get_relax_control_variate(self, input_tensor, z_sample, eta,
                                temperature, scaling_factor):
    relaxed_b = tf.math.sigmoid(z_sample/temperature)
    control_value = (
        -1. * scaling_factor *
        tf.squeeze(self.get_loss(relaxed_b[None, ...], eta, input_tensor), axis=0))
    if self.control_nn is not None:
      control_nn_input = tf.concat((input_tensor, z_sample), axis=-1)
      control_value += (scaling_factor
                        * tf.squeeze(self.control_nn(control_nn_input),
                                     axis=-1))
    return control_value

  def get_relax_loss(self, input_batch, temperature=None, scaling_factor=None):
    # genmo_loss, control_variate, conditional_control should be of [batch_size]
    # log_q is of [batch_size, event_dim]
    genmo_loss, control_variate, conditional_control, log_q = (
        self.get_relax_parameters(
            input_batch,
            temperature=temperature,
            scaling_factor=scaling_factor))

    reparam_loss = -1. * (control_variate - conditional_control)

    # [batch_size]
    learning_signal = genmo_loss + conditional_control
    self.mean_learning_signal = tf.reduce_mean(learning_signal)

    # [batch_size, hidden_size]
    learning_signal = tf.tile(
        tf.expand_dims(learning_signal, axis=-1),
        [1, tf.shape(log_q)[-1]])

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

  def _get_eta_grad_variance(self, grad_eta_variable, grad_eta_sq_variable, grad_tensor):
    grad_eta_variable.assign(grad_tensor)
    grad_eta_sq_variable.assign(tf.square(grad_tensor))
    self.eta_ema.apply([grad_eta_variable, grad_eta_sq_variable])

    # mean per component variance
    grad_var = (
        self.eta_ema.average(grad_eta_sq_variable)
        - tf.square(self.eta_ema.average(grad_eta_variable)))
    return grad_var

  def compute_eta_grad_variance(
      self,
      grad_eta_variable,
      grad_eta_sq_variable,
      grad_tensor):
    # In order to use `tf.train.ExponentialMovingAverage`, one has to
    # use `tf.Variable`.
    grad_var = self._get_eta_grad_variance(grad_eta_variable, grad_eta_sq_variable, grad_tensor)
    return tf.reduce_mean(grad_var)

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
    var_list = []
    if self.likelihood == "gaussian":
      var_list.append(self.obs_var_base)
    if self.num_layers == 1:
      var_list.extend(self.decoder.trainable_variables)
    else:
      for dec in self.decoder_list:
        var_list.extend(dec.trainable_variables)
    return var_list

  @property
  def prior_vars(self):
    return self.prior_dist.trainable_variables


class SteinControlNNSingleLayer(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.densex = keras.layers.Dense(100, dtype=tf.float32)
    self.densez = keras.layers.Dense(100, dtype=tf.float32)
    self.activation = keras.layers.LeakyReLU(alpha=0.3)
    self.dense2 = keras.layers.Dense(2, dtype=tf.float32)

  def call(self, x, z):
    hx = self.densex(x)
    hz = self.densez(z)
    y = self.dense2(self.activation(hx + hz))
    return y
