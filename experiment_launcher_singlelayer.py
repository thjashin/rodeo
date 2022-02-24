# Copyright 2022 by Jiaxin Shi, Yuhao Zhou, Jessica Hwang, Michalis Titsias, Lester Mackey
# https://arxiv.org/abs/2202.09497
#
# Copyright 2021 by Michalis Titsias, Jiaxin Shi 
# modified from https://github.com/alekdimi/arms 
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

from collections import defaultdict
import os
from datetime import datetime

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
import time

import tensorflow as tf
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow_probability as tfp

import dataset
import networks

tfd = tfp.distributions

layers = tf.keras.layers


flags.DEFINE_enum('dataset', 'dynamic_mnist',
                  ['static_mnist', 'dynamic_mnist', 'continuous_mnist',
                   'fashion_mnist', 'continuous_fashion', 'omniglot', 'continuous_omniglot'],
                  'Dataset to use.')
flags.DEFINE_string('data_dir', os.getenv('PT_DATA_DIR', './data/'), 
                  'Directory where data is stored.')
flags.DEFINE_float('genmo_lr', 1e-3, 
                   'Learning rate for decoder, Generation network.')
flags.DEFINE_float('infnet_lr', 1e-3, 
                   'Learning rate for encoder, Inference network.')
flags.DEFINE_float('prior_lr', 0.,
                   'Learning rate for prior variables.')
flags.DEFINE_float('hyper_lr', 1e-3, 
                   'Learning rate for hyperparameters in gradient estimators.')
flags.DEFINE_float('control_lr', 1e-3, 
                   'Learning rate for the control network in DiscreteStein gradient estimators.')
flags.DEFINE_integer('batch_size', 100, 'Training batch size.')
flags.DEFINE_integer('K', 2, 'Number of samples used gradient estimators.')
flags.DEFINE_integer('D', 200, 'Number of latent dimensions.')
flags.DEFINE_integer('num_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_enum('grad_type', 'double_cv', ['disarm', 'reinforce_loo', 'double_cv', 'relax', 'arms', 
                                             # without 'output':
                                             #    'mu': H(f(\mu), \nabla f(\mu) * (x - \mu))
                                             #    'avg': H( avg. of f, avg. of \nabla f(x_j) * (x - x_j))
                                             #    'oldavg': H( avg. of f, avg. of \nabla f(x_j) * (x - \mu))
                                             #    'jointavg': H( avg. of f, avg. of \nabla f(x_j) * (x - x_j), avg. of \nabla f(x_j) * (x - \mu))
                                             # with 'output', 
                                             #    'mu': H(f(\mu), \nabla f(\mu) * (x - \mu))
                                             #    'avg': avg. of H( f, \nabla f(x_j) * (x - x_j))
                                             #    'oldavg': avg. of H( f, \nabla f(x_j) * (x - \mu))
                                             #    'jointavg': avg. of H( f, \nabla f(x_j) * (x - x_j), avg. of \nabla f(x_j) * (x - \mu))
                                             'discrete_stein_MPF_oldavg', 'discrete_stein_MPF_jointavg', 'discrete_stein_MPF_mu', 'discrete_stein_MPF_avg', 
                                             'discrete_stein_DIFF_oldavg', 'discrete_stein_DIFF_jointavg', 'discrete_stein_DIFF_mu', 'discrete_stein_DIFF_avg', 
                                             'discrete_stein_output_oldavg', 'discrete_stein_output_jointavg',
                                             'discrete_stein_output_mu', 'discrete_stein_output_avg',
                                             'discrete_stein_output_fast_jointavg', 'discrete_stein_output_fast_avg',
                                             'discrete_stein_output_fast_oldavg',
                                             'discrete_stein_oldavg', 'discrete_stein_jointavg', 'discrete_stein_mu', 'discrete_stein_avg',
                                             ],
                  'Choice of gradient estimator.')
flags.DEFINE_string('encoder_type', 'nonlinear',
                    'Choice supported: linear, nonlinear')
flags.DEFINE_string('logdir', os.getenv('PT_OUTPUT_DIR', "./logs/" + datetime.now().strftime("%Y%m%d%H%M%S")),
                    'Directory for storing logs.')
flags.DEFINE_string('type_remap', "none", "re-define the grad_type on logdir")
flags.DEFINE_bool('verbose', False,
                  'Whether to turn on training result logging.')
flags.DEFINE_float('temperature', None,
                   'Temperature for RELAX estimator.')
flags.DEFINE_float('scaling_factor', None,
                   'Scaling factor for RELAX estimator.')
flags.DEFINE_bool('bias_check', False,
                  'Carry out bias check for RELAX and baseline')
flags.DEFINE_integer('seed', 1234, 'Global random seed.')
flags.DEFINE_bool('estimate_grad_basket', False,
                  'Estimate gradients for multiple estimators.')
flags.DEFINE_integer('num_eval_samples', 100,
                     'Number of samples for evaluation.')
flags.DEFINE_integer('num_train_samples', 1,
                     'Number of samples for evaluation.')
flags.DEFINE_bool('debug', False, 'Turn on debugging mode.')
FLAGS = flags.FLAGS


def process_batch_input(input_batch):
  input_batch = tf.reshape(input_batch, [tf.shape(input_batch)[0], -1])
  input_batch = tf.cast(input_batch, tf.float32)
  return input_batch


def initialize_grad_variables(target_variable_list):
  return [tf.Variable(tf.zeros(shape=i.shape)) for i in target_variable_list]


def estimate_gradients(input_batch, bvae_model, gradient_type, hyper_optimizer=None, control_optimizer=None):
  """Estimate gradient for inference and generation networks."""

  if gradient_type == 'relax':
    with tf.GradientTape(persistent=True) as tape:
      genmo_loss, reparam_loss, learning_signal, log_q = (
        bvae_model.get_relax_loss(
          input_batch,
          temperature=FLAGS.temperature,
          scaling_factor=FLAGS.scaling_factor))

    genmo_grads = tape.gradient(genmo_loss, bvae_model.decoder_vars)
    prior_grads = tape.gradient(genmo_loss, bvae_model.prior_vars)

    infnet_vars = bvae_model.encoder_vars
    infnet_grads_1 = tape.gradient(log_q, infnet_vars,
                                   output_gradients=learning_signal)
    infnet_grads_2 = tape.gradient(reparam_loss, infnet_vars)
    # infnet_grads_1/2 are list of tf.Tensors.
    infnet_grads = [infnet_grads_1[i] + infnet_grads_2[i]
                    for i in range(len(infnet_vars))]
    eta_grads = None

  else:
    with tf.GradientTape(persistent=True) as tape:
      b, eta, theta, u = bvae_model.sample_latent(input_batch, FLAGS.K)
      tape.watch(b)
      # f_b: [K, bs]
      f_b = bvae_model.get_loss(b, eta, input_batch)
      # loss function for generative model
      # genmo_loss = tf.reduce_mean(f_b)
      genmo_loss = tf.reduce_mean(f_b, axis=0)
      fb_sum = tf.reduce_sum(genmo_loss)

    genmo_grads, prior_grads = tape.gradient(genmo_loss, [bvae_model.decoder_vars, bvae_model.prior_vars])

    if gradient_type == "double_cv":
      grad_b = tape.gradient(fb_sum, b) * FLAGS.K
      with tf.GradientTape() as hyper_tape:
        eta_grads = bvae_model.get_grad_estimation(
          input_batch, f_b, b, eta, theta, u, grad_type=gradient_type, grad_b=grad_b)
        variance_loss = tf.reduce_mean(tf.square(eta_grads))
      if hyper_optimizer is not None:
        hyper_grads = hyper_tape.gradient(variance_loss, bvae_model.alpha)
        hyper_optimizer.apply_gradients([(hyper_grads, bvae_model.alpha)])

    elif gradient_type.startswith('discrete_stein'):
      grad_b = None
      if 'avg' in gradient_type or 'double' in gradient_type:
        # grad_b: [K, bs, D]
        grad_b = tape.gradient(fb_sum, b) * FLAGS.K

      with tf.GradientTape() as hyper_tape:
        eta_grads = bvae_model.get_grad_estimation(
          input_batch, f_b, b, eta, theta, u, grad_type=gradient_type, grad_b=grad_b) #, extra_params=FLAGS.discrete_stein)
        variance_loss = tf.reduce_mean(tf.square(eta_grads))
      if hyper_optimizer is not None or control_optimizer is not None:
        hyper_vars = [bvae_model.alpha, bvae_model.beta]
        var_list = bvae_model.control_nn.trainable_variables + hyper_vars
        hyper_grads = hyper_tape.gradient(variance_loss, var_list)
        if hyper_optimizer is not None:
          hyper_optimizer.apply_gradients(zip(hyper_grads[-2:], hyper_vars))
        if control_optimizer is not None:
          control_optimizer.apply_gradients(zip(hyper_grads[:-2], bvae_model.control_nn.trainable_variables))

    else:
      eta_grads = bvae_model.get_grad_estimation(
          input_batch, f_b, b, eta, theta, u, grad_type=gradient_type, grad_b=None)

    infnet_grads = tape.gradient(
        eta,
        bvae_model.encoder_vars,
        output_gradients=eta_grads)

  del tape
  return (genmo_grads, prior_grads, infnet_grads, eta_grads, genmo_loss)


@tf.function
def evaluate(model, tf_dataset, max_step=1000, num_eval_samples=None):
  """Evaluate the model."""
  if num_eval_samples:
    num_samples = num_eval_samples
  elif FLAGS.num_eval_samples:
    num_samples = FLAGS.num_eval_samples
  else:
    num_samples = FLAGS.K
  tf.print('Evaluate with samples: ', num_samples)
  loss = 0.
  n = 0.
  for batch in tf_dataset:
    batch = process_batch_input(batch)
    if n >= max_step:  # used for train_ds, which is a `repeat` dataset.
      break
    if num_samples > 1:
      b, eta, _, _ = model.sample_latent(batch, FLAGS.K)
      # elbo: [K, bs]
      elbo = -1. * model.get_loss(b, eta, batch)
      objectives = (tf.reduce_logsumexp(elbo, axis=0, keepdims=False) -
                    tf.math.log(tf.cast(tf.shape(elbo)[0], tf.float32)))
    else:
      b, eta, _, _ = model.sample_latent(batch, FLAGS.K)
      # elbo: [K, bs]
      objectives = -1. * model.get_loss(b, eta, batch)
    loss -= tf.reduce_mean(objectives)
    n += 1.
  return loss / n


def run_bias_check(model, batch, target_type, baseline_type):
  """Run bias check."""
  tf.print(f'Running a bias check comparing {target_type} and {baseline_type}.')
  mu = 0.
  s = 0.
  for step in range(1, int(1e6) + 1):
    diff = run_bias_check_step(
      batch,
      model,
      target_type=target_type,
      baseline_type=baseline_type)
    prev_mu = mu
    mu = mu + (diff - mu) / step
    s = s + (diff - mu) * (diff - prev_mu)

    if step % 1000 == 0:
      sigma = tf.math.sqrt(s / step)
      z_score = mu / (sigma / tf.math.sqrt(float(step)))
      tf.print(step, 'z_score: ', z_score, 'sigma: ', sigma)


@tf.function
def run_bias_check_step(
    train_batch_i,
    bvae_model,
    target_type='disarm',
    baseline_type='reinforce_loo'):
  """Run bias check for 1 batch."""
  input_batch = process_batch_input(train_batch_i)
  sample_size = FLAGS.K

  infnet_grads = estimate_gradients(
    input_batch, bvae_model, target_type, sample_size)[2]
  baseline_infnet_grads = estimate_gradients(
    input_batch, bvae_model, baseline_type, sample_size)[2]
  diff = tf.concat([tf.reshape(x - y, [-1])
                    for x, y in zip(infnet_grads, baseline_infnet_grads)],
                   axis=0)
  return tf.reduce_mean(diff)


def main(grad_type, test_run=False):
  seed = FLAGS.seed
  tf.random.set_seed(seed)
  np.random.seed(seed)
  batch_size = FLAGS.batch_size
  likelihood = "bernoulli"
  if FLAGS.dataset == 'static_mnist':
    train_ds, valid_ds, test_ds = dataset.get_static_mnist_batch(batch_size, seed)
    train_size = 50000
  elif FLAGS.dataset == 'dynamic_mnist':
    train_ds, valid_ds, test_ds = dataset.get_dynamic_mnist_batch(batch_size, seed)
    train_size = 50000
  elif FLAGS.dataset == 'continuous_mnist':
    train_ds, valid_ds, test_ds = dataset.get_continuous_mnist_batch(batch_size, seed)
    train_size = 50000
    likelihood = "gaussian"
  elif FLAGS.dataset == "continuous_fashion":
    train_ds, valid_ds, test_ds = dataset.get_continuous_mnist_batch(
      batch_size, seed, fashion_mnist=True)
    train_size = 50000
    likelihood = "gaussian"
  elif FLAGS.dataset == 'fashion_mnist':
    train_ds, valid_ds, test_ds = dataset.get_dynamic_mnist_batch(
      batch_size, seed, fashion_mnist=True)
    train_size = 50000
  elif FLAGS.dataset == "continuous_omniglot":
    train_ds, valid_ds, test_ds = dataset.get_continuous_omniglot_batch(batch_size, seed, FLAGS.data_dir)
    train_size = 23000
    likelihood = "gaussian"
  elif FLAGS.dataset == 'omniglot':
    train_ds, valid_ds, test_ds = dataset.get_omniglot_batch(batch_size, seed, FLAGS.data_dir)
    train_size = 23000

  @tf.function
  def train_one_step(
      train_batch_i,
      bvae_model,
      genmo_optimizer,
      infnet_optimizer,
      prior_optimizer,
      hyper_optimizer,
      control_optimizer,
      theta_optimizer,
      encoder_grad_variable,
      encoder_grad_sq_variable,
      eta_grad_variable,
      eta_grad_sq_variable,
      grad_variable_dict,
      grad_sq_variable_dict,
      grad_type):
    """Train Discrete VAE for 1 step."""
    metrics = {}
    input_batch = process_batch_input(train_batch_i)

    if grad_type == 'relax':
      with tf.GradientTape(persistent=True) as theta_tape:
        (genmo_grads, prior_grads, infnet_grads, eta_grads, genmo_loss) = estimate_gradients(
          input_batch, bvae_model, grad_type)
        infnet_grads_sq = [tf.square(grad_i) for grad_i in infnet_grads]

      # Update generative model
      genmo_vars = bvae_model.decoder_vars
      genmo_optimizer.apply_gradients(list(zip(genmo_grads, genmo_vars)))

      prior_vars = bvae_model.prior_vars
      prior_optimizer.apply_gradients(list(zip(prior_grads, prior_vars)))

      infnet_vars = bvae_model.encoder_vars
      infnet_optimizer.apply_gradients(list(zip(infnet_grads, infnet_vars)))

      theta_vars = []
      if bvae_model.control_nn:
        theta_vars.extend(bvae_model.control_nn.trainable_variables)
      if FLAGS.temperature is None:
        theta_vars.append(bvae_model.log_temperature_variable)
      if FLAGS.scaling_factor is None:
        theta_vars.append(bvae_model.scaling_variable)
      theta_grads = theta_tape.gradient(infnet_grads_sq, theta_vars)
      theta_optimizer.apply_gradients(zip(theta_grads, theta_vars))
      del theta_tape

      metrics['learning_signal'] = bvae_model.mean_learning_signal
      eta_grad_var = None

    else:
      (genmo_grads, prior_grads, infnet_grads, eta_grads, genmo_loss) = estimate_gradients(
        input_batch, bvae_model, grad_type, hyper_optimizer=hyper_optimizer, control_optimizer=control_optimizer)

      genmo_vars = bvae_model.decoder_vars
      genmo_optimizer.apply_gradients(list(zip(genmo_grads, genmo_vars)))

      prior_vars = bvae_model.prior_vars
      prior_optimizer.apply_gradients(list(zip(prior_grads, prior_vars)))

      infnet_vars = bvae_model.encoder_vars
      infnet_optimizer.apply_gradients(list(zip(infnet_grads, infnet_vars)))

      eta_grad_var = bvae_model.compute_eta_grad_variance(
        eta_grad_variable, eta_grad_sq_variable, eta_grads)

    batch_size_sq = tf.cast(FLAGS.batch_size * FLAGS.batch_size, tf.float32)
    encoder_grad_var = bvae_model.compute_grad_variance(
      encoder_grad_variable, encoder_grad_sq_variable, infnet_grads) / batch_size_sq

    if grad_variable_dict is not None:
      variance_dict = dict()
      for k in grad_variable_dict.keys():
        encoder_grads = estimate_gradients(
          input_batch, bvae_model, gradient_type=k)[2]
        variance_dict['var/' + k] = bvae_model.compute_grad_variance(
          grad_variable_dict[k], grad_sq_variable_dict[k],
          encoder_grads) / batch_size_sq
    else:
      variance_dict = None

    return (encoder_grad_var, eta_grad_var, variance_dict, genmo_loss, metrics)

  logdir = FLAGS.logdir

  genmo_lr = tf.constant(FLAGS.genmo_lr)
  infnet_lr = tf.constant(FLAGS.infnet_lr)
  prior_lr = tf.constant(FLAGS.prior_lr)
  hyper_lr = tf.constant(FLAGS.hyper_lr)
  control_lr = tf.constant(FLAGS.control_lr)

  genmo_optimizer = tf.keras.optimizers.Adam(learning_rate=genmo_lr)
  infnet_optimizer = tf.keras.optimizers.Adam(learning_rate=infnet_lr)
  prior_optimizer = tf.keras.optimizers.SGD(learning_rate=prior_lr)
  theta_optimizer = tf.keras.optimizers.Adam(learning_rate=infnet_lr,
                                             beta_1=0.999)
  hyper_optimizer = tf.keras.optimizers.Adam(learning_rate=hyper_lr)
  control_optimizer = tf.keras.optimizers.Adam(learning_rate=control_lr)

  num_steps_per_epoch = int(train_size / batch_size)

  if FLAGS.encoder_type == 'linear':
    encoder_hidden_sizes = [FLAGS.D]
    encoder_activations = ['linear']
    decoder_hidden_sizes = [784]
    decoder_activations = ['linear']
  elif FLAGS.encoder_type == 'nonlinear':
    encoder_hidden_sizes = [200, 200, FLAGS.D]
    encoder_activations = [
      # tf.nn.relu,
      # tf.nn.relu,
      layers.LeakyReLU(alpha=0.3),
      layers.LeakyReLU(alpha=0.3),
      'linear']
    decoder_hidden_sizes = [200, 200, 784]
    decoder_activations = [
      # tf.nn.relu,
      # tf.nn.relu,
      layers.LeakyReLU(alpha=0.3),
      layers.LeakyReLU(alpha=0.3),
      'linear']
  else:
    raise NotImplementedError

  encoder = [networks.BinaryNetwork(
    encoder_hidden_sizes,
    encoder_activations,
    name='bvae_encoder')]
  decoder = [networks.BinaryNetwork(
    decoder_hidden_sizes,
    decoder_activations,
    name='bvae_decoder')]
  encoder[0].build(input_shape=(None, 784))
  decoder[0].build(input_shape=(None, None, FLAGS.D))

  prior_logit = tf.Variable(tf.zeros([FLAGS.D], tf.float32))

  if grad_type == 'relax':
    control_network = tf.keras.Sequential()
    control_network.add(
        layers.Dense(137, activation=layers.LeakyReLU(alpha=0.3)))
    control_network.add(
        layers.Dense(1))
  elif grad_type.startswith('discrete_stein'):
    control_network = networks.SteinControlNNSingleLayer()
  else:
    control_network = None

  bvae_model = networks.DiscreteVAE(
    encoder,
    decoder,
    prior_logit,
    likelihood=likelihood,
    grad_type=grad_type,
    control_nn=control_network)

  # In order to use `tf.train.ExponentialMovingAverage`, one has to
  # use `tf.Variable`.
  encoder_grad_variable = initialize_grad_variables(bvae_model.encoder_vars)
  encoder_grad_sq_variable = initialize_grad_variables(bvae_model.encoder_vars)

  eta_grad_variable = tf.Variable(tf.zeros([FLAGS.batch_size, FLAGS.D]))
  eta_grad_sq_variable = tf.Variable(tf.zeros([FLAGS.batch_size, FLAGS.D]))

  if FLAGS.estimate_grad_basket:
    if grad_type == 'reinforce_loo':
      grad_basket = ['disarm', 'reinforce_loo', 'relax']
    else:
      raise NotImplementedError

    grad_variable_dict = {
      k: initialize_grad_variables(bvae_model.encoder_vars)
      for k in grad_basket}
    grad_sq_variable_dict = {
      k: initialize_grad_variables(bvae_model.encoder_vars)
      for k in grad_basket}

  else:
    grad_variable_dict = None
    grad_sq_variable_dict = None

  start_step = infnet_optimizer.iterations.numpy()
  logging.info('Training start from step: %s', start_step)
  if test_run:
    step_freq = 200
    steps = tqdm(range(start_step, 20000))
  else:
    tensorboard_file_writer = tf.summary.create_file_writer(logdir)
    step_freq = 1000
    steps = range(start_step, FLAGS.num_steps)

  train_iter = iter(train_ds)
  variances = defaultdict(list)
  losses = []
  alphas = []
  df_rows = []
  timer = -time.time()


  for step_i in steps:
    x_batch = train_iter.get_next()
    # x_batch, y_batch = train_iter.get_next()
    # x_batch = 2 * tf.cast(x_batch, tf.float64) / 255. - 1.
    # x_batch = tf.reshape(x_batch, [-1, 784])
    (encoder_grad_var, eta_grad_var, variance_dict, genmo_loss, metrics) = train_one_step(
      x_batch,
      bvae_model,
      genmo_optimizer,
      infnet_optimizer,
      prior_optimizer,
      hyper_optimizer,
      control_optimizer,
      theta_optimizer,
      encoder_grad_variable,
      encoder_grad_sq_variable,
      eta_grad_variable,
      eta_grad_sq_variable,
      grad_variable_dict,
      grad_sq_variable_dict,
      grad_type)
    train_loss = tf.reduce_mean(genmo_loss)
    # print("loss:", train_loss)
    # if step_i > 2:
    #   exit(0)

    # Summarize
    if step_i % step_freq == 0:
      timer += time.time()
      metrics.update({
        'train_objective': train_loss,
        'var/grad': encoder_grad_var
      })
      if grad_type == 'relax':
        if FLAGS.temperature is None:
          metrics['relax/temperature'] = tf.math.exp(
            bvae_model.log_temperature_variable)
        if FLAGS.scaling_factor is None:
          metrics['relax/scaling'] = bvae_model.scaling_variable
      elif grad_type == "double_cv":
        metrics['double_cv/alpha'] = bvae_model.alpha
      elif grad_type.startswith("discrete_stein"):
        metrics['discrete_stein/alpha'] = bvae_model.alpha
        metrics['discrete_stein/beta'] = bvae_model.beta
        metrics['discrete_stein/gamma'] = bvae_model.gamma
      if test_run:
        variances[grad_type].append(encoder_grad_var.numpy())
        losses.append(train_loss.numpy())
        if grad_type == "double_cv":
          alphas.append(bvae_model.alpha.numpy())
        # variances[grad_type].append(eta_grad_var.numpy())
      else:
        tf.print("time:", timer / step_freq)
        metrics.update({
          'eval_metric/train': evaluate(
            bvae_model, train_ds,
            max_step=num_steps_per_epoch,
            num_eval_samples=FLAGS.num_train_samples),
          'eval_metric/valid': evaluate(
            bvae_model, valid_ds,
            num_eval_samples=FLAGS.num_eval_samples),
          'eval_metric/test': evaluate(
            bvae_model, test_ds,
            num_eval_samples=FLAGS.num_eval_samples)
        })
        tf.print(step_i, metrics)
        with tensorboard_file_writer.as_default():
          for k, v in metrics.items():
            tf.summary.scalar(k, v, step=step_i)
          if variance_dict is not None:
            tf.print(variance_dict)
            for k, v in variance_dict.items():
              tf.summary.scalar(k, v, step=step_i)

      row = {"step": step_i, "seed": seed}
      for k, v in metrics.items():
        row[k] = v.numpy()
      df_rows.append(row)

      timer = -time.time()


  # if FLAGS.bias_check:
  #   if grad_type == 'reinforce_loo':
  #     baseline_type = 'disarm'
  #   else:
  #     baseline_type = 'reinforce_loo'
  #   run_bias_check(bvae_model,
  #                  train_iter.next(),
  #                  grad_type,
  #                  baseline_type)

  if test_run:
    return variances[grad_type], losses, alphas
  else:
    df = pd.DataFrame(df_rows)
    return df


def test_run(_):
  os.makedirs(FLAGS.logdir, exist_ok=True)
  import matplotlib.pyplot as plt
  variances = {}
  losses = {}
  alphas = {}
  grad_types = ["discrete_stein_avg", "discrete_stein_mu", "double_cv", "reinforce_loo"]
  for grad_type in grad_types:
      print("\n{}...\n".format(grad_type))
      variances[grad_type], losses[grad_type], alphas[grad_type] = main(grad_type, test_run=True)

  plt.figure()
  for grad_type in grad_types:
      print(variances[grad_type][:10])
      if grad_type == "reinforce_loo": 
          plt.plot(variances[grad_type], '--', label=grad_type)
      else:
          plt.plot(variances[grad_type], label=grad_type)
  plt.legend()
  # plt.yscale("log")
  plt.title("Variance")
  plt.savefig("var.pdf", bbox_inches="tight")

  plt.figure()
  for grad_type in grad_types:
      print(losses[grad_type][:10])
      if grad_type == "reinforce_loo": 
          plt.plot(losses[grad_type], '--', label=grad_type)
      else:
          plt.plot(losses[grad_type], label=grad_type)
  plt.legend()
  plt.title("Loss")
  plt.savefig("loss.pdf", bbox_inches="tight")

  # plt.figure()
  # print(alphas["double_cv"][:10])
  # plt.plot(alphas["double_cv"], label="double_cv")
  # plt.legend()
  # plt.title("alpha")
  # plt.savefig("alpha.pdf", bbox_inches="tight")


def run(_):
  physical_devices = tf.config.list_physical_devices('GPU')
  for gpu in physical_devices:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
    except:
      # Invalid device or cannot modify virtual devices once initialized.
      pass
  if FLAGS.logdir == 'none':
    type_name = FLAGS.grad_type
    if FLAGS.type_remap != 'none':
      type_name = FLAGS.type_remap
    FLAGS.logdir = "./logs/%s/%s-%d/%s-%d/%d-%s" % (
        FLAGS.dataset,
        FLAGS.encoder_type,
        FLAGS.D,
        type_name,
        FLAGS.K,
        FLAGS.seed,
        datetime.now().strftime("%Y%m%d_%H%M%S")
    )
  os.makedirs(FLAGS.logdir, exist_ok=True)
  os.makedirs(FLAGS.logdir + '/code', exist_ok=True)
  os.system('cp -v src/*.py "%s/code"' % FLAGS.logdir)
  with open(os.path.join(FLAGS.logdir, 'flags.txt'), 'w') as f:
    f.write(FLAGS.flags_into_string())
  df = main(FLAGS.grad_type)
  df.to_csv(os.path.join(FLAGS.logdir, "results.csv"))


if __name__ == '__main__':
  app.run(run)
  # app.run(run)
