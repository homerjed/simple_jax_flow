from absl import app
from absl import flags

import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
from flax.training import train_state
import optax

import matplotlib.pyplot as plt
from matplotlib import animation, rc
from sklearn import datasets

#from data import * 
from model import NormalizingFlow


FLAGS = flags.FLAGS

flags.DEFINE_integer(
		"n_epochs", default=2 * int(5e4),
		help=("total training epochs")
)
flags.DEFINE_integer(
		"n_batch", default=256,
		help=("batch size")
)
flags.DEFINE_integer(
		"n_hidden", default=64,
		help=("number of hidden neurons in layers")
)
flags.DEFINE_float(
		"lr", default=0.01,
		help=("learning rate")
)
flags.DEFINE_integer(
		"n_data", default=4000,
		help=("number of data points to sample from data dist")
)
flags.DEFINE_integer(
		"n_flows", default=6,
		help=("number of flows in normalizing flow model")
)
flags.DEFINE_integer(
		"n_sample", default=2000,
		help=("number of points to sample when plotting")
)
flags.DEFINE_integer(
		"n_dims", default=2,
		help=("number of dimensions data / flow have")
)


@jax.jit
def train_step(state, x):
	def loss_fn(params):
		"""
		Get logL-loss using prior log prob and flow determinants
		"""
		z, prior_logprob, log_det, _ = NormalizingFlow(
			n_flows=FLAGS.n_flows, 
			n_hidden=FLAGS.n_hidden, 
		).apply({"params": params}, x, forward=True)
		loss = -jnp.sum(prior_logprob + log_det)
		return loss

	grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
	loss, grads = grad_fn(state.params)
	state = state.apply_gradients(grads=grads)
	return loss, state


@jax.jit
def eval_step(params, x):
	# return reconstructed img / latent
	z, _, _, _ = NormalizingFlow(
		n_flows=FLAGS.n_flows, 
		n_hidden=FLAGS.n_hidden, 
	).apply({"params": params}, x)
	x, _, _, _ = NormalizingFlow(
		n_flows=FLAGS.n_flows, 
		n_hidden=FLAGS.n_hidden, 
	).apply({"params": params}, z)
	return z, x


def select_data(data, key):
	ix = random.randint(key, (FLAGS.n_batch,), 0, data.shape[0])
	return data[ix]


def main(argv):
	del argv 

	rng = random.PRNGKey(0)

	# get dataset
	moons = datasets.make_moons(FLAGS.n_data, noise=0.05)[0].astype(jnp.float32)
	S = random.uniform(rng, (2,2))
	cov = S.T @ S
	moons = random.multivariate_normal(
			rng, 
			shape=(FLAGS.n_data,), 
			mean=jnp.zeros((2,)), 
			cov=cov
	)

	def oliver_dist(data):
		x, y = data.T
		z_1 = 0.3 * x
		z_2 = y
		z_1 = z_1 + 0.3*z_2**2
		data = jnp.stack([z_1, z_2])
		print(data.shape)
		return data.T

	moons = oliver_dist(moons)

	key1, key2 = random.split(rng)
	"""
	x2_samples = random.normal(key1, (FLAGS.n_data,)) * 4.0
	x1_samples = random.normal(key2, (FLAGS.n_data,)) * 1.0 + 0.25 * jnp.multiply(x2_samples,x2_samples)
	moons = jnp.stack([x1_samples, x2_samples], 1)
	"""

	rng, init_rng, init_x_rng = random.split(rng, 3)

	# instantiate model and parameters
	model = NormalizingFlow(
			n_flows=FLAGS.n_flows, 
			n_hidden=FLAGS.n_hidden, 
	)
	params = model.init(init_rng, jnp.ones([3, 2]))["params"]

	# get optimizer and train state holder
	state = train_state.TrainState.create(
		apply_fn=model.apply, 
		params=params, 
		tx=optax.adam(FLAGS.lr)
	)

	# sample some initial points to generate with before/after training
	z = random.normal(init_x_rng, (FLAGS.n_sample, 2))

	# initialise flow
	init_x, _, _, _ = NormalizingFlow(
		n_flows=FLAGS.n_flows, 
		n_hidden=FLAGS.n_hidden, 
	).apply({"params": params}, z)

	losses = []
	for e in range(FLAGS.n_epochs):
		rng, data_rng = random.split(rng)

		x = select_data(moons, data_rng)

		loss, state = train_step(state, x)

		losses.append(loss)
		print("\r %08d %.3E" % (e, loss), end="")

	# sample some points after training
	x, _, _, xs = NormalizingFlow(
		n_flows=FLAGS.n_flows, 
		n_hidden=FLAGS.n_hidden, 
	).apply({"params": state.params}, z, forward=False)

	# z, x = eval_step(state.params, x)


	"""
		plot initial / final flow outputs, data distribution
	"""

	# check the data dist vs the flow dist
	fig, ax = plt.subplots(1, 2, dpi=200, figsize=(6.0, 3.0))
	# moons dataset
	ax[0].scatter(*moons.T, alpha=0.6, s=0.5, label="d")
	# before training (this isn't the prior)
	ax[0].scatter(*init_x.T, alpha=0.6, s=0.5, label="i")
	# after training
	ax[0].scatter(*x.T, alpha=0.6, s=0.5, label="f")
	ax[0].set_xlim(-5.0, 5.0)
	ax[0].set_ylim(-5.0, 5.0)
	ax[0].legend()
	# losses
	ax[0].axis("off")
	#ax[1].plot(losses[::200])
	ax[1].semilogy(losses[::200])
	plt.savefig("test.png")
	plt.close()

	"""
		Grid plot of all flows	
	"""

	s = int(FLAGS.n_flows ** 0.5)
	fig, ax = plt.subplots(s, s, dpi=200, figsize=(6.0, 6.0))
	c = 0
	for i in range(s):
		for j in range(s):
			ax[i, j].scatter(*xs[c].T, alpha=0.5, s=0.5, label="%d" % c)
			ax[i, j].set_xlim(-8.0, 8.0)
			ax[i, j].set_ylim(-8.0, 8.0)
			ax[i, j].axis("off")
			# ax[i,j].legend()
			c += 1
	ax[i, j].scatter(*moons.T, alpha=0.5, s=0.5, label="d")
	fig.tight_layout()
	fig.subplots_adjust(wspace=0.0, hspace=0.0)
	plt.savefig("steps.png")

	"""
		Animation of flows
	"""

	fig, ax = plt.subplots(figsize=(3.0, 3.0))
	paths = ax.scatter(*xs[0].T, s=0.5, color="rebeccapurple")
	w = 8.0
	ax.set_xlim([-w, w])
	ax.set_ylim([-w, w])
	ax.axis("off")
	fig.tight_layout()
	fig.subplots_adjust(wspace=0.0, hspace=0.0)

	# add initial and final outputs again...
	x_init = random.normal(rng, (FLAGS.n_sample, 2))
	xs = [x_init] + xs
	x_final = z
	xs = xs + [z]

	def animate(i):
		l = i // 48
		t = (float(i % 48)) / 48
		print("\r %d" % l, end="")
		y = (1 - t) * xs[l] + t * xs[l + 1]
		paths.set_offsets(y)
		return (paths,)

	anim = animation.FuncAnimation(
		fig, animate, frames=48 * FLAGS.n_flows, interval=1, blit=False
	)
	anim.save("anim.gif", writer="imagemagick", fps=60)


if __name__ == "__main__":
	app.run(main)
