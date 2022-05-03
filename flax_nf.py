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

from data import * 


FLAGS = flags.FLAGS

flags.DEFINE_integer(
		"n_epochs", default=int(5e4),
		help=("total training epochs")
)
flags.DEFINE_integer(
		"n_batch", default=512,
		help=("batch size")
)
flags.DEFINE_integer(
		"n_hidden", default=256,
		help=("number of hidden neurons in layers")
)
flags.DEFINE_float(
		"lr", default=0.0001,
		help=("learning rate")
)
flags.DEFINE_integer(
		"n_data", default=4000,
		help=("number of data points to sample from data")
)
flags.DEFINE_integer(
		"n_flows", default=8,
		help=("number of flows in normalizing flow model")
)
flags.DEFINE_integer(
		"n_sample", default=2000,
		help=("number of points to sample when plotting")
)


class NormalizingFlow(nn.Module):
    n_flows: int 
    n_hidden: int
    forward: bool

    def setup(self):
        self.flows = [
            FlowUnit(forward=True, n_hidden=FLAGS.n_hidden, flip=True) 
			for _ in range(self.n_flows)
        ]

    def prior_log_prob(self, x):
        # return (-jnp.log(jnp.sqrt(2 * jnp.pi)) - (x ** 2) / (2.0 ** 2)).sum(1)
        return jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=jnp.zeros((2,)), cov=jnp.eye(2)
        )

    def __call__(self, x):
        logdet = 0.0
        xs = []  # latents
        if self.forward:
            for flow in self.flows:
                x, logdet_i = flow(x, sample=False)
                xs.append(x)
                logdet += logdet_i
        else:
            for flow in self.flows[::-1]:
                x, logdet_i = flow(x, sample=True)
                xs.append(x)
                logdet += logdet_i

        return x, self.prior_log_prob(x), logdet, xs


class FlowUnit(nn.Module):
	forward: bool
	n_hidden: int
	flip: bool

	def setup(self):
		self.scale_shift = nn.Sequential(
			[
				nn.Dense(self.n_hidden), nn.relu,
				nn.Dense(self.n_hidden // 2), nn.relu,
				nn.Dense(2),
			]
		)

	def _forward(self, x):
		d = x.shape[-1] // 2
		x1, x2 = x[:, :d], x[:, d:]
		if self.flip:
			x2, x1 = x1, x2
		
		shift, log_scale = jnp.split(self.scale_shift(x1), 2, axis=1)

		y2 = x2 * jnp.exp(log_scale) + shift

		if self.flip:
			x1, y2 = y2, x1

		x = jnp.concatenate([x1, y2], axis=-1)
		return x, jnp.sum(log_scale, axis=1)

	def _backward(self, y):
		d = y.shape[-1] // 2
		y1, y2 = y[:, :d], y[:, d:]
		if self.flip:
			y1, y2 = y2, y1

		shift, log_scale = jnp.split(self.scale_shift(y1), 2, axis=1)

		x2 = (y2 - shift) * jnp.exp(-log_scale)

		if self.flip:
			y1, x2 = x2, y1

		x = jnp.concatenate([y1, x2], axis=-1)
		return x, jnp.sum(-log_scale, axis=1)
	
	def __call__(self, x, sample=False):
		if not sample:
			x, logdet = self._forward(x)
		else:
			x, logdet = self._backward(x)
		return x, logdet


@jax.jit
def train_step(state, x):
    def loss_fn(params):
        """
        Get logL-loss using prior log prob and flow determinants
        """
        z, prior_logprob, log_det, _ = NormalizingFlow(
            n_flows=FLAGS.n_flows, 
			n_hidden=FLAGS.n_hidden, 
			forward=True
        ).apply({"params": params}, x)
        loss = -jnp.mean(prior_logprob + log_det)
        # loss = jnp.nan_to_num(loss)
        return loss

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


@jax.jit
def eval_step(params, x):
    z, _, _, _ = NormalizingFlow(
        n_flows=FLAGS.n_flows, 
		n_hidden=FLAGS.n_hidden, 
		forward=True
    ).apply({"params": params}, x)
    x, _, _, _ = NormalizingFlow(
        n_flows=FLAGS.n_flows, 
		n_hidden=FLAGS.n_hidden, 
		forward=False
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

	rng, init_rng, init_x_rng = random.split(rng, 3)

	# instantiate model and parameters
	model = NormalizingFlow(n_flows=FLAGS.n_flows, n_hidden=FLAGS.n_hidden, forward=True)
	params = model.init(init_rng, jnp.ones([3, 2]))["params"]

	# get optimizer and train state holder
	state = train_state.TrainState.create(
		apply_fn=model.apply, 
		params=params, 
		tx=optax.adam(FLAGS.lr)
	)

	# sample some initial points to generate with before/after training
	z = random.normal(init_x_rng, (FLAGS.n_sample, 2))
	init_x, _, _, _ = NormalizingFlow(
		n_flows=FLAGS.n_flows, 
		n_hidden=FLAGS.n_hidden, 
		forward=False
	).apply({"params": params}, z)

	losses = []

	for e in range(FLAGS.n_epochs):
		rng, data_rng = random.split(rng)

		x = select_data(moons, data_rng)

		loss, state = train_step(state, x)

		losses.append(loss)
		print("\r %08d %.8f" % (e, loss), end="")

	# sample some points after training
	x, _, _, xs = NormalizingFlow(
		n_flows=FLAGS.n_flows, 
		n_hidden=FLAGS.n_hidden, 
		forward=False
	).apply({"params": state.params}, z)

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
	ax[1].plot(losses[::200])
	plt.savefig("test.png")

	"""
		Grid plot of all flows	
	"""

	s = int(FLAGS.n_flows ** 0.5)
	fig, ax = plt.subplots(s, s, dpi=200, figsize=(6.0, 6.0))
	# moons dataset
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
