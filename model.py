import jax 
import jax.numpy as jnp
import flax.linen as nn

class NormalizingFlow(nn.Module):
	n_flows: int 
	n_hidden: int

	def setup(self):
		self.flows = [
				FlowUnit(
					n_hidden=self.n_hidden, 
					flip=True
				) 
				for _ in range(self.n_flows)
		]

	def prior_log_prob(self, x):
		return jax.scipy.stats.multivariate_normal.logpdf(
				x, 
				mean=jnp.zeros((2,)), 
				cov=jnp.eye(2)
		)

	def __call__(self, x, forward=True):
		logdet = 0.0
		xs = []  # latents
		if forward: # forward: x-->z => if forward, not sampling.
			for flow in self.flows:
				x, logdet_i = flow(x, forward=forward)
				xs.append(x)
				logdet += logdet_i
		else:
			for flow in self.flows[::-1]:
				x, logdet_i = flow(x, forward=forward)
				xs.append(x)
				logdet += logdet_i

		return x, self.prior_log_prob(x), logdet, xs


class FlowUnit(nn.Module):
	n_hidden: int
	flip: bool

	def setup(self):
		"""
		self.scale_shift = nn.Sequential(
			[
				nn.Dense(self.n_hidden), nn.relu,
				nn.Dense(self.n_hidden // 2), nn.relu,
				nn.Dense(2),
			]
		)
		"""
		self.scale = nn.Sequential(
				[
					nn.Dense(self.n_hidden), nn.relu,
					nn.Dense(self.n_hidden // 2), nn.relu,
					nn.Dense(1),
					]
				)
		self.shift = nn.Sequential(
				[
					nn.Dense(self.n_hidden), nn.relu,
					nn.Dense(self.n_hidden // 2), nn.relu,
					nn.Dense(1),
					]
				)

	def _forward(self, x):
		x1, x2 = x[:, ::2], x[:, 1::2]
		if self.flip:
			x2, x1 = x1, x2
			
		"""
		shift, log_scale = self.scale_shift(x1).split(2, axis=1)
		"""
		shift = self.shift(x1)
		log_scale = self.scale(x1)

		y1 = x1
		y2 = x2 * jnp.exp(log_scale) + shift

		if self.flip:
			y1, y2 = y2, y1

		x = jnp.concatenate([y1, y2], axis=-1)
		return x, jnp.sum(log_scale, axis=1)

	def _backward(self, y):
		y1, y2 = y[:, ::2], y[:, 1::2]
		if self.flip:
			y1, y2 = y2, y1

		"""
		shift, log_scale = self.scale_shift(y1).split(2, axis=1)
		"""
		shift = self.shift(y1)
		log_scale = self.scale(y1)

		x1 = y1
		x2 = (y2 - shift) * jnp.exp(-log_scale)

		if self.flip:
			x1, x2 = x2, x1

		x = jnp.concatenate([x1, x2], axis=-1)
		return x, jnp.sum(-log_scale, axis=1)

	def __call__(self, x, forward=True):
		if forward:
			x, logdet = self._forward(x)
		else:
			x, logdet = self._backward(x)
		return x, logdet
