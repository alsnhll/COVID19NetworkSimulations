"""Tests for seir.py. Also gives examples of how to run simulations."""

from absl.testing import absltest
from jax import random
from jax import vmap
import jax.numpy as np
import jax.test_util as test_util
import seir


class SEIRTest(test_util.JaxTestCase):

  def test_interaction_sampler(self):
    """Checks sampling of intractions occurs as expected."""
    _, sample = seir.interaction_sampler(
        random.PRNGKey(0), 0.3 * np.ones([4, 4]))
    self.assertAllClose(
        sample,
        np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, 0, 0, 1]]),
        check_dtypes=True)

  def test_interaction_step(self):
    """Given seeding, checks state develops as expected in an interaction."""
    key = random.PRNGKey(0)
    n = 4
    state = np.array([2, 0, 1, 0])
    state_timer = np.array([2, 0, 2, 0])
    w = 0.6 * np.ones([n, n])
    infection_probabilities = np.array([0., 0., 0.5, 0.5, 0.5, 0., 0.])
    state_length_sampler = lambda k, s: (k, np.ones([n], dtype=np.int32))
    _, new_state, new_state_timer = seir.interaction_step(
        key, state, state_timer, w, infection_probabilities,
        state_length_sampler)
    self.assertAllClose(new_state, np.array([2, 1, 1, 0]), check_dtypes=True)
    self.assertAllClose(
        new_state_timer, np.array([2, 1, 2, 0]), check_dtypes=True)

    state = np.array([3, 5, 4, 6])
    state_timer = np.array([2, 0, 2, 0])
    _, new_state, new_state_timer = seir.interaction_step(
        key, state, state_timer, w, infection_probabilities,
        state_length_sampler)
    self.assertAllClose(new_state, state, check_dtypes=True)
    self.assertAllClose(new_state_timer, state_timer, check_dtypes=True)

  def test_sparse_interaction_step(self):
    """Given seeding, checks state develops as expected in an interaction."""
    key = random.PRNGKey(5)
    n = 4
    state = np.array([2, 0, 1, 0])
    state_timer = np.array([2, 0, 2, 0])
    w = [np.reshape(np.tile(np.arange(n), [n, 1]).T, (-1,)),
         np.tile(np.arange(n), [1, n]),
         0.6 * np.ones([n**2])]
    infection_probabilities = np.array([0., 0., 0.5, 0.5, 0.5, 0., 0.])
    state_length_sampler = lambda k, s: (k, np.ones([n], dtype=np.int32))
    _, new_state, new_state_timer = seir.sparse_interaction_step(
        key, state, state_timer, w, infection_probabilities,
        state_length_sampler)
    self.assertAllClose(new_state, np.array([2, 1, 1, 0]), check_dtypes=True)
    self.assertAllClose(
        new_state_timer, np.array([2, 1, 2, 0]), check_dtypes=True)

    state = np.array([3, 5, 4, 6])
    state_timer = np.array([2, 0, 2, 0])
    _, new_state, new_state_timer = seir.sparse_interaction_step(
        key, state, state_timer, w, infection_probabilities,
        state_length_sampler)
    self.assertAllClose(new_state, state, check_dtypes=True)
    self.assertAllClose(new_state_timer, state_timer, check_dtypes=True)

  def test_sample_development(self):
    """Checks sampling of developments occurs as expected."""
    key = random.PRNGKey(0)
    state = np.reshape(
        vmap(lambda x: x * np.ones(1000, dtype=np.int32))(np.arange(7)), (-1))
    recovery_probabilities = np.array([0., 0., 0.9, 0.5, 0.1, 0., 0.])
    _, new_state = seir.sample_development(key, state, recovery_probabilities)
    average_developments = np.mean(np.reshape(new_state, [7, -1]), axis=-1)
    self.assertAllClose(np.array([1., 2., 5.676, 5.016, 5.08, 6., 7.]),
                        average_developments, check_dtypes=True)

  def test_developing_step(self):
    """Given seeding, checks state develops as expected in an develop step."""
    key = random.PRNGKey(0)
    state = np.arange(7)
    state_timer = np.array([0, 2, 2, 2, 2, 0, 0])
    recovery_probabilities = np.array([0., 0., 1., 1., 0., 0., 0.])
    state_length_sampler = lambda k, s: (k, s * np.ones([7], dtype=np.int32))
    _, new_state, new_state_timer = seir.developing_step(
        key, state, state_timer, recovery_probabilities, state_length_sampler)
    self.assertAllClose(state, new_state, check_dtypes=True)
    self.assertAllClose(
        np.array([0, 1, 1, 1, 1, 0, 0]), new_state_timer, check_dtypes=True)

    state_timer = np.array([0, 1, 1, 1, 1, 0, 0])
    _, new_state, new_state_timer = seir.developing_step(
        key, state, state_timer, recovery_probabilities, state_length_sampler)
    self.assertAllClose(
        np.array([0, 2, 6, 6, 5, 5, 6]), new_state, check_dtypes=True)
    self.assertAllClose(
        np.array([0, 2, 6, 6, 5, 0, 0]), new_state_timer, check_dtypes=True)

  def test_simulate(self):
    """Ensures a simple simulation runs."""
    key = random.PRNGKey(0)
    n = 4
    init_state = np.array([2, 0, 1, 0])
    init_state_timer = np.array([2, 0, 2, 0])
    w = 0.3 * np.ones([n, n])
    infection_probabilities = np.array([0., 0., 0.9, 0.8, 0.7, 0., 0.])
    recovery_probabilities = np.array([0., 0., 1., 1., 0., 0., 0.])
    state_length_sampler = lambda k, s: (k, np.ones([n], dtype=np.int32))
    total_steps = int(10. / 0.5)
    _ = seir.simulate(
        w, total_steps, state_length_sampler, infection_probabilities,
        recovery_probabilities, init_state, init_state_timer, key, epoch_len=1)

  def test_simulate_intervention(self):
    """Ensures a simple intervention simulation runs."""
    key = random.PRNGKey(0)
    n = 4
    init_state = np.array([2, 0, 1, 0])
    init_state_timer = np.array([2, 0, 2, 0])
    w1 = [np.reshape(np.tile(np.arange(n), [n, 1]).T, (-1,)),
          np.tile(np.arange(n), [1, n]),
          0.3 * np.ones([n**2])]
    w2 = [np.reshape(np.tile(np.arange(n), [n, 1]).T, (-1,)),
          np.tile(np.arange(n), [1, n]),
          0.03 * np.ones([n**2])]
    infection_probabilities = np.array([0., 0., 0.9, 0.8, 0.7, 0., 0.])
    recovery_probabilities = np.array([0., 0., 1., 1., 0., 0., 0.])
    state_length_sampler = lambda k, s: (k, np.ones([n], dtype=np.int32))
    step_intervals = [int(10. / 0.5), int(10. / 0.5)]
    _ = seir.simulate_intervention(
        [w1, w2], step_intervals, state_length_sampler, infection_probabilities,
        recovery_probabilities, init_state, init_state_timer, key, epoch_len=1)


if __name__ == '__main__':
  absltest.main()
