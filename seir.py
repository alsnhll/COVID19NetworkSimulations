"""Code to simulate microscopic SEIR dynamics on a weighted, directed graph.

See https://alhill.shinyapps.io/COVID19seir/ for more details on the ODE version
of the model.

The states of the individuals in the population are stored as ints:
S, E, I1, I2, I3, D, R
0, 1,  2,  3,  4, 5, 6
"""
import functools
from jax import jit
from jax import random
from jax.lax import fori_loop
from jax.nn import relu
import jax.numpy as np
from jax.ops import index_add
import tqdm

SUSCEPTIBLE = 0
EXPOSED = 1
INFECTED_1 = 2
INFECTED_2 = 3
INFECTED_3 = 4
DEAD = 5
RECOVERED = 6
NUM_STATES = 7
INFECTIOUS_STATES = (INFECTED_1, INFECTED_2, INFECTED_3)
NON_INFECTIOUS_STATES = (SUSCEPTIBLE, EXPOSED, DEAD, RECOVERED)
TRANSITIONAL_STATES = (EXPOSED, INFECTED_1, INFECTED_2, INFECTED_3)


@jit
def to_one_hot(state):
  return state[:, np.newaxis] == np.arange(NUM_STATES)[np.newaxis]


@jit
def is_susceptible(state):
  """Checks whether individuals are susceptible based on state."""
  return state == SUSCEPTIBLE


@jit
def is_transitional(state):
  """Checks whether individuals are in a state that can develop."""
  return np.logical_and(EXPOSED <= state, state <= INFECTED_3)


@jit
def interaction_sampler(key, w):
  key, subkey = random.split(key)
  return key, random.bernoulli(subkey, w).astype(np.int32)


@functools.partial(jit, static_argnums=(3, 4, 5))
def interaction_step(key, state, state_timer, w, infection_probabilities,
                     state_length_sampler):
  """Determines new infections from the state and population structure."""
  key, interaction_sample = interaction_sampler(
      key, infection_probabilities[state][:, np.newaxis] * w)
  new_infections = is_susceptible(state) * np.max(interaction_sample, axis=0)
  key, infection_lengths = state_length_sampler(key, 1)
  return (key,
          state + new_infections,
          state_timer + new_infections * infection_lengths)


@functools.partial(jit, static_argnums=(3, 4, 5))
def sparse_interaction_step(key, state, state_timer, w, infection_probabilities,
                            state_length_sampler):
  """Determines new infections from the state and population structure."""
  rows, cols, ps = w
  key, interaction_sample = interaction_sampler(
      key, infection_probabilities[state[rows]] * ps)

  new_infections = is_susceptible(state) * np.sign(
      index_add(np.zeros_like(state), cols, interaction_sample))

  key, infection_lengths = state_length_sampler(key, 1)
  return (key,
          state + new_infections,
          state_timer + new_infections * infection_lengths)


@functools.partial(jit, static_argnums=(2,))
def sample_development(key, state, recovery_probabilities):
  """Individuals who are in a transitional state either progress or recover."""
  key, subkey = random.split(key)
  is_recovered = random.bernoulli(subkey, recovery_probabilities[state])
  return key, (state + 1) * (1 - is_recovered) + RECOVERED * is_recovered


@functools.partial(jit, static_argnums=(3, 4))
def developing_step(key, state, state_timer, recovery_probabilities,
                    state_length_sampler):
  to_develop = np.logical_and(state_timer == 1, is_transitional(state))
  state_timer = relu(state_timer - 1)
  key, new_state = sample_development(key, state, recovery_probabilities)
  key, new_state_timer = state_length_sampler(key, new_state)
  return (key,
          state * (1 - to_develop) + new_state * to_develop,
          state_timer * (1 - to_develop) + new_state_timer * to_develop)


def simulate(w, total_steps, state_length_sampler, infection_probabilities,
             recovery_probabilities, init_state, init_state_timer, key=0,
             epoch_len=1, states_cumulative=None):
  """Simulates microscopic SEI^3R dynamics on a weighted, directed graph.

  The simulation is Markov chain, whose state is recorded by three device
  arrays, state, state_timer, and states_cumulative. The ith entry of state
  indicates the state of individual i. The ith entry of state_timer indicates
  the time number of timesteps that individual i will remain in its current
  state, with 0 indicating that it will remain in the current state
  indefinietely. The (i,j)th entry of states_cumulative is an indicator for
  whether individual i has ever been in state j.

  Args:
    w: There are two otpions for w. 1) A DeviceArray of shape [n, n], where n
      is the population size. The entry ij represents the probability that
      individual i infects j. 2) A list of DeviceArrays [rows, cols, ps], where
      the ith entries are the probability ps[i] that individual rows[i] infects
      individual cols[i].
    total_steps: The total number of updates to the Markov chain.
    state_length_sampler: A function taking a PRNGKey that returns a
      DeviceArray of shape [n]. Each entry is an iid sample from the distibution
      specifying the amount of time that the individual remains infected.
    infection_probabilities: A DeviceArray of shape [7], where each entry is
      the probability of an infection given that an interaction occurs. Note
      that the 0, 1, 5, and 6 entries must be 0.
    recovery_probabilities: A DeviceArray of shape [7], where each entry is
      the probability of recovering from that state. Note that the 0, 1, 5, and
      6 entries must be 0.
    init_state: A DeviceArray of shape [n] containing ints for the initial state
      of the simulation.
    init_state_timer: A DeviceArray of shape [n] containing ints for the number
      of time steps an individual will remain in the current state. When the int
      is 0, the state persists indefinitely.
    key: An int to use as the PRNGKey.
    epoch_len: The number of steps that are JIT'ed in the computation. After
      each epoch the current state of the Markov chain is logged.
    states_cumulative: A DeviceArray of Bools of shape [n, 7] indicating whether
      an individual has ever been in a state.

  Returns:
    A tuple (key, state, state_timer, states_cumulative, history), where state,
    state_timer, and states_cumulative are the final state of the simulation and
    history is the number of each type over the course of the simulation.
  """
  if any(infection_probabilities[state] > 0 for state in NON_INFECTIOUS_STATES):
    raise ValueError('Only states i1, i2, and i3 are infectious! Other entries'
                     ' of infection_probabilities must be 0.')
  if any(recovery_probabilities[state] > 0 for state in NON_INFECTIOUS_STATES):
    raise ValueError('Recovery can only occur from states i1, i2, and i3! Other'
                     ' entries of recovery_probabilities must be 0.')

  if isinstance(key, int):
    key = random.PRNGKey(key)

  interaction_step_ = interaction_step
  if isinstance(w, list):
    interaction_step_ = sparse_interaction_step

  def eval_fn(t, state, state_timer, states_cumulative, history):
    del t, state_timer
    history.append([np.mean(to_one_hot(state), axis=0),
                    np.mean(states_cumulative, axis=0)])
    return history

  @jit
  def step(t, args):
    del t
    key, state, state_timer, states_cumulative = args
    key, state, state_timer = interaction_step_(
        key, state, state_timer, w, infection_probabilities,
        state_length_sampler)
    key, state, state_timer = developing_step(
        key, state, state_timer, recovery_probabilities, state_length_sampler)
    states_cumulative = np.logical_or(to_one_hot(state), states_cumulative)
    return key, state, state_timer, states_cumulative

  state, state_timer = init_state, init_state_timer
  if states_cumulative is None:
    states_cumulative = np.logical_or(
        to_one_hot(state), np.zeros_like(to_one_hot(state), dtype=np.bool_))

  epochs = int(total_steps // epoch_len)
  history = []
  for epoch in tqdm.tqdm(range(epochs), total=epochs, position=0):
    key, state, state_timer, states_cumulative = fori_loop(
        0, epoch_len, step, (key, state, state_timer, states_cumulative))
    history = eval_fn(
        epoch*epoch_len, state, state_timer, states_cumulative, history)

  return key, state, state_timer, states_cumulative, history


def simulate_intervention(
    ws, step_intervals, state_length_sampler, infection_probabilities,
    recovery_probabilities, init_state, init_state_timer, key=0, epoch_len=1):
  """Simulates an intervention with the SEI^3R model above.

  By passing a list of population strucutres and time intervals. Several runs
  of simulate() are called sequentially with different w for fixed time lengths.
  This models the effect of interventions that affect the population strucure
  to mitigate virus spread, such as social distancing.

  Args:
    ws: A list of DeviceArrays of shape [n, n], where n is the population size.
      The dynamics will be simulated on each strucutre sequentially.
    step_intervals: A list of ints indicating the number fo simulation steps
      performed on each population strucutre.
    state_length_sampler: See simulate function above.
    infection_probabilities: See simulate function above.
    recovery_probabilities: See simulate function above.
    init_state: See simulate function above.
    init_state_timer: See simulate function above.
    key: See simulate function above.
    epoch_len: See simulate function above.

  Returns:
    A tuple (key, state, state_timer, states_cumulative, history), where state,
    state_timer, and states_cumulative are the final state of the simulation and
    history is the number of each type over the course of the simulation.
  """
  history = []
  state, state_timer = init_state, init_state_timer
  states_cumulative = np.logical_or(
      to_one_hot(state), np.zeros_like(to_one_hot(state), dtype=np.bool_))
  for t, (w, total_steps) in enumerate(zip(ws, step_intervals)):
    key, state, state_timer, states_cumulative, history_ = simulate(
        w, total_steps, state_length_sampler, infection_probabilities,
        recovery_probabilities, state, state_timer, key, epoch_len)
    history.extend(history_)
    print('Completed interval {} of {}'.format(t+1, len(ws)))

  return key, state, state_timer, states_cumulative, history
