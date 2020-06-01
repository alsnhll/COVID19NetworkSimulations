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
import numpy as np2
from jax.ops import index_add, index_update, index
import tqdm
import matplotlib.pyplot as plt

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
        total_steps: The total number of updates to the Markov chain. Else can be
      a tuple (max_steps, break_fn), where break_fn is a function returning
      a bool indicating whether the simulation should terminate.
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
                     ' of infection_probabilities must be 0. Got {}.'.format(
                         infection_probabilities))
  if any(recovery_probabilities[state] > 0 for state in NON_INFECTIOUS_STATES):
    raise ValueError('Recovery can only occur from states i1, i2, and i3! Other'
                     ' entries of recovery_probabilities must be 0. Got '
                     '{}.'.format(recovery_probabilities))

  if isinstance(key, int):
    key = random.PRNGKey(key)

  interaction_step_ = interaction_step
  if isinstance(w, list):
    interaction_step_ = sparse_interaction_step
    
  if isinstance(total_steps, tuple):
    total_steps, break_fn = total_steps
  else:
    break_fn = lambda *args, **kwargs: False

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
    if break_fn(
        epoch*epoch_len, state, state_timer, states_cumulative, history):
      break

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
      performed on each population strucutre. Else a list of tuples of the form
      (max_steps, break_fn) see simulate function above.
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
        recovery_probabilities, state, state_timer, key, epoch_len,
        states_cumulative)
    history.extend(history_)
    print('Completed interval {} of {}'.format(t+1, len(ws)))

  return key, state, state_timer, states_cumulative, history

def plot_single(history,tvec,n,ymax=1,scale=1,int=0,Tint=0,plotThis=False,plotName="test"):
  """
  plots the output (prevalence) from a single simulation, with or without an intervention
  history: 2D array of values for each variable at each timepoint
  tvec: 1D vector of timepoints
  ymax : Optional, highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: Optional, amount to multiple all frequency values by (e.g. "1" keeps as frequency, "n" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  plotThis: True or False, whether a plot will be saved as pdf 
  plotName: string, name of the plot to be saved
  """
 
  plt.figure(figsize=(2*6.4, 4.0))
  plt.subplot(121)
  plt.plot(tvec,history*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
      plt.plot([Tint,Tint],[0,ymax*scale],'k--')
  plt.ylim([0,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Number")

  plt.subplot(122)
  plt.plot(tvec,history*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
    plt.plot([Tint,Tint],[scale/n,ymax*scale],'k--')
  plt.semilogy()
  plt.ylim([scale/n,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Number")
  plt.tight_layout()
  if plotThis==True:
  	plt.savefig('/content/gdrive/My Drive/COVID19 Network Simulations/figures/'+plotName+'.pdf',bbox_inches='tight')
  plt.show()

def plot_single_cumulative(cumulative_history,tvec,n,ymax=1,scale=1,int=0,Tint=0,plotThis=False,plotName="test"):
  """
  plots the output (cumulative prevalence) from a single simulation, with or without an intervention
  cumulative_history: 2D array of values for each variable at each timepoint
  tvec: 1D vector of timepoints
  ymax : Optional, highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: Optional, amount to multiple all frequency values by (e.g. "1" keeps as frequency, "n" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  plotThis: True or False, whether a plot will be saved as pdf 
  plotName: string, name of the plot to be saved
  """

  plt.figure(figsize=(2*6.4, 4.0))
  plt.subplot(121)
  plt.plot(tvec,cumulative_history*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
      plt.plot([Tint,Tint],[0,ymax*scale],'k--')
  plt.ylim([0,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Cumulative umber")

  plt.subplot(122)
  plt.plot(tvec,cumulative_history*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
    plt.plot([time_int,time_int],[scale/n,ymax*scale],'k--')
  plt.semilogy()
  plt.ylim([scale/n,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Cumulative number")
  plt.tight_layout()
  if plotThis==True:
  	plt.savefig('/content/gdrive/My Drive/COVID19 Network Simulations/figures/'+plotName+'.pdf',bbox_inches='tight')
  plt.show()

def get_daily(cumulative_history,tvec):
  """ 
  Gets the daily incidence for a single run
  cumulative_history: 2D array of cumulative values for each variable at each timepoint
  tvec: 1D vector of timepoints
  """
  
  Tmax=int(tvec[-1])
  delta_t=tvec[1]-tvec[0]
  total_steps=int(Tmax/delta_t)

  # first pick out entries corresponding to each day
  per_day=int(1/delta_t) # number of entries per day
  days_ind=np.arange(start=0,stop=total_steps,step=per_day)
  daily_cumulative_history=cumulative_history[days_ind,:]
  
  # then get differences between each day
  daily_incidence=daily_cumulative_history[1:Tmax,:]-daily_cumulative_history[0:(Tmax-1),:]

  return daily_incidence


def plot_single_daily(daily_incidence,n,ymax=1,scale=1,int=0,Tint=0,plotThis=False,plotName="test"):
  """
  plots the output (daily incidence) from a single simulation, with or without an intervention
  daily_incidence: 2D array of values for each variable at each timepoint
  ymax : Optional, highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: Optional, amount to multiple all frequency values by (e.g. "1" keeps as frequency, "n" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  plotThis: True or False, whether a plot will be saved as pdf 
  plotName: string, name of the plot to be saved
  """

  tvec=np.arange(1,len(daily_incidence)+1)

  plt.figure(figsize=(2*6.4, 4.0))
  plt.subplot(121)
  plt.plot(tvec,daily_incidence*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
      plt.plot([Tint,Tint],[0,ymax*scale],'k--')
  plt.ylim([0,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Daily incidence")

  plt.subplot(122)
  plt.plot(tvec,daily_incidence*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
    plt.plot([Tint,Tint],[scale/n,ymax*scale],'k--')
  plt.semilogy()
  plt.ylim([scale/n,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Daily incidence")
  plt.tight_layout()
  if plotThis==True:
  	plt.savefig('/content/gdrive/My Drive/COVID19 Network Simulations/figures/'+plotName+'.pdf',bbox_inches='tight')
  plt.show()

def get_peaks_single(history,tvec,int=0,Tint=0):

  """
  calculates the peak prevalence for a single run, with or without an intervention
  history: 2D array of values for each variable at each timepoint
  tvec: 1D vector of timepoints
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  """

  delta_t=tvec[1]-tvec[0]

  if int==0:
    time_int=0
  else:
    time_int=Tint
      
  # Final values
  print('Final recovered: {:3.1f}%'.format(100 * history[-1][6]))
  print('Final deaths: {:3.1f}%'.format(100 * history[-1][5]))
  print('Remaining infections: {:3.1f}%'.format(
      100 * np.sum(history[-1][1:5], axis=-1)))

  # Peak prevalence
  print('Peak I1: {:3.1f}%'.format(
      100 * np.max(history[:, 2])))
  print('Peak I2: {:3.1f}%'.format(
      100 * np.max(history[:, 3])))
  print('Peak I3: {:3.1f}%'.format(
      100 * np.max(history[:, 4])))

  # Time of peaks
  print('Time of peak I1: {:3.1f} days'.format(
      np.argmax(history[:, 2])*delta_t - time_int))
  print('Time of peak I2: {:3.1f} days'.format(
      np.argmax(history[:, 3])*delta_t - time_int))
  print('Time of peak I3: {:3.1f} days'.format(
      np.argmax(history[:, 4])*delta_t - time_int))
  
  # First time when all infections go extinct
  all_cases=history[:, 1]+history[:, 2]+history[:, 3]+history[:, 4]
  extinct=np.where(all_cases == 0)[0]
  if len(extinct) != 0:
    extinction_time=np.min(extinct)*delta_t - time_int
    print('Time of extinction of all infections: {:3.1f} days'.format(extinction_time))
  else:
     print('Infections did not go extinct by end of simulation')
 
  return

def get_peaks_single_daily(daily_incidence,int=0,Tint=0):

  """
  calculates the peak daily incidence for a single run, with or without an intervention
  daily_incidence: 2D array of values for each variable at each timepoint
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  """

  if int==0:
    time_int=0
  else:
    time_int=Tint

  # Peak incidence
  print('Peak daily I1: {:3.1f}%'.format(
      100 * np.max(daily_incidence[:, 2])))
  print('Peak daily I2: {:3.1f}%'.format(
      100 * np.max(daily_incidence[:, 3])))
  print('Peak daily I3: {:3.1f}%'.format(
      100 * np.max(daily_incidence[:, 4])))
  print('Peak daily D: {:3.1f}%'.format(
      100 * np.max(daily_incidence[:, 5]))) 

  # Time of peak incidence
  print('Time of peak daily I1: {:3.1f} days'.format(
      np.argmax(daily_incidence[:, 2])+1-time_int))
  print('Time of peak daily I2: {:3.1f} days'.format(
      np.argmax(daily_incidence[:, 3])+1-time_int))
  print('Time of peak daily I3: {:3.1f} days'.format(
      np.argmax(daily_incidence[:, 4])+1-time_int))
  print('Time of peak daily D: {:3.1f} days'.format(
      np.argmax(daily_incidence[:, 5])+1-time_int))  

  return

def plot_iter(soln,tvec,n,ymax=1,scale=1,int=0,Tint=0,plotThis=False,plotName="test"):

  """
  plots the output (prevalence) from a multiple simulation, with or without an intervention. Shows all trajectories
  soln: 3D array of values for each iteration for each variable at each timepoint
  tvec: 1D vector of timepoints
  n: total population size
  ymax : highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: amount to multiple all frequency values by (e.g. "1" keeps as frequency, "N" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  plotThis: True or False, whether a plot will be saved as pdf 
  plotName: string, name of the plot to be saved
  """

  number_trials=np.shape(soln)[0]

  plt.figure(figsize=(2*6.4, 4.0))
  plt.subplot(121)

  for i in range(number_trials):
    plt.gca().set_prop_cycle(None)
    plt.plot(tvec,soln[i,:,:]*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
      plt.plot([Tint,Tint],[0,ymax*scale],'k--')
  plt.ylim([0,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Number")

  plt.subplot(122)
  for i in range(number_trials):
    plt.gca().set_prop_cycle(None)
    plt.plot(tvec,soln[i,:,:]*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
    plt.plot([Tint,Tint],[scale/n,ymax*scale],'k--')
  plt.ylim([scale/n,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Number")
  plt.semilogy()
  plt.tight_layout()
  if plotThis==True:
  	plt.savefig('/content/gdrive/My Drive/COVID19 Network Simulations/figures/'+plotName+'.pdf',bbox_inches='tight')
  plt.show()

def plot_iter_cumulative(soln_cum,tvec,n,ymax=1,scale=1,int=0,Tint=0,plotThis=False,plotName="test"):

  """
  plots the output (cumulative prevalence) from a multiple simulation, with or without an intervention. Shows all trajectories
  soln_cum: 3D array of values for each iteration for each variable at each timepoint
  tvec: 1D vector of timepoints
  n: total population size
  ymax : highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: amount to multiple all frequency values by (e.g. "1" keeps as frequency, "N" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  plotThis: True or False, whether a plot will be saved as pdf 
  plotName: string, name of the plot to be saved
  """

  number_trials=np.shape(soln_cum)[0]

  plt.figure(figsize=(2*6.4, 4.0))
  plt.subplot(121)
  for i in range(number_trials):
    plt.gca().set_prop_cycle(None)
    plt.plot(tvec,soln_cum[i,:,:]*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'R', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
      plt.plot([Tint,Tint],[0,ymax*scale],'k--')
  plt.ylim([0,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Cumulative number")

  plt.subplot(122)
  for i in range(number_trials):
    plt.gca().set_prop_cycle(None)
    plt.plot(tvec,soln_cum[i,:,:]*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'RD'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
    plt.plot([Tint,Tint],[scale/n,ymax*scale],'k--')
  plt.ylim([scale/n,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Cumulative number")
  plt.semilogy()
  plt.tight_layout()
  if plotThis==True:
  	plt.savefig('/content/gdrive/My Drive/COVID19 Network Simulations/figures/'+plotName+'.pdf',bbox_inches='tight')
  plt.show()

def plot_iter_shade(soln,tvec,n,ymax=1,scale=1,int=0,Tint=0,loCI=5,upCI=95,plotThis=False,plotName="test"):

  """
  plots the output (prevalence) from a multiple simulation, with or without an intervention. Shows mean and 95% CI
  soln: 3D array of values for each iteration for each variable at each timepoint
  tvec: 1D vector of timepoints
  n: total population size
  ymax : highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: amount to multiple all frequency values by (e.g. "1" keeps as frequency, "N" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  loCI,upCI: Optional, upper and lower percentiles for confidence intervals. Defaults to 90% interval
  plotThis: True or False, whether a plot will be saved as pdf 
  plotName: string, name of the plot to be saved
  """

  soln_avg=np.average(soln,axis=0)
  soln_loCI=np.percentile(soln,loCI,axis=0)
  soln_upCI=np.percentile(soln,upCI,axis=0)
 
  # linear scale
  # add averages
  plt.figure(figsize=(2*6.4, 4.0))
  plt.subplot(121)
  plt.plot(tvec,soln_avg*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  # add ranges
  plt.gca().set_prop_cycle(None)
  for i in range(0,7):
    plt.fill_between(tvec,soln_loCI[:,i]*scale,soln_upCI[:,i]*scale,alpha=0.3)
  if int==1:
      plt.plot([Tint,Tint],[0,ymax*scale],'k--')
  plt.ylim([0,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Number")

  # log scale
  # add averages
  plt.subplot(122)
  plt.plot(tvec,soln_avg*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  # add ranges
  plt.gca().set_prop_cycle(None)
  for i in range(0,7):
    plt.fill_between(tvec,soln_loCI[:,i]*scale,soln_upCI[:,i]*scale,alpha=0.3)
  if int==1:
    plt.plot([Tint,Tint],[scale/n,ymax*scale],'k--')
  plt.ylim([scale/n,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Number")
  plt.semilogy()
  plt.tight_layout()
  if plotThis==True:
  	plt.savefig('/content/gdrive/My Drive/COVID19 Network Simulations/figures/'+plotName+'.pdf',bbox_inches='tight')
  plt.show()

def plot_iter_cumulative_shade(soln_cum,tvec,n,ymax=1,scale=1,int=0,Tint=0,loCI=5,upCI=95,plotThis=False,plotName="test"):

  """
  plots the output (cumulative prevalence) from a multiple simulation, with or without an intervention. Shows mean and 95% CI
  ymax : highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: amount to multiple all frequency values by (e.g. "1" keeps as frequency, "N" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  loCI,upCI: Optional, upper and lower percentiles for confidence intervals. Defaults to 90% interval
  plotThis: True or False, whether a plot will be saved as pdf 
  plotName: string, name of the plot to be saved
  """

  soln_avg=np.average(soln_cum,axis=0)
  soln_loCI=np.percentile(soln_cum,loCI,axis=0)
  soln_upCI=np.percentile(soln_cum,upCI,axis=0)
 
  # linear scale
  # add averages
  plt.figure(figsize=(2*6.4, 4.0))
  plt.subplot(121)
  plt.plot(tvec,soln_avg*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  # add ranges
  plt.gca().set_prop_cycle(None)
  for i in range(0,7):
    plt.fill_between(tvec,soln_loCI[:,i]*scale,soln_upCI[:,i]*scale,alpha=0.3)
  if int==1:
      plt.plot([Tint,Tint],[0,ymax*scale],'k--')
  plt.ylim([0,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Cumulative number")

  # log scale
  # add averages
  plt.subplot(122)
  plt.plot(tvec,soln_avg*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  # add ranges
  plt.gca().set_prop_cycle(None)
  for i in range(0,7):
    plt.fill_between(tvec,soln_loCI[:,i]*scale,soln_upCI[:,i]*scale,alpha=0.3)
  if int==1:
    plt.plot([Tint,Tint],[scale/n,ymax*scale],'k--')
  plt.ylim([scale/n,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Cumulative number")
  plt.semilogy()
  plt.tight_layout()
  if plotThis==True:
  	plt.savefig('/content/gdrive/My Drive/COVID19 Network Simulations/figures/'+plotName+'.pdf',bbox_inches='tight')
  plt.show()

def get_daily_iter(soln_cum,tvec):

  """
  Calculates daily incidence for multiple runs
  soln_cum: 2D array of cumulative values for each variable at each timepoint
  tvec: 1D vector of timepoints
  """
  
  Tmax=int(tvec[-1])
  delta_t=tvec[1]-tvec[0]
  total_steps=int(Tmax/delta_t)

  # get daily incidence

  per_day=int(1/delta_t) # number of entries per day
  days_ind=np.arange(start=0,stop=total_steps,step=per_day)

  soln_inc=np.zeros((np.shape(soln_cum)[0],Tmax-1,np.shape(soln_cum)[2]))

  for i in range(0,7):
    daily_cumulative_history=soln_cum[:,days_ind,i] # first pick out entries corresponding to each day
    soln_inc=index_add(soln_inc,index[:,:,i],daily_cumulative_history[:,1:Tmax]-daily_cumulative_history[:,0:(Tmax-1)]) # then get differences between each day

  return soln_inc

def plot_iter_daily(soln_inc,n,ymax=1,scale=1,int=0,Tint=1,plotThis=False,plotName="test"):

  """
  plots the output (daily incidence) from a multiple simulation, with or without an intervention. Shows all trajectories
  soln_inc: 3D array of values for each iteration for each variable at each timepoint
  tvec: 1D vector of timepoints
  n: total population size
  ymax : highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: amount to multiple all frequency values by (e.g. "1" keeps as frequency, "N" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  plotThis: True or False, whether a plot will be saved as pdf 
  plotName: string, name of the plot to be saved
  """

  number_trials=np.shape(soln_inc)[0]

  tvec=np.arange(1,np.shape(soln_inc)[1]+1)

  plt.figure(figsize=(2*6.4, 4.0))
  plt.subplot(121)

  for i in range(number_trials):
    plt.gca().set_prop_cycle(None)
    plt.plot(tvec,soln_inc[i,:,:]*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
      plt.plot([Tint,Tint],[0,ymax*scale],'k--')
  plt.ylim([0,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Daily incidence")

  plt.subplot(122)
  for i in range(number_trials):
    plt.gca().set_prop_cycle(None)
    plt.plot(tvec,soln_inc[i,:,:]*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  if int==1:
    plt.plot([Tint,Tint],[scale/n,ymax*scale],'k--')
  plt.ylim([scale/n,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Daily incidence")
  plt.semilogy()
  plt.tight_layout()
  if plotThis==True:
  	plt.savefig('/content/gdrive/My Drive/COVID19 Network Simulations/figures/'+plotName+'.pdf',bbox_inches='tight')
  plt.show()

def plot_iter_daily_shade(soln_inc,n,ymax=1,scale=1,int=0,Tint=1,loCI=5,upCI=95,plotThis=False,plotName="test"):

  """
  plots the output (cumulative prevalence) from a multiple simulation, with or without an intervention. Shows mean and 95% CI
  soln_inc: 3D array of values for each iteration for each variable at each timepoint
  tvec: 1D vector of timepoints
  n: total population size
  ymax : highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: amount to multiple all frequency values by (e.g. "1" keeps as frequency, "N" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  loCI,upCI: Optional, upper and lower percentiles for confidence intervals. Defaults to 90% interval
  plotThis: True or False, whether a plot will be saved as pdf 
  plotName: string, name of the plot to be saved
  """

  tvec=np.arange(1,np.shape(soln_inc)[1]+1)

  soln_avg=np.average(soln_inc,axis=0)
  soln_loCI=np.percentile(soln_inc,loCI,axis=0)
  soln_upCI=np.percentile(soln_inc,upCI,axis=0)

  # linear scale
  # add averages
  plt.figure(figsize=(2*6.4, 4.0))
  plt.subplot(121)
  plt.plot(tvec,soln_avg*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  # add ranges
  plt.gca().set_prop_cycle(None)
  for i in range(0,7):
    plt.fill_between(tvec,soln_loCI[:,i]*scale,soln_upCI[:,i]*scale,alpha=0.3)
  if int==1:
      plt.plot([Tint,Tint],[0,ymax*scale],'k--')
  plt.ylim([0,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Daily incidence")

  # log scale
  # add averages
  plt.subplot(122)
  plt.plot(tvec,soln_avg*scale)
  plt.legend(['S', 'E', 'I1', 'I2', 'I3', 'D', 'R'],frameon=False,framealpha=0.0,bbox_to_anchor=(1.04,1), loc="upper left")
  # add ranges
  plt.gca().set_prop_cycle(None)
  for i in range(0,7):
    plt.fill_between(tvec,soln_loCI[:,i]*scale,soln_upCI[:,i]*scale,alpha=0.3)
  if int==1:
    plt.plot([Tint,Tint],[scale/n,ymax*scale],'k--')
  plt.ylim([scale/n,ymax*scale])
  plt.xlabel("Time (days)")
  plt.ylabel("Daily incidence")
  plt.semilogy()
  plt.tight_layout()
  if plotThis==True:
  	plt.savefig('/content/gdrive/My Drive/COVID19 Network Simulations/figures/'+plotName+'.pdf',bbox_inches='tight')
  plt.show()

def get_extinction_time(sol, t):
  """ 
  Calculates the extinction time each of multiple runs
  """
  extinction_time = []
  incomplete_runs = 0
  for i in range(len(sol)):
    extinct = np.where(sol[i][t:] == 0)[0]
    if len(extinct) != 0: 
        extinction_time.append(np.min(extinct))
    else:
        incomplete_runs += 1
    
  #assert extinction_time != [], 'Extinction did not occur for any of the iterations, run simulation for longer'

  if extinction_time == []:
    extinction_time.append(float("inf"))
  
  if incomplete_runs != 0:
    print('Extinction did not occur during %i iterations'%incomplete_runs)

  return extinction_time

def get_peaks_iter(soln,tvec,int=0,Tint=0,loCI=5,upCI=95):

  """
  calculates the peak prevalence for a multiple runs, with or without an intervention
  soln: 3D array of values for each iteration for each variable at each timepoint
  tvec: 1D vector of timepoints
  ymax : highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: amount to multiple all frequency values by (e.g. "1" keeps as frequency, "N" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  loCI,upCI: Optional, upper and lower percentiles for confidence intervals. Defaults to 90% interval
  """

  delta_t=tvec[1]-tvec[0]

  if int==0:
    time_int=0
  else:
    time_int=Tint

  all_cases=soln[:,:,1]+soln[:,:,2]+soln[:,:,3]+soln[:,:,4]

  # Final values
  print('Final recovered: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100 * np.average(soln[:,-1,6]), 100*np.percentile(soln[:,-1,6],loCI), 100*np.percentile(soln[:,-1,6],upCI)))
  print('Final deaths: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100 * np.average(soln[:,-1,5]), 100*np.percentile(soln[:,-1,5],loCI), 100*np.percentile(soln[:,-1,5],upCI)))
  print('Remaining infections: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100*np.average(all_cases[:,-1]),100*np.percentile(all_cases[:,-1],loCI),100*np.percentile(all_cases[:,-1],upCI)))

  # Peak prevalence
  peaks=np.amax(soln[:,:,2],axis=1)
  print('Peak I1: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100 * np.average(peaks),100 * np.percentile(peaks,loCI),100 * np.percentile(peaks,upCI)))
  peaks=np.amax(soln[:,:,3],axis=1)
  print('Peak I2: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100 * np.average(peaks),100 * np.percentile(peaks,loCI),100 * np.percentile(peaks,upCI)))
  peaks=np.amax(soln[:,:,4],axis=1)
  print('Peak I3: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100 * np.average(peaks),100 * np.percentile(peaks,loCI),100 * np.percentile(peaks,upCI)))
  
  # Timing of peaks
  tpeak=np.argmax(soln[:,:,2],axis=1)*delta_t-time_int
  print('Time of peak I1: avg {:4.2f} days, median {:4.2f} days [{:4.2f}, {:4.2f}]'.format(
      np.average(tpeak),np.median(tpeak), np.percentile(tpeak,loCI),np.percentile(tpeak,upCI)))
  tpeak=np.argmax(soln[:,:,3],axis=1)*delta_t-time_int
  print('Time of peak I2: avg {:4.2f} days, median {:4.2f} days [{:4.2f}, {:4.2f}]'.format(
      np.average(tpeak),np.median(tpeak),np.percentile(tpeak,loCI),np.percentile(tpeak,upCI)))
  tpeak=np.argmax(soln[:,:,4],axis=1)*delta_t-time_int
  print('Time of peak I3: avg {:4.2f} days, median {:4.2f} days [{:4.2f}, {:4.2f}]'.format(
      np.average(tpeak),np.median(tpeak),np.percentile(tpeak,loCI),np.percentile(tpeak,upCI)))
  
  # Time when all the infections go extinct
  time_all_extinct = np.array(get_extinction_time(all_cases,0))*delta_t-time_int

  print('Time of extinction of all infections post intervention: {:4.2f} days  [{:4.2f}, {:4.2f}]'.format(
      np.average(time_all_extinct),np.percentile(time_all_extinct,loCI),np.percentile(time_all_extinct,upCI)))
  
  return

def get_peaks_iter_daily(soln_inc,int=0,Tint=0,loCI=5,upCI=95):

  """
  calculates the peak daily incidence for a multiple runs, with or without an intervention
  soln_inc: 3D array of values for each iteration for each variable at each timepoint
  ymax : highest value on y axis, relative to "scale" value (e.g. 0.5 makes ymax=0.5 or 50% for scale=1 or N)
  scale: amount to multiple all frequency values by (e.g. "1" keeps as frequency, "N" turns to absolute values)
  int: Optional, 1 or 0 for whether or not there was an intervention. Defaults to 0
  Tint: Optional, timepoint (days) at which intervention was started
  loCI,upCI: Optional, upper and lower percentiles for confidence intervals. Defaults to 90% interval
  """

  if int==0:
    time_int=0
  else:
    time_int=Tint

  # Peak incidence
  peaks=np.amax(soln_inc[:,:,2],axis=1)
  print('Peak daily I1: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100 * np.average(peaks),100 * np.percentile(peaks,loCI),100 * np.percentile(peaks,upCI)))
  peaks=np.amax(soln_inc[:,:,3],axis=1)
  print('Peak daily I2: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100 * np.average(peaks),100 * np.percentile(peaks,loCI),100 * np.percentile(peaks,upCI)))
  peaks=np.amax(soln_inc[:,:,4],axis=1)
  print('Peak daily I3: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100 * np.average(peaks),100 * np.percentile(peaks,loCI),100 * np.percentile(peaks,upCI)))
  peaks=np.amax(soln_inc[:,:,5],axis=1)
  print('Peak daily deaths: {:4.2f}% [{:4.2f}, {:4.2f}]'.format(
      100 * np.average(peaks),100 * np.percentile(peaks,loCI),100 * np.percentile(peaks,upCI)))

  # Timing of peak incidence  
  tpeak=np.argmax(soln_inc[:,:,2],axis=1)+1.0-time_int
  print('Time of peak I1: avg {:4.2f} days, median {:4.2f} days [{:4.2f}, {:4.2f}]'.format(
      np.average(tpeak),np.median(tpeak),np.percentile(tpeak,5.0),np.percentile(tpeak,95.0)))
  tpeak=np.argmax(soln_inc[:,:,3],axis=1)+1.0-time_int
  print('Time of peak I2: avg {:4.2f} days, median {:4.2f} days [{:4.2f}, {:4.2f}]'.format(
      np.average(tpeak),np.median(tpeak),np.percentile(tpeak,5.0),np.percentile(tpeak,95.0)))
  tpeak=np.argmax(soln_inc[:,:,4],axis=1)+1.0-time_int
  print('Time of peak I3: avg {:4.2f} days, median {:4.2f} days [{:4.2f}, {:4.2f}]'.format(
      np.average(tpeak),np.median(tpeak),np.percentile(tpeak,5.0),np.percentile(tpeak,95.0)))
  tpeak=np.argmax(soln_inc[:,:,5],axis=1)+1.0-time_int
  print('Time of peak deaths: avg {:4.2f} days, median {:4.2f} days [{:4.2f}, {:4.2f}]'.format(
      np.average(tpeak),np.median(tpeak),np.percentile(tpeak,5.0),np.percentile(tpeak,95.0)))

  return

def smooth_timecourse(soln,o):
  """
  replaces each entry with the moving average over time
  soln: solution vector, 3D array, to smooth. Assumes time is second dimension
  o: # of days (entries) on either side of the current value to average over. o=3 -> 1 week
  """
  soln_smooth=soln
  for iter in range(np.shape(soln)[0]):
    for var in range(np.shape(soln)[2]):
      z=moving_average(soln[iter,:,var],1)
      soln_smooth=index_update(soln_smooth,index[iter,:,var],z)
  return soln_smooth

def moving_average(x, o):
  """
  x: array to take moving average og
  o: # of days (entries) on either side of the current value to average over
  """
  w=o*2+1 # width of window to average over, current day in center of window
  y=np.convolve(x, np.ones(w), 'full')
  den=np.concatenate((np.arange(o+1,w),w*np.ones(len(x)-w+1),np.arange(w-1,o,step=-1)))
  z=y[o:-o]/den
  return z

def prob_inf_house_size_iter(state, hh_sizes_, house_dist):
  """ Function that computes the probability of an individual getting infected given their household size.
  @param state : A Device Array that encodes the state of each individual in the population at the end of each iteration of the simulation
  @type : Device Array of shape (# of iterations, population size)
  @param hh_sizes_ : An array which keeps track of the size of each individual's household
  @type : Array of length = population size
  @param house_dist : Distribution of household sizes 
  @type : List or 1D array
  @return : Returns the probability of infection given household size and the mean probability of infection
  @type : Tuple
  """
  hh_sizes = np.asarray(hh_sizes_)
  iterations = len(state)
  prob_hh_size = np.zeros((iterations, len(house_dist)))
  pop = len(state[0])
  mean_inf_prob = np.zeros(iterations)
  
  # First compute the probability of the household size given that the person was infected and then use Bayes rule
  for i in range(iterations):
    if_inf = np.where(state[i] > 0)[0]
    inf_size = len(if_inf)
    hh_inf = hh_sizes[if_inf]
    prob = ((np.array(np.unique(hh_inf, return_counts= True))[-1])/inf_size) * (inf_size/pop) * (1/house_dist) # Bayes rule
    prob_hh_size = index_add(prob_hh_size, i, prob)
    mean_inf_prob = index_add(mean_inf_prob, i, inf_size/pop)

  # Returns the probability of infection given household size
  return np.average(prob_hh_size, axis = 0) , np.average(mean_inf_prob)

def prob_inf_workplace_open(indx_active, state):
  """ Function that computes the probability of infection for an individual who is still working during intervention.
  @param indx_active : Numpy array with indices of individuals still working during intervention
  @type : 1D array
  @param state : A Device Array that encodes the state of each individual in the population at the end of each iteration of the simulation
  @type : Device Array of shape (# of iterations, population size)
  @return : Returns the probability of infection for individuals working during intervention and the population average, averaged over the number of iterations
  @type : Tuple
  """
  iterations = len(state)
  prob_inf_work = np.zeros(iterations)
  pop = len(state[0])
  mean_inf_prob = np.zeros(iterations)

  for i in range(iterations):

    # Get indices of infected people
    if_inf = np.where(state[i] > 0)[0]
    inf_size = len(if_inf)
    
    # Calculate the conditional probability
    prob = (sum(np2.isin(indx_active, if_inf))/inf_size) * (inf_size/pop) * (pop/len(indx_active))
    prob_inf_work = index_add(prob_inf_work, i, prob)
    mean_inf_prob = index_add(mean_inf_prob, i, inf_size/pop)

  return np.average(prob_inf_work), np.average(mean_inf_prob)

def prob_inf_working_hh_member(indx_active, state, house_indices, household_sizes):
    """ Function that computes the probability of infection for individuals living with a household member working during intervention along with the probability of infection for individuals who aren't working and
    living with a working household member.
    @param indx_active : Numpy array with indices of individuals still working during intervention
    @type : 1D array
    @param state : A Device Array that encodes the state of each individual in the population at the end of each iteration of the simulation
    @type : Device Array of shape (# of iterations, population size)
    @param house_indices : Numpy array that keeps track of the house an individual belongs to
    @type : 1D array
    @param household_sizes : Numpy array that keeps track of the size of each individual's household
    @type : 1D array
    @return : Returns the probability of infection for individuals living with working household members, probability for non-workers living with no working household member, and the population average, averaged over the number of iterations
    @type : Tuple
    """
    iterations = len(state)
    prob_inf = np.zeros(iterations)
    prob_inf_not_working = np.zeros(iterations)
    pop = len(state[0])
    mean_inf_prob = np.zeros(iterations)

    for i in range(iterations):

        # Get indices of infected people
        if_inf = np.where(state[i] > 0)[0]
        inf_size = len(if_inf)

        # Houses of people who are still working 
        house_working = np2.unique(house_indices[indx_active])
        
        # Indices of all people who aren't working and their house index
        not_working = np2.setdiff1d(np2.arange(0, pop, 1), indx_active)
        house_not_working = house_indices[not_working]

        # Probability of living with atleast one working household member
        prob_house_working = (sum(np2.isin(house_not_working, house_working))/pop)

        # Probability of living no working household member
        prob_house_not_working = (sum(~np2.isin(house_not_working, house_working))/pop)

        # Indices of infected people who aren't working and their house index
        if_inf_not_working = np2.setdiff1d(if_inf, indx_active)
        house_inf_not_working = house_indices[if_inf_not_working]

        # Probability of infection given atleast one hh member was working during intervention
        prob_1 = (sum(np2.isin(house_inf_not_working, house_working))/inf_size) * (inf_size/pop) * (1/prob_house_working)
        prob_inf = index_add(prob_inf, i, prob_1)

        # Probability of infection given no hh member was working during intervention
        prob_2 = (sum(~np2.isin(house_inf_not_working, house_working))/inf_size) * (inf_size/pop) * (1/prob_house_not_working)
        prob_inf_not_working = index_add(prob_inf_not_working, i, prob_2)

        # Population average probability of infection
        mean_inf_prob = index_add(mean_inf_prob, i, inf_size/pop)

    return np.average(prob_inf), np.average(prob_inf_not_working), np.average(mean_inf_prob)
