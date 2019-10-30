import pandas as pd
from pandas import Series
import theano.tensor as tt
import numpy as np
import pymc3 as pm

N = 100
X = 35

with pm.Model() as model:
    p = pm.Uniform('freq_cheating', 0, 1)

with model:
    # does testval matter? i don't think so
    true_answers = pm.Bernoulli('truths', p, shape=N, testval=np.random.binomial(1, 0.5, N))

with model:
    first_coin_flips = pm.Bernoulli('first_flips', 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))
    second_coin_flips = pm.Bernoulli('second_flips', 0.5, shape=N, testval=np.random.binomial(1, 0.5, N))

with model:
    val = first_coin_flips*true_answers + (1 - first_coin_flips)*second_coin_flips

    # how to know we have to do float(N) here?
    obs_prop = pm.Deterministic('obs_prop', tt.sum(val)/float(N))

with model:
    obs = pm.Binomial('obs', N, obs_prop, observed=X)

with model:
    # why are we specifying vars here and not in ab example
    step = pm.Metropolis(vars=[p])
    trace = pm.sample(2000, step=step)
    burned_trace = trace[1000:]

ps = Series(burned_trace['freq_cheating'])
(ps < 0.35).mean()
