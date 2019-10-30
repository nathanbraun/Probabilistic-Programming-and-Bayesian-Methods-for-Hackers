import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
from pandas import DataFrame, Series
import seaborn as sns
import pymc3 as pm

true_prob_a = 0.05
n_a = 1500
true_prob_b = 0.04
n_b = 750

obs_a_data = Series(stats.bernoulli.rvs(true_prob_a, size=n_a), index=range(0, n_a))
obs_b_data = Series(stats.bernoulli.rvs(true_prob_b, size=n_b), index=range(0, n_b))

# set prior on a
# note: can take samples from this
with pm.Model() as model:
    p_a = pm.Uniform('p_a', lower=0, upper=1)
    p_b = pm.Uniform('p_b', lower=0, upper=1)

# now update with actual obs
# CANNOT take samples from this
with model:
    # what does it mean that pm.Bernoulli takes p_a?
    # i guess that we're chaining these things together
    obs_a = pm.Bernoulli('obs_a', p_a, observed=obs_a_data)
    obs_b = pm.Bernoulli('obs_b', p_b, observed=obs_b_data)

# now run model (explained later)
# note: not referring obs_a or obs_b anywhere in here
# fits whole system
with model:
    step = pm.Metropolis()
    trace = pm.sample(18000, step=step)
    burned_trace = trace[1000:]

# now this is a bunch of samples from the posterior
ps_a = Series(burned_trace['p_a'])
ps_b = Series(burned_trace['p_b'])
diff = ps_a - ps_b

ps = pd.concat([
    ps_a.to_frame('p').assign(test = 'a'),
    ps_b.to_frame('p').assign(test = 'b'),
    diff.to_frame('p').assign(test = 'diff')], ignore_index=True)

# now plot
# note: should re-anki FacetGrid thing, didn't remember
plt.clf()
g = (sns.FacetGrid(ps, hue='test', col='test')
     .map(sns.distplot, 'p'))
g.add_legend()
