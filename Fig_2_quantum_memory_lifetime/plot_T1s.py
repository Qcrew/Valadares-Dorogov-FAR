import numpy as np
import pylab as plt
import h5py
import lmfit
from lmfit import Model
from scipy.stats import sem

def create_params(**kwargs):
    """
    Patch method because lmfit does not like working with numpy datatypes
    """
    params = {}
    for name, value in kwargs.items():
        if isinstance(value, dict):
            value = {k: v.item() for k, v in value.items() if isinstance(v, np.number)}
        elif isinstance(value, np.number):
            value = value.item()
        params[name] = value
    return lmfit.create_params(**params)

def doube_exponential_decay(x, amp, alpha0, tau, ofs, n):
        """
        Poissonian distribution given photon projection:
        alpha = alpha0 * exp(-xs / tau)
        """
        alphas = alpha0 * np.exp(-x / 2.0 / tau)
        nbars = alphas**2
        return ofs + amp * nbars**n / 1 * np.exp(-nbars)

def my_cohstate_decay_with_errors(x_values, y_means, y_sems=None, fn=doube_exponential_decay):
    """ """
    def params(y, x):
        """ """
        mul = 1 if (x[-1] > x[0]) else -1
        amp = mul * (y[-1] - y[0])
        if amp < 0:
            ofs = np.max(y)
        else:
            ofs = np.min(y)
        tau = x[-1] / 5
        return create_params(n=0, amp=amp, ofs=ofs, alpha0=1.0, tau=tau)
    if y_sems is None:
        result = Model(fn).fit(y_means, params(y_means, x_values), x=x_values)
    else:
        weights = 1.0 / y_sems
        result = Model(fn).fit(y_means, params(y_means, x_values), x=x_values, weights=weights)
    return result

# Import data
path = "/Users/dorogov/env/Valadares-Dorogov-FAR/Fig_2_quantum_memory_lifetime/Cavity_T1s/"
detunings_data = {"232": ["21-43-40_QubitSpec.hdf5", "#bc343a"],
                  "116": ["17-49-57_QubitSpec.hdf5", "#f19f3c"],
                  "90": ["10-42-53_QubitSpec.hdf5", "#88b742"],
                  "63": ["08-34-52_QubitSpec.hdf5", "#3E9491"]}
# Choose the detuning here
detuning_to_plot = "116"
filename, plot_colour = detunings_data[detuning_to_plot]
with h5py.File(path+filename, "r") as hdf:
        print("Keys: %s" % hdf.keys())
        I_amp = hdf["I"][:, :]
        freqs = hdf["cavity_decay"][:]
        print("Shape of I:", I_amp.shape)
        print("Shape of freqs:", freqs.shape)
I_repetitions = I_amp[:11000, :] # same number of averages for all detunings
I_ = np.mean(I_repetitions, axis=0)
freqs = freqs / 1e6

# Fit
x_values = freqs
y_means = I_ * 1e6
y_sems = sem(I_repetitions)

result = my_cohstate_decay_with_errors(x_values=x_values, y_means=y_means, y_sems=y_sems,
                                                      fn=doube_exponential_decay)
best_fit, best_values = result.best_fit, result.best_values

# Computing the erros
report_ = result.fit_report()
print(report_)

# Plotting
CM = 1 / 2.54
FONTSIZE = 11
fig = plt.figure(figsize=(9*CM, 3.3*CM))
ax = plt.gca()
ax.plot(x_values, y_means, color=plot_colour, marker="o", markeredgewidth=0.25,
        mfc='none', markersize=2.25, linewidth=0)
smoothened_freqs = np.linspace(start=freqs[0], stop=freqs[-1], num=40_000)
ax.plot(smoothened_freqs, doube_exponential_decay(smoothened_freqs, **best_values),
        color=plot_colour, linewidth=0.38)
ax.set(xlim=[-0.05, 4.07])
ax.tick_params(axis='x', labelbottom=False)
ax.tick_params(axis='y', labelleft=False)
plt.show()
# fig.savefig("Cavity_T1_.pdf", bbox_inches="tight", transparent=True)

