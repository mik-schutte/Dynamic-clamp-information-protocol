'''Docstring
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats


def plot_special(axes, array, col=None, label=None):
    x = np.linspace(np.min(array), np.max(array))
    density = stats.gaussian_kde(array)
    axes.plot(x, density(x), color=col, label=label)
    return


def plot_clampcell_MI(MI_data):
    '''docstring
    '''
    # Load Data
    MI_PC_current = [run['MI'] for run in MI_data['PC_current']]
    MI_PC_dynamic = [run['MI'] for run in MI_data['PC_dynamic']]
    MI_IN_current = [run['MI'] for run in MI_data['IN_current']]
    MI_IN_dynamic = [run['MI'] for run in MI_data['IN_dynamic']]

    ## Statistical data
    PC_N = len(MI_PC_current)
    IN_N = len(MI_IN_current)
    current_means = [np.nanmean(MI_PC_current), np.nanmean(MI_IN_current)]
    current_sem = [np.nanstd(MI_PC_current)/PC_N, np.nanstd(MI_IN_current)/IN_N]
    dynamic_means = [np.nanmean(MI_PC_dynamic), np.nanmean(MI_IN_dynamic)]
    dynamic_sem = [np.nanstd(MI_PC_dynamic)/PC_N, np.nanstd(MI_IN_dynamic)/IN_N]

    # Plot
    sns.set_context('talk')
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(2)
    bar_width = 0.25

    ## Make bars
    b1 = ax.bar(x, height=current_means, label='Current Clamp', yerr=current_sem, capsize=4,
    color='blue', width=bar_width, edgecolor='black')
    b2 = ax.bar(x + bar_width, height=dynamic_means, label='Dynamic Clamp', yerr=dynamic_sem, capsize=4,
    color='green', width=bar_width, edgecolor='black')

    ## Fix x-axis
    ax.set_xticks(x + bar_width/2)
    ax.set_xticklabels(['Pyramidal Cell', 'Interneuron'])

    # Add legend
    plt.legend()

    # Axis styling
    ax.set_ylabel('Mutual Information')
    ax.set_title('Mutual Information in different clamps')
    plt.show()


def plot_regime_compare(pathtofolder):
    #Load
    regimes = ['slow.npy', 'fast.npy', 'slow_high.npy', 'fast_low.npy']
   
    slow = np.load('results/saved/regime_compare/'+regimes[0], allow_pickle=True).item()
    fast = np.load('results/saved/regime_compare/'+regimes[1], allow_pickle=True).item()
    slow_high = np.load('results/saved/regime_compare/'+regimes[2], allow_pickle=True).item()
    fast_low = np.load('results/saved/regime_compare/'+regimes[3], allow_pickle=True).item()

    # Plot
    fig, axs = plt.subplots(ncols=2, figsize=(10, 10))
    plot_special(axs[0], slow['input'], col='blue', label='Slow')
    plot_special(axs[0], fast['input'], col='red', label='Fast')
    plot_special(axs[0], slow_high['input'], col='green', label='Slow High')
    plot_special(axs[0], fast_low['input'], col='purple', label='Fast Low')
    axs[0].set(xlabel='Input Current [nA]')
    axs[0].title.set_text('Input Current distribution')

    plot_special(axs[1], slow['potential'], col='blue', label='Slow')
    plot_special(axs[1], fast['potential'], col='red', label='Fast')
    plot_special(axs[1], slow_high['potential'], col='green', label='Slow High')
    plot_special(axs[1], fast_low['potential'], col='purple', label='Fast Low')
    axs[1].set(xlabel='Membrane Potential [mV]')
    axs[1].title.set_text('Membrane Potential distribution')
    plt.legend()
    plt.show()
