import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
from utils.plot import truncate_colormap

def plot_lsvs(dists, highlight_elem, excl = None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlabel(r'Potential vs. RHE [mV]', fontsize=18)
    ax.set_ylabel(r'Current density [mA/cm$^2$]', fontsize=18)
    for j, d in enumerate(dists):
        if excl == None:
            pass
        elif j in excl:
            continue
        ax.plot(np.linspace(0, 900, 500), d['curr'], alpha=0.3,
                color=cmap((d['comp'][highlight_elem] - 0.0) / (0.75 - 0.00)))
    ax.axhline(0.0, linestyle='--', color='k', alpha=0.5)
    ax.set(xlim=(600, 900), ylim=(-0.8, 0.1))
    ax.tick_params(labelsize=14)
    return fig

library_names = ['AgPdPtRu',
                 'AgPdPt',
                 'AgPdRu',
                 'AgPtRu',
                 'PdPtRu',
                 ]

excl_ids = {}

pls = np.linspace(0.0,0.9,500)
cmap = truncate_colormap(plt.get_cmap('plasma'), minval=0.9, maxval=0.0, n=100)

fig, ax = plt.subplots(figsize=(8, 0.5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
for spine in ['right', 'left', 'top','bottom']:
    ax.spines[spine].set_visible(False)
mpl.colorbar.ColorbarBase(ax, cmap=cmap, orientation = 'horizontal')
filename = f'misc/frac_colorbar.png'
fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
print(f'[SAVED] {filename}')
plt.close()

for i, lib in enumerate(library_names):
    elems = [lib[i:i+2] for i in range(len(lib))[::2]]

    with open(f'dist_libraries/{lib}_exp.pkl', 'rb') as input:
        dists = pickle.load(input)

    # identification and exclusion of noisy LSVs
    curr_diff = np.zeros((len(dists),445))

    for j, d in enumerate(dists):
        curr_diff[j,:] = d['curr'][55:]

    for j in np.arange(0,444):
        curr_diff[:,j] = curr_diff[:,j] - curr_diff[:,j+1]

    excl = []
    for j, row in enumerate(curr_diff[:,:-1]):
        if np.any(np.abs(row) > 0.075):
            excl.append(j)
            continue
        if dists[j]['curr'][55] > -0.6:
            excl.append(j)
            continue
        if dists[j]['curr'][0] > -0.25 and lib == 'AgPdPt':
            excl.append(j)
            continue

        comp = np.array([dists[j]['comp'][e] for e in elems])
        if np.any(comp <= 0.0):
            excl.append(j)
            continue

    excl_ids[lib] = excl
    for e in ['Ag','Pd','Pt','Ru']:
        fig = plot_lsvs(dists,e)
        plt.tight_layout()
        filename = f'plots_lsvs/{e}_{lib}.png'
        fig.savefig(filename)
        print(f'[SAVED] {filename}')
        plt.close()
        
    if lib == 'AgPtRu':
        for d in dists:
            d['curr'] = d['curr'] - 0.1032307347701786 # correction obtained from "02_shift_agptru.py"
        with open(f'dist_libraries/{lib}_exp_adj.pkl', 'wb') as output:
            pickle.dump(dists, output)

    for e in ['Ag','Pd','Pt','Ru']:
        fig = plot_lsvs(dists,e,excl=excl)
        plt.tight_layout()
        filename = f'plots_lsvs/{e}_adj_{lib}.png'
        fig.savefig(filename)
        print(f'[SAVED] {filename}')
        plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlabel(r'Potential vs. RHE [mV]', fontsize=18)
    ax.set_ylabel(r'Current density [mA/cm$^2$]', fontsize=18)
    ax.tick_params(labelsize=16)
    for j, d in enumerate(dists):
        if excl == None:
            pass
        elif j in excl:
            continue
        ax.plot(np.linspace(0, 900, 500), d['curr'], alpha=0.3, color=plt.get_cmap('viridis')(-d['curr'][471] / 0.6))
    ax.axvline(850, linestyle='--', color='k', alpha=0.5)
    ax.set(xlim=(600, 900), ylim=(-0.8, 0.1))
    filename = f'plots_lsvs/{lib}_curr.png'
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    fig.savefig(filename)
    print(f'[SAVED] {filename}')
    plt.close()

with open(f'misc/exclude_ids.lst', 'wb') as output:
    pickle.dump(excl_ids, output)

