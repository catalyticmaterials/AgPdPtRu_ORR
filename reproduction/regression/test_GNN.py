import numpy as np
import pickle
from torch_geometric.data import DataLoader
import torch
import matplotlib.pyplot as plt
from utils.regression import load_GCN, test
import glob

def plot_parity(test_pred, test_true, test_site, alloy):
    start, stop = -2, 1.5
    ontop_mask = np.array(test_site) == 'ontop'
    fcc_mask = np.array(test_site) == 'fcc'

    colors = ['steelblue', 'maroon']
    color_list = []
    for entry in test_site:
        if entry == 'ontop':
            color_list.append(colors[0])
        elif entry == 'fcc':
            color_list.append(colors[1])

    for i, site in enumerate(['*OH ontop', '*O fcc']):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        if i == 0:
            ax.set_xlabel(
                r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*OH}}-\Delta \mathrm{E}_{\mathrm{*OH}}^\mathrm{Pt}}$ [eV]',
                fontsize=16)
            ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*OH}} \, [\mathrm{eV}]$', fontsize=24)
        elif i == 1:
            ax.set_xlabel(
                r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*O}}-\Delta \mathrm{E}_{\mathrm{*O}}^\mathrm{Pt}}$ [eV]',
                fontsize=16)
            ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*O}} \, [\mathrm{eV}]$', fontsize=24)
        ax.set_xlim(start, stop)
        ax.set_ylim(start, stop)
        ax.tick_params(labelsize=16)
        ax.text(0.01, 0.98, f'GNN model on {alloy}', family='monospace', fontsize=18, transform=ax.transAxes,
                verticalalignment='top', color='k')
        if i == 0:
            ax.scatter(np.array(test_true)[ontop_mask], np.array(test_pred)[ontop_mask], s=2, c='steelblue', alpha=0.75)
        elif i == 1:
            ax.scatter(np.array(test_true)[fcc_mask], np.array(test_pred)[fcc_mask], s=2, c='maroon', alpha=0.75)

        # plot solid diagonal line
        ax.plot([start, stop], [start, stop], 'k-', linewidth=1.0,
                label=r'$\Delta \mathrm{E}^{\mathrm{pred}} = \Delta \mathrm{E}^{\mathrm{DFT}}$')

        # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
        pm = 0.1
        ax.plot([start, stop], [start + pm, stop + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
        ax.plot([start + pm, stop], [start, stop - pm], 'k--', linewidth=1.0)
        if i == 0:
            ontop_L1loss = np.array(test_pred)[ontop_mask] - np.array(test_true)[ontop_mask]
            ax.text(0.01, 0.93,
                    f'ontop *OH MAE: {np.mean(np.abs(ontop_L1loss)):.3f} eV',
                    family='monospace', fontsize=18, transform=ax.transAxes,
                    verticalalignment='top', color='steelblue')
        elif i == 1:
            fcc_L1loss = np.array(test_pred)[fcc_mask] - np.array(test_true)[fcc_mask]
            ax.text(0.01, 0.93,
                    f'fcc *O MAE:    {np.mean(np.abs(fcc_L1loss)):.3f} eV',
                    family='monospace', fontsize=18, transform=ax.transAxes,
                    verticalalignment='top', color='maroon')

        axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
        axins.patch.set_alpha(0)
        if i == 0:
            axins.hist(ontop_L1loss, bins=20, range=(-3 * pm, 3 * pm), color='steelblue', alpha=0.5)
        elif i == 1:
            axins.hist(fcc_L1loss, bins=20, range=(-3 * pm, 3 * pm), color='maroon', alpha=0.5)
        axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
        axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
        axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
        axins.get_yaxis().set_visible(False)
        for spine in ['right', 'left', 'top']:
            axins.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'parity/{alloy}_{site[1:].replace(" ", "")}_GNN.png')
        plt.close()
    return ontop_L1loss, fcc_L1loss

# set random seeds
np.random.seed(42)
torch.manual_seed(42)

# set Dataloader batch size, learning rate and max epochs
batch_size = 64
max_epochs = 3000
learning_rate = 1e-3

# early stopping is evaluated based on rolling validation error.
# if the val error has not decreased 1% during the prior *patience* number of epochs early stopping is invoked.
roll_val_width = 20  # mean of [current_epoch-roll_val_width/2 : current_epoch+roll_val_width/2 +1]
patience = 100
report_every = 100

# set grid of search parameters
kwargs = {'n_conv_layers': 3,  # number of gated graph convolution layers
          'n_hidden_layers': 0,  # number of hidden layers
          'conv_dim': 18,  # number of fully connected layers
          'act': 'relu', # activation function in hidden layers.
         }

# load training set
paths = glob.glob(f'../features/*.graphs')

train_graphs, val_graphs, test_graphs, all_test_graphs = [], [], {}, []
for p in paths:
    with open(p, 'rb') as input:
        graphs = pickle.load(input)
    np.random.shuffle(graphs)

    train_graphs += graphs[:int(len(graphs) * 0.7)]
    val_graphs += graphs[int(len(graphs) * 0.7):int(len(graphs) * 0.85)]
    test_graphs[p.split('/')[-1][:-7]] = graphs[int(len(graphs) * 0.85):]
    all_test_graphs += graphs[int(len(graphs) * 0.85):]

# load trained state
with open(f'model_states/GC3H0reludim18BS64lr0.001.state', 'rb') as input:
    trained_state = pickle.load(input)

# set model parameters and load trained model
kwargs = {'n_conv_layers': 3,  # number of gated graph convolution layers
          'n_hidden_layers': 0,  # number of hidden layers
          'conv_dim': 18,  # number of fully connected layers
          'act': 'relu' # activation function in hidden layers.
         }
regressor = load_GCN(kwargs,trained_state=trained_state)

OH_list, O_list = [], []
for key in np.sort(list(test_graphs.keys())):
    test_loader = DataLoader(test_graphs[key], batch_size=len(test_graphs[key]))
    _, test_pred, test_true, test_site, test_ads = test(regressor, test_loader, len(test_graphs[key]))

    l = [c.upper() if i % 2 == 0 else c for i, c in enumerate(key)]
    l = ''.join(l)
    if l[-9:] == 'dIrIcHlEt':
        l = l[:-10] + ('(uniform)')
    elif l[-3:] == '3X4':
        l = l[:-4] + ('(3x4)')

    ontop_mae, fcc_mae = plot_parity(test_pred, test_true, test_site, l)
    OH_list.append(ontop_mae)
    O_list.append(fcc_mae)

fig, ax = plt.subplots(figsize=(16, 8))

#for i, alloy in enumerate(test_graphs.keys()):
#    print(f'{alloy},{np.mean(OH_list[i]):.3f},{np.mean(O_list[i]):.3f}')

OH = ax.boxplot(OH_list, positions=np.arange(0, 5*len(OH_list))[::5],
                whis=(5, 95),patch_artist = True, showfliers = False)
O = ax.boxplot(O_list, positions=np.arange(0, 5*len(O_list))[1::5],
                whis=(5, 95),patch_artist = True, showfliers = False)

for patch in OH['boxes']:
    patch.set(facecolor = 'royalblue',
              alpha = 0.75,
              edgecolor = 'mediumblue')
for whisker in OH['whiskers']:
    whisker.set(color ='mediumblue',
                linewidth = 1.5)
for cap in OH['caps']:
    cap.set(color ='mediumblue',
            linewidth = 1.5)
for median in OH['medians']:
    median.set(color ='mediumblue',
               linewidth = 2.0)


for patch in O['boxes']:
    patch.set(facecolor = 'firebrick',
              alpha = 0.75,
              edgecolor = 'darkred')
for whisker in O['whiskers']:
    whisker.set(color ='darkred',
                linewidth = 1.5)
for cap in O['caps']:
    cap.set(color ='darkred',
            linewidth = 1.5)
for median in O['medians']:
    median.set(color ='darkred',
               linewidth = 2.0)

# changing color and linewidth of
# whiskers
labels = np.sort(list(test_graphs.keys()))
for i, l in enumerate(labels):
    l = [c.upper() if i % 2 == 0 else c for i, c in enumerate(l)]
    l = ''.join(l)
    if l[-9:] == 'dIrIcHlEt':
        l = l[:-10] + ('(uniform)')
    elif l[-3:] == '3X4':
        l = l[:-4] + ('(3x4)')
    labels[i] = l

plt.xticks(ticks=np.arange(0, 5*len(OH_list))[::5]+0.5, labels=labels, rotation=90, fontsize=16)
plt.yticks(fontsize=16)
ax.set(xlim = (-5, 5*len(OH_list)))
ax.set_ylabel('Mean error [eV]', fontsize=20)
ax.axhline(0.0,linestyle='-',color='black',alpha=0.5)
ax.axhline(0.1,linestyle='--',color='black',alpha=0.5)
ax.axhline(-0.1,linestyle='--',color='black',alpha=0.5)
plt.tight_layout()
fig.savefig('boxplot.png')


test_loader = DataLoader(all_test_graphs, batch_size=len(all_test_graphs))
_, test_pred, test_true, test_site, test_ads = test(regressor, test_loader, len(all_test_graphs))

start, stop = -2, 1.5
ontop_mask = np.array(test_site) == 'ontop'
fcc_mask = np.array(test_site) == 'fcc'

colors = ['steelblue', 'maroon']
color_list = []
for entry in test_site:
	if entry == 'ontop':
		color_list.append(colors[0])
	elif entry == 'fcc':
		color_list.append(colors[1])

for i, site in enumerate(['*OH ontop', '*O fcc']):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if i == 0:
        ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*OH}}-\Delta \mathrm{E}_{\mathrm{*OH}}^\mathrm{Pt}}$ [eV]',fontsize=16)
        ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*OH}} \, [\mathrm{eV}]$', fontsize=18)
    elif i == 1:
        ax.set_xlabel(r'$\Delta \mathrm{E}^{\mathrm{DFT}}_{\mathrm{*O}}-\Delta \mathrm{E}_{\mathrm{*O}}^\mathrm{Pt}}$ [eV]',fontsize=16)
        ax.set_ylabel(r'$\Delta \mathrm{E}^{\mathrm{pred}}_{\mathrm{*O}} \, [\mathrm{eV}]$', fontsize=18)
    ax.set_xlim(start, stop)
    ax.set_ylim(start, stop)
    ax.text(0.01, 0.98, f'GNN model on AgIrPdPtRu testset', family='monospace', fontsize=18, transform=ax.transAxes,
            verticalalignment='top', color='k')
    if i == 0:
        ax.scatter(np.array(test_true)[ontop_mask], np.array(test_pred)[ontop_mask], s=2, c='steelblue', alpha=0.75)
    elif i == 1:
        ax.scatter(np.array(test_true)[fcc_mask], np.array(test_pred)[fcc_mask], s=2, c='maroon', alpha=0.75)

    # plot solid diagonal line
    ax.plot([start, stop], [start, stop], 'k-', linewidth=1.0,
            label=r'$\Delta \mathrm{E}^{\mathrm{pred}} = \Delta \mathrm{E}^{\mathrm{DFT}}$')
    ax.tick_params(labelsize=14)
    # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
    pm = 0.1
    ax.plot([start, stop], [start + pm, stop + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([start + pm, stop], [start, stop - pm], 'k--', linewidth=1.0)
    if i == 0:
        ontop_L1loss = np.array(test_pred)[ontop_mask] - np.array(test_true)[ontop_mask]
        ax.text(0.01, 0.93,
                f'ontop *OH MAE: {np.mean(np.abs(ontop_L1loss)):.3f} eV',
                family='monospace', fontsize=18, transform=ax.transAxes,
                verticalalignment='top', color='steelblue')
    elif i == 1:
        fcc_L1loss = np.array(test_pred)[fcc_mask] - np.array(test_true)[fcc_mask]
        ax.text(0.01, 0.93,
                f'fcc *O MAE:    {np.mean(np.abs(fcc_L1loss)):.3f} eV',
                family='monospace', fontsize=18, transform=ax.transAxes,
                verticalalignment='top', color='maroon')

    axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
    axins.patch.set_alpha(0)
    if i == 0:
        axins.hist(ontop_L1loss, bins=20, range=(-3 * pm, 3 * pm), color='steelblue', alpha=0.5)
    elif i == 1:
        axins.hist(fcc_L1loss, bins=20, range=(-3 * pm, 3 * pm), color='maroon', alpha=0.5)
    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    axins.axvline(-pm, linestyle='--', linewidth=0.5, color='black')
    axins.axvline(pm, linestyle='--', linewidth=0.5, color='black')
    axins.get_yaxis().set_visible(False)
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'parity_{site[1:].replace(" ", "")}_GNN.png')

