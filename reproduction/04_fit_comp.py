import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pickle
from utils.plot import  truncate_colormap
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

np.random.RandomState(42)

def plot_parity(trp, trt, tep, tet, string, limits, comp=None):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlabel(r'Exp. current density [mA/cm$^2$]',fontsize=18)
    ax.set_ylabel(r'Pred. current density [mA/cm$^2$]', fontsize=18)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[0], limits[1])

    ax.scatter(np.array(trt), np.array(trp), c='grey', s=10, alpha=0.3)
    if comp.all() != None:
        ax.scatter(np.array(tet), np.array(tep), c=comp, cmap=cmap, s=10, alpha=0.75, vmin=0.0, vmax=0.75)
    else:
        ax.scatter(np.array(tet), np.array(tep), c='crimson', s=10, alpha=0.75)

    # plot solid diagonal line
    ax.plot([limits[0], limits[1]], [limits[0], limits[1]], 'k-', linewidth=1.0)

    # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
    pm = 0.1
    ax.plot([limits[0], limits[1]], [limits[0] + pm, limits[1] + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([limits[0] + pm, limits[1]], [limits[0], limits[1] - pm], 'k--', linewidth=1.0)

    ax.text(0.01, 0.99, string, family='monospace', fontsize=18, transform=ax.transAxes, va='top', color='k')

    axins = ax.inset_axes([0.65, 0.1, 0.3, 0.3])
    axins.patch.set_alpha(0)
    axins.hist(np.array(trp) - np.array(trt), bins=20, range=(limits[0]/2,-limits[0]/2), color='grey', alpha=0.5)
    axins.hist(np.array(tep) - np.array(tet), bins=20, range=(limits[0]/2,-limits[0]/2), color='crimson', alpha=0.5)
    axins.axvline(0.0, linestyle='-', linewidth=0.5, color='black')
    ax.tick_params(labelsize=14)
    axins.get_yaxis().set_visible(False)
    for spine in ['right', 'left', 'top']:
        axins.spines[spine].set_visible(False)
    return fig

def theo_act(dist, G_opt=0.10, scale=1.0, eU=0.85, j_d=1):
    kb, T = 8.617e-5, 298.15
    j_ki = np.exp((-np.abs(dist - G_opt) + 0.86 - eU) / (kb * T))
    j = 1/96**2 * np.sum(scale / (1 / j_d + 1 / j_ki))
    return j

cmap = truncate_colormap(plt.get_cmap('viridis'), minval=1.0, maxval=0.0, n=100)

library_names = ['AgPdPtRu',
                 'AgPdPt',
                 'AgPdRu',
                 'AgPtRu',
                 'PdPtRu',
                 ]

with open(f'misc/exclude_ids.lst', 'rb') as input:
    exclude_ids = pickle.load(input)

cmap = truncate_colormap(plt.get_cmap('plasma'), minval=1.0, maxval=0.0, n=100)

potid = {550:305,650:360,750:416,850:471}
pls = np.linspace(0,0.9,500)

with open(f'dist_libraries/AgPdPtRu_equi.pkl', 'rb') as input:
    equi_dists = pickle.load(input)

l, c, a = [], [], []
for i, lib in enumerate(library_names):

    if lib == 'AgPtRu':
        with open(f'dist_libraries/{lib}_exp_adj.pkl', 'rb') as input:
            dists = pickle.load(input)
    else:
        with open(f'dist_libraries/{lib}_exp.pkl', 'rb') as input:
            dists = pickle.load(input)

    for j, d in enumerate(dists):
        comp = [d['comp'][e] for e in ['Ag','Pd','Pt','Ru']]
        if j not in exclude_ids[lib]:
            l.append(d)
            c.append(comp)
            a.append(lib)

# Define kernel to use for Gaussian process regressors
kernel = C(constant_value=0.05, constant_value_bounds=(1e-5, 1e1))\
        *RBF(length_scale=0.5, length_scale_bounds=(1e-5, 1e1))\

# Define Gaussian process regressor
gpr = GPR(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)

reg = [LinearRegression(),gpr, GradientBoostingRegressor(),RandomForestRegressor()]
reg_label = ['Multiple linear regression','Gaussian process regression','Grad. boost. dec. tree', 'Random Forest']
reg_file = ['LR','GP','GBDT','RF']

for i, r in enumerate(reg):
    all_test_err = np.array([])
    comp_x, y = np.zeros((len(l), 4)), np.zeros((len(l)))
    for j, d in enumerate(l):
        comp_x[j, :] = c[j]
        y[j] = d['curr'][potid[850]]

    for j, lib in enumerate(library_names):

        mask = np.array(a) != lib
        train_ids = np.arange(len(a))[mask]
        test_ids = np.arange(len(a))[~mask]

        r.fit(comp_x[train_ids],y[train_ids])
        trp = r.predict(comp_x[train_ids])
        trt = y[train_ids]
        tep = r.predict(comp_x[test_ids])
        tet = y[test_ids]

        test_c = np.array(c)[test_ids, 3]

        all_test_err = np.concatenate((all_test_err, np.array(tep) - np.array(tet)))

    print(f'Fit on comp with {reg_label[i]} yielded test MAE of {np.mean(np.abs(all_test_err)):.3f} mA')