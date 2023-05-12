import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pickle
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

np.random.RandomState(42)

def plot_parity(trp, trt, tep, tet, string, limits, comp=[]):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.set_xlabel(r'Exp. current density [mA/cm$^2$]',fontsize=18)
    ax.set_ylabel(r'Pred. current density [mA/cm$^2$]', fontsize=18)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[0], limits[1])

    ax.scatter(np.array(trt), np.array(trp), c='grey', s=10, alpha=0.20)
    if len(comp) == 0:
        ax.scatter(np.array(tet), np.array(tep), c='crimson', s=10, alpha=0.80)
    else:
        ax.scatter(np.array(tet), np.array(tep), c=comp, cmap=cmap, s=10, alpha=0.8, vmin=0.0, vmax=0.75)

    # plot solid diagonal line
    ax.plot([limits[0], limits[1]], [limits[0], limits[1]], 'k-', linewidth=1.0)

    # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
    pm = 0.1
    ax.plot([limits[0], limits[1]], [limits[0] + pm, limits[1] + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([limits[0] + pm, limits[1]], [limits[0], limits[1] - pm], 'k--', linewidth=1.0)

    ax.text(0.01, 0.99, string, family='monospace', fontsize=18, transform=ax.transAxes, va='top', color='k')
    ax.tick_params(labelsize=14)
    return fig

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = truncate_colormap(plt.get_cmap('plasma'), minval=0.9, maxval=0.0, n=100)

library_names = ['AgPdPtRu',
                 'AgPdPt',
                 'AgPdRu',
                 'AgPtRu',
                 'PdPtRu',
                 ]

with open(f'misc/exclude_ids.lst', 'rb') as input:
    exclude_ids = pickle.load(input)

potid = {550:305,650:360,750:416,850:471}
pls = np.linspace(0,0.9,500)

l, c, a = [], [], []
for i, lib in enumerate(library_names):

    elems = [lib[i:i+2] for i in range(len(lib))[::2]]

    with open(f'dist_libraries/{lib}_exp_adj.pkl', 'rb') as input:
        dists = pickle.load(input)

    for j, d in enumerate(dists):
        comp = [d['comp'][e] for e in ['Ag','Pd','Pt','Ru']]
        if j not in exclude_ids[lib]:
            l.append(d)
            c.append(comp)
            a.append(lib)

min_e, max_e = 0, 0
for d in l:
    d_max = max([max(d[('ontop', 'OH', 'gross')]), max(d[('fcc', 'O', 'gross')])])
    d_min = min([min(d[('ontop', 'OH', 'gross')]), min(d[('fcc', 'O', 'gross')])])
    if d_max > max_e:
        max_e = np.ceil(d_max*10)/10 + 0.05
    if d_min < min_e:
        min_e = np.floor(d_min*10)/10 - 0.05

binsize = 0.1
n_bins = int((max_e - min_e) / binsize)

class LR:
    def __init__(self, negative=False):
        self.negative = negative
        if negative:
            self.reg = LinearRegression(fit_intercept=False, positive=True)
    def fit(self,x,y):
        if self.negative:
            self.reg.fit(x, -y)
        else:
            self.reg.fit(x, y)
    def predict(self,x):
        if self.negative:
            return -self.reg.predict(x)
        else:
            return self.reg.predict(x)

kernel = C(constant_value=1, constant_value_bounds=(1e-5, 1e1))*RBF(length_scale=1, length_scale_bounds=(1e-5, 1e5))
GP = GPR(kernel=kernel, alpha=0.25, n_restarts_optimizer=10)
GBDT = GradientBoostingRegressor()
RF = RandomForestRegressor()
reg_label = ['Multiple linear regression','Gaussian process regression','Grad. boost. dec. tree', 'Random Forest']
regressors = [LR(negative=True),GP,GBDT,RF]

for i, reg in enumerate(regressors):
    for dist in ['gross','net']:

        hist_x, y = np.zeros((len(l), n_bins*2)), np.zeros((len(l)))
        for j, d in enumerate(l):
                hOH = np.histogram(d[('ontop', 'OH', dist)], bins=n_bins, range=(min_e, max_e))
                hO = np.histogram(d[('fcc', 'O', dist)], bins=n_bins, range=(min_e, max_e))
                if j == 0: bin_centers = (hOH[1][:-1] + hOH[1][1:]) / 2
                hist_x[j, :] = np.concatenate((hOH[0],hO[0]))
                y[j] = d['curr'][471]

        all_test_err, all_contributions = np.array([]), np.zeros(hist_x.shape)
        for lib in library_names:
            
            mask = np.array(a) != lib
            train_ids = np.arange(len(a))[mask]
            test_ids = np.arange(len(a))[~mask]

            reg.fit(hist_x[train_ids],y[train_ids])

            result = [[],[]]
    
            for j, ids in enumerate([train_ids,test_ids]):
                result[j].append(reg.predict(hist_x[ids,:]))
                result[j].append(y[ids])

            all_test_err = np.concatenate((all_test_err, np.array(result[1][0]) - np.array(result[1][1])))
            
            if i == 0 and dist == 'gross':
                all_contributions[test_ids, :] = hist_x[test_ids, :] * -reg.reg.coef_.T 
                s0 = f'Test set: {lib}'
                s1 = f'Train MAE = {np.mean(np.abs(np.array(result[0][0]) - np.array(result[0][1]))):.3f} mA'
                s2 = f'Test MAE = {np.mean(np.abs(np.array(result[1][0]) - np.array(result[1][1]))):.3f} mA'
                s = '\n'.join([s0,s1,s2])
                fig = plot_parity(*result[0],*result[1], s, [-0.7, 0.1], comp=np.array(c)[test_ids,3])
                plt.tight_layout()
                filename = f'plots_parity/{dist}_LR_{lib}_Ru.png'
                fig.savefig(filename)
                print(f'[SAVED] {filename}')
                plt.close()
        
        if i == 0 and dist == 'gross':
            fig, ax = plt.subplots(2, 1, figsize=(20, 12))
            h = int(all_contributions.shape[1] / 2)

            OH = ax[0].violinplot([all_contributions[:, i] for i in range(h * 2)][:h], positions=range(h), widths=0.75,
                                showextrema=False, showmeans=False, showmedians=False)
            O = ax[1].violinplot([all_contributions[:, i] for i in range(h * 2)][h:], positions=range(h), widths=0.75,
                                showextrema=False, showmeans=False, showmedians=False)

            for patch in OH['bodies']:
                patch.set(facecolor='royalblue',
                        alpha=0.75,
                        edgecolor='mediumblue')
            for patch in O['bodies']:
                patch.set(facecolor='firebrick',
                        alpha=0.75,
                        edgecolor='darkred')

            for j, ax in enumerate(ax):
                ax.set_xticks(range(len(bin_centers))[::2])
                ax.set_xticklabels(labels=[f'{s:.2f}' for s in bin_centers[::2]])
                ax.set_yticks(np.linspace(-0.4,0.0,5))
                ax.tick_params(axis='both', labelsize=16)
                ax.set(ylim=(-0.4,0.025))
                ax.text(0.03, 0.03, fr'*{["OH","O"][j]} contributions', fontsize=24, transform=ax.transAxes, 
                        ha='left',va='bottom',color=['royalblue','firebrick'][j])
                
            text_ax = fig.add_axes([0, 0, 1, 1])
            text_ax.patch.set_alpha(0.0)
            for spine in ['right', 'left', 'top', 'bottom']:
                text_ax.spines[spine].set_visible(False)
            fig.text(0.5, 0.025, r'$\Delta\mathrm{G}_{\mathrm{ads}}-\Delta\mathrm{G}^{\mathrm{Pt}}_{\mathrm{ads}}$ [eV]', fontsize=24, transform=text_ax.transAxes,ha='center') 
            fig.text(0.055, 0.5, r'Contribution to current density [mA/cm$^2$]', rotation=90, fontsize=24, transform=text_ax.transAxes, va='center')
            filename = f'misc/{dist}_LR_cont.png'
            fig.savefig(filename)
            print(f'[SAVED] {filename}')

        print(f'Fit with {reg_label[i]} on {dist} distributions yielded test MAE of {np.mean(np.abs(all_test_err)):.3f} mA')
        