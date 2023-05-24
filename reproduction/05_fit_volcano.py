import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import minimize
from utils.plot import truncate_colormap

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

with open(f'dist_libraries/AgPdPtRu_equi.pkl', 'rb') as input:
    equi_dists = pickle.load(input)

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

cmap = truncate_colormap(plt.get_cmap('plasma'), minval=0.9, maxval=0.0, n=100)

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

def theo_act(dist, G_opt=0.1, a=-1, b=0.5, eU=0.85):
    kb, T = 8.617e-5, 298.15
    j_ki = b*np.exp((-np.abs(dist - G_opt) + 0.86 - eU) / (kb * T))
    j = a/96**2 * np.sum(1 / (1 / (1-b) + 1 / j_ki))
    return j

bnds = {'a':(-np.inf,0),'b':(0,1),'G_opt':(-0.1,0.3)}
start_value = {'a':-1,'b':0.5,'G_opt':0.1}

def fit(x0, train_ids, dist, params = ['a','b','G_opt'], ads = 'both'):
    p = dict(zip(params, x0))
    t = np.zeros((len(train_ids),2))
    for n, tid in enumerate(train_ids):
        OH = theo_act(l[tid][('ontop', 'OH', dist)], **p)
        O = theo_act(l[tid][('fcc', 'O', dist)], **p)
        s = OH+O if ads == 'both' else locals()[ads]
        t[n,0], t[n,1] = s, -l[tid]['curr'][potid[850]]
    return np.mean(np.abs(np.sum(t,axis=1)))

param_combinations = [['a'],['a','b'],['a','G_opt'],['a','b','G_opt']]
ads = 'both'

for i, params in enumerate(param_combinations):
    for dist in ['gross','net']:
        all_test_err = np.array([])

        for lib in library_names:
            mask = np.array(a) != lib
            train_ids = np.arange(len(a))[mask]
            test_ids = np.arange(len(a))[~mask]
        
            res = minimize(fit, x0=np.array([start_value[k] for k in params]), args=(train_ids,dist,params,ads), 
                        tol=1e-8, bounds=[bnds[k] for k in params], method='L-BFGS-B')
            
            p = dict(zip(params, res.x))
            result = [[], []], [[], []]

            for j, ids in enumerate([train_ids,test_ids]):
                for tid in ids:
                    OH = theo_act(l[tid][('ontop', 'OH', dist)], **p)
                    O = theo_act(l[tid][('fcc', 'O', dist)], **p)
                    s = OH+O if ads == 'both' else locals()[ads]
                    result[j][0].append(s)
                    result[j][1].append(l[tid]['curr'][potid[850]])

            all_test_err = np.concatenate((all_test_err, np.array(result[1][0]) - np.array(result[1][1])))

            if i == 1 and dist == 'gross':
                s0 = f'Testset: {lib}'
                s1 = f'Train MAE = {np.mean(np.abs(np.array(result[0][0]) - np.array(result[0][1]))):.3f} mA'
                s2 = f'Test MAE = {np.mean(np.abs(np.array(result[1][0]) - np.array(result[1][1]))):.3f} mA'
                s3 = [f'{k} = {p[k]:.2f}' for k in p.keys()]
                s = '\n'.join([s0,s1,s2,*s3])
                
                for k, e in enumerate(['Ag','Pd','Pt','Ru']):
                    fig = plot_parity(*result[0],*result[1], s, [-0.7, 0.1], comp=np.array(c)[test_ids,k])
                    plt.tight_layout()
                    filename = f'plots_parity/{dist}_{i}_{lib}_{e}.png'
                    fig.savefig(filename)
                    print(f'[SAVED] {filename}')
                    plt.close()
        
        print(f'Fit with {params} on {dist} distributions yielded test MAE of {np.mean(np.abs(all_test_err)):.3f} mA')
