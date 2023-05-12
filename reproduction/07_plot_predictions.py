import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pickle
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from utils.plot import molar_fractions_to_cartesians, prepare_triangle_plot, truncate_colormap
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors

np.random.RandomState(42)

cmap = truncate_colormap(plt.get_cmap('viridis'), minval=1.0, maxval=0.0, n=100)

fig, ax = plt.subplots(figsize=(8, 0.5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
for spine in ['right', 'left', 'top','bottom']:
    ax.spines[spine].set_visible(False)
mpl.colorbar.ColorbarBase(ax, cmap=cmap, orientation = 'horizontal')
filename = f'misc/curr_colorbar.png'
fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
print(f'[SAVED] {filename}')
plt.close()

colors1 = plt.cm.viridis(np.linspace(0., 1, 128))
colors2 = truncate_colormap(plt.get_cmap('autumn'), minval=0.0, maxval=0.91, n=100)(np.linspace(0, 1, 128))
colors = np.vstack((colors2,colors1[::-1]))

xtr_cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

fig, ax = plt.subplots(figsize=(8, 0.5))
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
for spine in ['right', 'left', 'top','bottom']:
    ax.spines[spine].set_visible(False)
mpl.colorbar.ColorbarBase(ax, cmap=xtr_cmap, orientation = 'horizontal')
filename = f'misc/xtr_curr_colorbar.png'
fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
print(f'[SAVED] {filename}')

fog_cmap = np.ones((256, 4))
fog_cmap[:,-1] = np.linspace(0, 0.75, 256)
fog_cmap = ListedColormap(fog_cmap)

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

with open(f'dist_libraries/AgPdPtRu_equi_ternaries.pkl', 'rb') as input:
    equi_dists = pickle.load(input)

l, c, a, y = [], [], [], []
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
            y.append(d['curr'][potid[850]])

def theo_act(dist, G_opt=0.1, a=-1, b=0.5, eU=0.85):
    kb, T = 8.617e-5, 298.15
    j_ki = b*np.exp((-np.abs(dist - G_opt) + 0.86 - eU) / (kb * T))
    j = a/96**2 * np.sum(1 / (1 / (1-b) + 1 / j_ki))
    return j

bnds = {'a':(-np.inf,0),'b':(0,1),'G_opt':(-0.1,0.3)}
start_value = {'a':-1,'b':0.5,'G_opt':0.1}
ads= 'both'

def fit(x0, train_ids, dist, params = ['a','b','G_opt'], ads = 'both'):
    p = dict(zip(params, x0))
    t = np.zeros((len(train_ids),2))
    for n, tid in enumerate(train_ids):
        OH = theo_act(l[tid][('ontop', 'OH', dist)], **p)
        O = theo_act(l[tid][('fcc', 'O', dist)], **p)
        s = OH+O if ads == 'both' else locals()[ads]
        t[n,0], t[n,1] = s, -l[tid]['curr'][potid[850]]
    return np.mean(np.abs(np.sum(t,axis=1)))


kernel = C(constant_value=0.05, constant_value_bounds=(1e-5, 1e1))*RBF(length_scale=0.5, length_scale_bounds=(1e-5, 1e1))
gpr = GPR(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)
gpr.fit(np.array(c),np.array(y))
mae = np.mean(np.abs(gpr.predict(np.array(c))-np.array(y)))

for lib in library_names[1:]:
    mask = np.array(a) != lib
    train_ids = np.arange(len(a))[mask]
    test_ids = np.arange(len(a))[~mask]

    elems = [lib[i:i+2] for i in range(len(lib))[::2]]
    exclude = np.where(~np.isin(['Ag', 'Pd', 'Pt', 'Ru'], elems))[0][0]

    rs, pred, std = [], [], []
    for j, d in enumerate(equi_dists):
        r = np.array([d['comp'][e] for e in ['Ag','Pd','Pt','Ru']])
        if r[exclude] == 0.0:
            rs.append(r)
            p, s = gpr.predict(r.reshape(1,-1), return_std=True)
            pred.append(p[0])
            std.append(s[0])

    rs = np.array(rs)[:,[a for a in range(4) if a != exclude]]
    rs = molar_fractions_to_cartesians(rs)
    fig, ax = plt.subplots(figsize=(8, 8))
    prepare_triangle_plot(ax, elems)
    ax.tricontourf(*rs, np.array(pred), levels=10, cmap=cmap, zorder=0,vmin=-0.6,vmax=0.0)

    filename = f'plots_ternary/{lib}_scatter_GPR.png'
    fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    print(f'[SAVED] {filename}')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    prepare_triangle_plot(ax, elems, labels=False)
    ax.tricontourf(*rs, np.array(std), levels=10, cmap='coolwarm', zorder=0, vmin=0.0, vmax=0.15)
    filename = f'plots_ternary/{lib}_scatter_GPR_std.png'
    fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    print(f'[SAVED] {filename}')
    plt.close()

 
    params, dist = ['a'], 'net'
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

    rs, pred = [], []
    for j, d in enumerate(equi_dists):
        r = np.array([d['comp'][e] for e in ['Ag','Pd','Pt','Ru']])
        if r[exclude] == 0.0:
            rs.append(r)
            OH = theo_act(d[('ontop', 'OH', dist)], **p)
            O = theo_act(d[('fcc', 'O', dist)], **p)
            pred.append(OH + O)
    
    rs = np.array(rs)[:,[a for a in range(4) if a != exclude]]
    rs = molar_fractions_to_cartesians(rs)

    fig, ax = plt.subplots(figsize=(8, 8))
    prepare_triangle_plot(ax, elems, labels=False)
    if lib == 'AgPdRu':
        ax.tricontourf(*rs, np.array(pred), levels=10, cmap=xtr_cmap, zorder=0,vmin=-1.2,vmax=0.0)
    else:
        ax.tricontourf(*rs, np.array(pred), levels=10, cmap=cmap, zorder=0,vmin=-0.6,vmax=0.0)
    vert = np.array([[[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.8, 0.0, 0.2]],
                             [[0.0, 1.0, 0.0], [0.2, 0.8, 0.0], [0.0, 0.8, 0.2]],
                             [[0.0, 0.0, 1.0], [0.2, 0.0, 0.8], [0.0, 0.2, 0.8]]
                             ])
    for v in vert:
        v = molar_fractions_to_cartesians(v)
        ax.add_patch(Polygon(v.T, facecolor='grey', edgecolor='grey', alpha=0.3, hatch='xx'))
    filename = f'plots_ternary/{lib}_scatter_NET_a.png'
    fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    print(f'[SAVED] {filename}')
    plt.close()


    params, dist = ['a','b'], 'gross'
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

    rs, pred = [], []
    for j, d in enumerate(equi_dists):
        r = np.array([d['comp'][e] for e in ['Ag','Pd','Pt','Ru']])
        if r[exclude] == 0.0:
            rs.append(r)
            OH = theo_act(d[('ontop', 'OH', dist)], **p)
            O = theo_act(d[('fcc', 'O', dist)], **p)
            pred.append(OH + O)
    
    rs = np.array(rs)[:,[a for a in range(4) if a != exclude]]
    rs = molar_fractions_to_cartesians(rs)

    fig, ax = plt.subplots(figsize=(8, 8))
    prepare_triangle_plot(ax, elems, labels=False)
    ax.tricontourf(*rs, np.array(pred), levels=10, cmap=cmap, zorder=0,vmin=-0.6,vmax=0.0)
    vert = np.array([[[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.8, 0.0, 0.2]],
                             [[0.0, 1.0, 0.0], [0.2, 0.8, 0.0], [0.0, 0.8, 0.2]],
                             [[0.0, 0.0, 1.0], [0.2, 0.0, 0.8], [0.0, 0.2, 0.8]]
                             ])
    for v in vert:
        v = molar_fractions_to_cartesians(v)
        ax.add_patch(Polygon(v.T, facecolor='grey', edgecolor='grey', alpha=0.3, hatch='xx'))
    filename = f'plots_ternary/{lib}_scatter_GROSS_ab.png'
    fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    print(f'[SAVED] {filename}')
    plt.close()