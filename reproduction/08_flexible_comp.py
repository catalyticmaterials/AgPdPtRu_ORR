import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.optimize import minimize
from matplotlib.patches import Polygon
from utils.plot import count_elements, get_molar_fractions, get_simplex_vertices,\
    molar_fractions_to_cartesians, make_triangle_ticks, prepare_triangle_plot,\
    truncate_colormap, plot_3Dsimplex, get_cartesian_array_from_barycentric
from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from matplotlib import gridspec


def plot_parity(ax, tep, tet, string, limits, comp=None):
    ax.set_xlabel(r'Exp. current density [mA/cm$^2$]',fontsize=16)
    ax.set_ylabel(r'Pred. current density [mA/cm$^2$]', fontsize=16)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[0], limits[1])

    try:
        ax.scatter(np.array(tet), np.array(tep), c=comp, cmap=cmap, s=10, alpha=0.75, vmin=-0.7, vmax=0.00)
    except:
        ax.scatter(np.array(tet), np.array(tep), c='crimson', s=10, alpha=0.75)

    # plot solid diagonal line
    ax.plot([limits[0], limits[1]], [limits[0], limits[1]], 'k-', linewidth=1.0)

    # plot dashed diagonal lines 0.1 eV above and below solid diagonal line
    pm = 0.1
    ax.plot([limits[0], limits[1]], [limits[0] + pm, limits[1] + pm], 'k--', linewidth=1.0, label=r'$\pm %.2f \mathrm{eV}$' % pm)
    ax.plot([limits[0] + pm, limits[1]], [limits[0], limits[1] - pm], 'k--', linewidth=1.0)

    ax.text(0.01, 0.99, string, family='monospace', fontsize=18, transform=ax.transAxes, va='top', color='k')


with open(f'dist_libraries/AgPdPtRu_equi_ternaries.pkl', 'rb') as input:
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

cmap = truncate_colormap(plt.get_cmap('viridis'), minval=1.0, maxval=0.0, n=100)

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

l, c, a = np.array(l), np.array(c), np.array(a)

def theo_act(dist, G_opt=0.10, a=-1, b=0.5, eU=0.85, offset=0):
    kb, T = 8.617e-5, 298.15
    j_ki = b*np.exp((-np.abs(dist - G_opt) + 0.86 - eU) / (kb * T))
    j = a/96**2 * np.sum(1 / (1 / (1-b) + 1 / j_ki))
    return offset + j

def fit(p, train_ids, dist):
    t = np.zeros((len(train_ids),2))
    for n, tid in enumerate(train_ids):
        OH = theo_act(l[tid][('ontop', 'OH', dist)], a=p[0], b=p[1])
        O = theo_act(l[tid][('fcc', 'O', dist)], a=p[0], b=p[1])
        t[n,0], t[n,1] = OH+O, -l[tid]['curr'][potid[850]]
    return np.mean(np.abs(np.sum(t,axis=1)))

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

all_true = np.array([])
for lib in library_names:
    if lib == 'AgPdPtRu':
        continue

    mask = np.array(a) != lib
    train_ids = np.arange(len(a))[mask]
    test_ids = np.arange(len(a))[~mask]

    res = minimize(fit, x0=np.array([start_value[k] for k in ['a','b']]), args=(train_ids,'gross',['a','b'],'both'), 
                tol=1e-8, bounds=[bnds[k] for k in ['a','b']], method='L-BFGS-B')

    elems = [lib[i:i+2] for i in range(len(lib))[::2]]
    exclude = np.where(~np.isin(['Ag', 'Pd', 'Pt', 'Ru'], elems))[0][0]
    rs, pred = [], []
    for j, d in enumerate(equi_dists):
        r = [d['comp'][e] for e in ['Ag','Pd','Pt','Ru']]
        if r[exclude] == 0.0:
            rs.append(r)
            OH = theo_act(d[('ontop', 'OH', 'gross')], a=res.x[0], b=res.x[1])
            O = theo_act(d[('fcc', 'O', 'gross')], a=res.x[0], b=res.x[1])
            pred.append(OH + O)

    max_steps = 50
    cutoff = 0.05
    true_list = []
    c_arr, pred_arr = np.zeros((len(test_ids),max_steps,4)), np.zeros((len(test_ids),max_steps))
    rs, pred = np.array(rs), np.array(pred)

    for i, j in enumerate(test_ids):
        act_c_arr, act_pred_arr = [], []
        true = l[j]['curr'][potid[850]]
        true_list.append(true)
        c_actual = c[j]
        c_arr[i,0,:] = c_actual

        OH = theo_act(l[j][('ontop', 'OH', 'gross')], a=res.x[0], b=res.x[1])
        O = theo_act(l[j][('fcc', 'O', 'gross')], a=res.x[0], b=res.x[1])
        pred_arr[i,0] = OH+O

        grid_id = np.argmin(np.sqrt(np.sum((rs-c_actual)**2,axis=1)))
        gc, gp = rs[grid_id], pred[grid_id]
        counter = 0
        break_flag = False

        while True:
            if break_flag == True:
                break
            counter += 1
            if counter == max_steps-1:
                break_flag = True

            c_arr[i,counter, :] = gc
            pred_arr[i,counter] = gp

            d = np.sqrt(np.sum((rs-gc)**2,axis=1))
            sorted = d.argsort()
            for id in sorted[1:]:
                if d[id] > cutoff:
                    break_flag = True
                    break
                abs_diff = np.abs(pred[id] - true)
                if abs_diff < (np.abs(gp-true)):
                    gc, gp = rs[id], pred[id]
                    break
  
    for i, j in enumerate(test_ids):
        try:
            del(locked_c)
            del(locked_p)
        except:
            pass
        for w in range(max_steps):
            if np.all(c_arr[i,w,:] == 0):
                try:
                    c_arr[i, w, :] = locked_c
                    pred_arr[i, w] = locked_p
                except:
                    locked_c = c_arr[i,w-1,:]
                    locked_p = pred_arr[i,w-1]
                    c_arr[i, w, :] = locked_c
                    pred_arr[i, w] = locked_p
    try:
        initial_comp = np.r_[initial_comp,c_arr[:,0,:]]
        final_comp = np.r_[final_comp,c_arr[:,-1,:]]
    except:
        initial_comp = c_arr[:,0,:]
        final_comp = c_arr[:,-1,:]
    all_true = np.concatenate((all_true,np.array(true_list)))

    fig, ax = plt.subplots(figsize=(8, 8))
    prepare_triangle_plot(ax, elems, labels=False)
    rs = c_arr[:, -1, [a for a in range(4) if a != exclude]]
    rs = molar_fractions_to_cartesians(rs)
    ax.scatter(*rs, c=np.array(true_list), s=70, cmap=cmap, marker='o', zorder=0, lw=0.2, vmin=-0.6, vmax=0.0)
    filename = f'plots_ternary/{lib}_scatter_flex.png'
    fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    plt.close(fig)
    print(f'[SAVED] {filename}')

fig3d, ax3d = plt.subplots(1,1,figsize=(16, 10),subplot_kw=dict(projection='3d'))
plot_3Dsimplex(ax3d,['Ag','Pd','Pt','Ru'])
rs3d = np.array(get_cartesian_array_from_barycentric(initial_comp))
ax3d.scatter(rs3d[:, 0], rs3d[:, 1], rs3d[:, 2], c=all_true, s=50, cmap=cmap, vmin=-0.6, vmax=0.0)
ax3d.view_init(elev=15., azim=300)
ax3d.set_axis_off()
fig3d.tight_layout(rect=[-0.1, -0.3, 1.1, 1.4])
filename = f'plots_3d/ternary_initial.png'
fig3d.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig3d)
print(f'[SAVED] {filename}')

fig3d, ax3d = plt.subplots(1,1,figsize=(16, 10),subplot_kw=dict(projection='3d'))
plot_3Dsimplex(ax3d,['Ag','Pd','Pt','Ru'])
rs3d = np.array(get_cartesian_array_from_barycentric(final_comp))
ax3d.scatter(rs3d[:, 0], rs3d[:, 1], rs3d[:, 2], c=all_true, s=50, cmap=cmap, vmin=-0.6, vmax=0.0)
ax3d.view_init(elev=15., azim=300)
ax3d.set_axis_off()
fig3d.tight_layout(rect=[-0.1, -0.3, 1.1, 1.4])
filename = f'plots_3d/ternary_final.png'
fig3d.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
plt.close(fig3d)
print(f'[SAVED] {filename}')