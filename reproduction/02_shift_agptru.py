import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils.plot import plot_3Dsimplex, get_cartesian_array_from_barycentric

c, y, color = [], [], []

library_names = ['AgPtRu','AgPdPtRu']

with open(f'misc/exclude_ids.lst', 'rb') as input:
    exclude_ids = pickle.load(input)

potid = {550:305,650:360,750:416,850:471}
pls = np.linspace(0,0.9,500)

arr = dict(zip(library_names,[[] for name in library_names]))
dis = dict(zip(library_names,[[] for name in library_names]))

for i, library in enumerate(library_names):

    lib = library.translate({ord(i): None for i in 'hi2'})
    elems = [lib[i:i+2] for i in range(len(lib))[::2]]

    with open(f'dist_libraries/{library}_exp.pkl', 'rb') as input:
        dists = pickle.load(input)

    for j, d in enumerate(dists):
        comp = np.array([d['comp'][e] for e in ['Ag','Pd','Pt','Ru']])
        dis[library].append(d)
        if j not in exclude_ids[library]:
            arr[library].append([j,*comp])

arr[library_names[0]] = np.array(arr[library_names[0]])
arr[library_names[1]] = np.array(arr[library_names[1]])

pairs = []
for i in arr[library_names[0]]:
    distance = np.sqrt(np.sum((arr[library_names[1]][:,1:] - i[1:])**2,axis=1))
    if min(distance) > 0.02:
        continue
    best_match = int(arr[library_names[1]][:,0][distance.argsort()][0])
    if len(pairs) == 1:
        if np.any([i[0] == pairs[0][0],best_match == pairs[0][1]]):
            continue
    if len(pairs) > 1:
        if np.any([i[0] == np.array(pairs)[:,0],best_match == np.array(pairs)[:,1]]):
            continue
    pairs.append([int(i[0]),best_match])

pairs = np.array(pairs)
total = []
for row in pairs:
    total.append(dis[library_names[1]][row[1]]['curr'][potid[850]] - dis[library_names[0]][row[0]]['curr'][potid[850]])
print(f'Correction: {np.mean(total)}')

for i, id in enumerate(pairs[:,0]):
    c.append(arr[library_names[0]][arr[library_names[0]][:,0] == id,1:].tolist()[0])
    y.append(dis[library_names[0]][id]['curr'][potid[850]])
    color.append('Red')

for i, id in enumerate(pairs[:,1]):
    c.append(arr[library_names[1]][arr[library_names[1]][:,0] == id,1:].tolist()[0])
    y.append(dis[library_names[1]][id]['curr'][potid[850]])
    color.append('Blue')

fig3d, ax3d = plt.subplots(1,1,figsize=(16, 10),subplot_kw=dict(projection='3d'))
plot_3Dsimplex(ax3d,['Ag','Pd','Pt','Ru'])
rs3d = np.array(get_cartesian_array_from_barycentric(c))
ax3d.scatter(rs3d[:, 0], rs3d[:, 1], rs3d[:, 2], c=color, s=50)
ax3d.view_init(elev=15., azim=300)
ax3d.set_axis_off()
fig3d.tight_layout(rect=[-0.1, -0.3, 1.1, 1.4])
filename = f'plots_3d/agpdru_shift.png'
fig3d.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
print(f'[SAVED] {filename}')
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.set_xlabel(r'Potential vs. RHE [mV]', fontsize=18)
ax.set_ylabel(r'Current density [mA/cm$^2$]', fontsize=18)

for p in pairs:
    ax.plot(np.linspace(0, 900, 500), dis[library_names[0]][p[0]]['curr'], alpha=0.7, color='red')
    ax.plot(np.linspace(0, 900, 500), dis[library_names[1]][p[1]]['curr'], alpha=0.7, color='blue')
ax.axhline(0.0, linestyle='--', color='k')
ax.set(xlim=(600,900),ylim=(-0.5,0.1))
ax.tick_params(labelsize=14)
plt.tight_layout()
filename = f'plots_lsvs/AgPtRu_comparison.png'
fig.savefig(filename)
print(f'[SAVED] {filename}')
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.set_xlabel(r'Potential vs. RHE [mV]', fontsize=18)
ax.set_ylabel(r'Current density [mA/cm$^2$]', fontsize=18)

for p in pairs:
    ax.plot(np.linspace(0, 900, 500), dis[library_names[0]][p[0]]['curr']+np.mean(total), alpha=0.7, color='red')
    ax.plot(np.linspace(0, 900, 500), dis[library_names[1]][p[1]]['curr'], alpha=0.7, color='blue')
ax.axhline(0.0, linestyle='--', color='k')
ax.set(xlim=(600,900),ylim=(-0.5,0.1))
ax.tick_params(labelsize=14)
plt.tight_layout()
filename = f'plots_lsvs/AgPtRu_shifted.png'
fig.savefig(filename)
print(f'[SAVED] {filename}')
plt.close()