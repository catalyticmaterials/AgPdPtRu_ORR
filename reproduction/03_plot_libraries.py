import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from tqdm import tqdm
from utils.plot import molar_fractions_to_cartesians, prepare_triangle_plot,\
    truncate_colormap, plot_3Dsimplex, get_cartesian_array_from_barycentric

cmap = truncate_colormap(plt.get_cmap('viridis'), minval=1.0, maxval=0.0, n=100)

library_names = ['AgPdPtRu',
                 'AgPdPt',
                 'AgPdRu',
                 'AgPtRu',
                 'PdPtRu',
                 ]

with open(f'misc/exclude_ids.lst', 'rb') as input:
    exclude_ids = pickle.load(input)

pot_index = {550:305,650:360,750:416,850:471}
pls = np.linspace(0,0.9,500)
lib_list, c, y = [], [], []
for i, lib in enumerate(library_names):
    elems = [lib[i:i+2] for i in range(len(lib))[::2]]

    with open(f'dist_libraries/{lib}_exp_adj.pkl', 'rb') as input:
        dists = pickle.load(input)

    rs, a = [], []
    m = []
    for j, d in enumerate(dists):

        comp = [d['comp'][e] for e in ['Ag','Pd','Pt','Ru']]
        if j not in exclude_ids[lib]:
            for l in [a,y]:
                l.append(d['curr'][pot_index[850]])
            lib_list.append(lib)
            c.append(comp)
            rs.append([d['comp'][e] for e in elems])

    if len(lib) > 6:
        continue

    rs = np.array(rs)
    rs = molar_fractions_to_cartesians(rs)
    fig, ax = plt.subplots(figsize=(8, 8))
    prepare_triangle_plot(ax, elems, labels=False)

    # Plot data
    ax.scatter(*rs, c=np.array(a), s=70, cmap=cmap, marker='o', zorder=0, lw=0.2, vmin=-0.6, vmax=0.0)
    # Save figure
    filename = f'plots_ternary/{lib}_scatter.png'
    fig.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    print(f'[SAVED] {filename}')
    plt.close()

comp_cmap = truncate_colormap(plt.get_cmap('plasma'), minval=0.9, maxval=0.0, n=100)

for lib in library_names:
    for i, e in enumerate(['Ag','Pd','Pt','Ru']):
        fig3d, ax3d = plt.subplots(1,1,figsize=(16, 10),subplot_kw=dict(projection='3d'))
        plot_3Dsimplex(ax3d,['Ag','Pd','Pt','Ru'])
        mask = lib == np.array(lib_list)
        rs3d = np.array(get_cartesian_array_from_barycentric(np.array(c)[mask]))
        ax3d.scatter(rs3d[:, 0], rs3d[:, 1], rs3d[:, 2], s=50, c=np.array(c)[mask,i], cmap=comp_cmap, vmin=0.0, vmax=0.75)
        rs3d = np.array(get_cartesian_array_from_barycentric(np.array(c)[~mask]))
        ax3d.scatter(rs3d[:, 0], rs3d[:, 1], rs3d[:, 2], c='grey',alpha=0.25)
        ax3d.view_init(elev=15., azim=300)
        ax3d.set_axis_off()
        fig3d.tight_layout(rect=[-0.1, -0.3, 1.1, 1.4])
        filename = f'plots_3d/{lib}_{e}_highlighted.png'
        fig3d.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
        print(f'[SAVED] {filename}')
        plt.close()

for i, e in enumerate(['Ag','Pd','Pt','Ru']):
    fig3d, ax3d = plt.subplots(1,1,figsize=(16, 10),subplot_kw=dict(projection='3d'))
    plot_3Dsimplex(ax3d,['Ag','Pd','Pt','Ru'])
    rs3d = np.array(get_cartesian_array_from_barycentric(c))
    ax3d.scatter(rs3d[:, 0], rs3d[:, 1], rs3d[:, 2], c=np.array(c)[:,i], s=50, cmap=comp_cmap, vmin=0.0, vmax=0.75)
    ax3d.view_init(elev=15., azim=300)
    ax3d.set_axis_off()
    fig3d.tight_layout(rect=[-0.1, -0.3, 1.1, 1.4])
    filename = f'plots_3d/{e}_comp.png'
    fig3d.savefig(filename, bbox_inches='tight', dpi=300, transparent=True)
    print(f'[SAVED] {filename}')
    plt.close()

fig3d, ax3d = plt.subplots(1,1,figsize=(16, 10),subplot_kw=dict(projection='3d'))
plot_3Dsimplex(ax3d,['Ag','Pd','Pt','Ru'])
rs3d = np.array(get_cartesian_array_from_barycentric(c))
ax3d.scatter(rs3d[:, 0], rs3d[:, 1], rs3d[:, 2], c=y, s=50, cmap=cmap, vmin=-0.6, vmax=0.0)
ax3d.view_init(elev=15., azim=300)
ax3d.set_axis_off()
fig3d.tight_layout(rect=[-0.1, -0.3, 1.1, 1.4])
filename = f'plots_3d/AgPdPtRu_scatter_shifted.png'
fig3d.savefig(filename, bbox_inches='tight', dpi=300, transparent=False)
print(f'[SAVED] {filename}')
plt.close()

GIF_length = 10
my_images = []

for azi in tqdm(np.linspace(0,360,200),total=200, desc='Making GIF images'):
    fig3d, ax3d = plt.subplots(1,1,figsize=(16, 10),subplot_kw=dict(projection='3d'))
    plot_3Dsimplex(ax3d,['Ag','Pd','Pt','Ru'])
    rs3d = np.array(get_cartesian_array_from_barycentric(c))
    ax3d.scatter(rs3d[:, 0], rs3d[:, 1], rs3d[:, 2], c=y, s=50, cmap=cmap, vmin=-0.6, vmax=0.0)
    ax3d.view_init(elev=15., azim=azi)
    ax3d.set_axis_off()
    fig3d.tight_layout(rect=[-0.1, -0.3, 1.1, 1.4])
    FigureCanvas(fig3d).draw()
    image = np.frombuffer(fig3d.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig3d.canvas.get_width_height()[::-1] + (3,))
    my_images.append(image)
    plt.close()

print('Saving GIF...')
filename = 'plots_3d/AgPdPtRu_scatter_shifted.gif'
imageio.mimsave(filename, my_images, fps=int(len(my_images)/GIF_length))
print(f'[SAVED] {filename}')
plt.close()