from matplotlib.ticker import AutoMinorLocator
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
#from .misc import uncertainty
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
import itertools as it
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from tqdm import tqdm

def format_ax(ax,xlabel,ylabel,ticklabel_size=10,axlabel_size=12, put_minor=True):
    ax.yaxis.set_tick_params(labelsize=ticklabel_size)
    ax.xaxis.set_tick_params(labelsize=ticklabel_size)
    if put_minor:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='minor', length=3)
        ax.tick_params(which='major', length=6)
    ax.set_xlabel(xlabel, fontsize=axlabel_size)
    ax.set_ylabel(ylabel, fontsize=axlabel_size)

def plot_cv(output, pm, metal_labels, regressor_label, colormap=True,no_color=False):
    train_score, train_std = output[0],output[1]
    test_score, test_std = output[2], output[3]
    train_pred, train_targets = output[4],output[6]
    test_pred, test_targets = output[5], output[7]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    n_samples = 0
    temp_min, temp_max = [], []
    for i in range(len(train_pred)):
        if no_color:
            ax.scatter(train_targets[i], train_pred[i], marker='o', s=5, alpha=0.5,
                       cmap=get_cmap('gist_rainbow'), c=np.arange(len(train_pred[i])))
            ax.scatter(test_targets[i], test_pred[i], marker='x', s=50, alpha=0.5,
                       cmap=get_cmap('gist_rainbow'), c=np.arange(len(test_pred[i])))
        elif colormap:
            cmap = get_colormap(get_color(metal_labels[0][:2]), get_color(metal_labels[1][:2]))
            ax.scatter(train_targets[i],train_pred[i],marker='o',s=5, alpha=0.75,
                       label= metal_labels[i], color=cmap(float(i/(len(train_pred)-1))))
            ax.scatter(test_targets[i], test_pred[i], marker='x', s=50, alpha=0.75,
                       color=cmap(float(i / (len(test_pred) - 1))))
        else:
            ax.scatter(train_targets[i], train_pred[i], marker='o', s=5, alpha=0.75,
                       label= metal_labels[i][:2], color=get_color(metal_labels[i][:2]))
            ax.scatter(test_targets[i], test_pred[i], marker='x', s=50, alpha=0.75,
                       color=get_color(metal_labels[i][:2]))

        n_samples += len(train_targets[i])
        n_samples += len(test_targets[i])
        temp_min.append(np.min(train_targets[i]))
        temp_min.append(np.min(train_pred[i]))
        temp_min.append(np.min(test_targets[i]))
        temp_min.append(np.min(test_pred[i]))
        temp_max.append(np.max(train_targets[i]))
        temp_max.append(np.max(train_pred[i]))
        temp_max.append(np.max(test_targets[i]))
        temp_max.append(np.max(test_pred[i]))

    min = np.min(temp_min) - 0.1
    max = np.max(temp_max) + 0.1

    ax.plot([min,max], [min,max], 'k-', linewidth=1.0)
    ax.plot([min,max], [min+pm,max+pm], 'k--', linewidth=1.0)
    ax.plot([min,max], [min-pm,max-pm], 'k--', linewidth=1.0)
    format_ax(ax, r'E$_{\mathrm{DFT}}$-E$_{\mathrm{DFT}}^{\mathrm{Pt}}$ [eV]', r'E$_{\mathrm{pred}}$-E$_{\mathrm{DFT}}^{\mathrm{Pt}}$ [eV]')
    ax.set(xlim=(min,max),ylim=(min,max))
    ax.text(0.02,0.98,f'{regressor_label}\nTrain MAE = {train_score:.3f}({uncertainty(train_std,3)})eV\nTest MAE = {test_score:.3f}({uncertainty(test_std,3)})eV\n' + r'N$_{\mathrm{samples}}$= '+ str(n_samples), family='monospace', transform=ax.transAxes,
              fontsize=14, verticalalignment='top', horizontalalignment='left', color='black')
    if not no_color:
        ax.legend(loc='lower right', fontsize=14, markerscale=3)
    plt.tight_layout()

    return fig

def get_color(metal_label, whiteout_param=0):

    color_dict = {'Ag':np.array([192,192,192]) / 256,
                  'Ir': np.array([0,85,138]) / 256,
                  'Pd': np.array([0,107,136]) / 256,
                  'Pt': np.array([208,208,224]) / 256,
                  'Ru': np.array([0,146,144]) / 256,
                  }

    return color_dict[metal_label] * (1 - whiteout_param) + whiteout_param

def get_dark_color(metal_label):
    color_dict = {'Ag': np.array([192, 192, 192])/2 / 256,
                  'Ir': np.array([0, 85, 138])/2 / 256,
                  'Pd': np.array([0, 107, 136])/2 / 256,
                  'Pt': np.array([208, 208, 224])/2 / 256,
                  'Ru': np.array([0, 146, 144])/2 / 256,
                  }
    return color_dict[metal_label]

def get_colormap(color1,color2):
    vals = np.ones((256, 3))
    vals[:, 0] = np.linspace(color1[0], color2[0], 256)
    vals[:, 1] = np.linspace(color1[1], color2[1], 256)
    vals[:, 2] = np.linspace(color1[2], color2[2], 256)
    return ListedColormap(vals)

def find_maxmin(list):
    all_max, all_min = None, None
    for ens in list:
        ens = np.array(ens)
        if all_max != None and all_min != None:
            if max(ens[:,-4]) > all_max:
                all_max = max(ens[:,-4])
            if min(ens[:,-4]) < all_min:
                all_min = min(ens[:,-4])
        else:
            all_max, all_min = max(ens[:,-4]), min(ens[:,-4])
    return all_min-0.2, all_max+0.2

def plot_histogram(ensemble_array,alloy_label,sites,adsorbate,bin_width,pure_eads, min_E, max_E):
    #min_E, max_E = find_maxmin(ensemble_array)

    bins = int((max_E-min_E)/bin_width)

    metals = []
    for i in range(int(len(alloy_label)/2)):
        metals.append(alloy_label[i*2:i*2+2])

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.hist(np.array([item for sublist in ensemble_array for item in sublist])[:, -4], bins=bins,
            range=(min_E, max_E), color='black', histtype='bar', alpha=0.3, label='Total')
    ax.hist(np.array([item for sublist in ensemble_array for item in sublist])[:, -4], bins=bins,
            range=(min_E, max_E), color='black', histtype='step', alpha=0.5)

    vert_list = [0.83, 0.77, 0.71, 0.65, 0.59]

    for i, ensemble in enumerate(ensemble_array):
        ens = np.array(ensemble)
        if adsorbate == 'OH':
            color, darkcolor = get_color(sites[i]), get_dark_color(sites[i])
        elif adsorbate == 'O' and len(metals) == 2:
            cmap = get_colormap(get_color(sites[0][:2]), get_color(sites[3][:2]))
            color = cmap(float(i/(len(ensemble_array)-1)))
        else:
            color = get_cmap('gist_rainbow')(float(i/(2)))

        ax.hist(ens[:, -4], bins=bins, range=(min_E, max_E), color=color, histtype='bar', alpha=0.5)
        ax.hist(ens[:, -4], bins=bins, range=(min_E, max_E), color=color, histtype='step')

        if len(metals) == 2 or not adsorbate == 'O':
            print(len(sites[i]))
            if len(sites[i]) > 6:
                if np.mean(ens[:, -4]) < 0:
                    d = sites[i] + r' {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))
                else:
                    d = sites[i] + r'  {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))
            else:
                if np.mean(ens[:, -4]) < 0:
                    d = sites[i] + r'   {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))
                else:
                    d = sites[i] + r'    {:.3f} ({:.3f})'.format(np.mean(ens[:, -4]), np.std(ens[:, -4]))

            ax.text(0.02, vert_list[i], d, family='monospace', transform=ax.transAxes,
                        fontsize=14, color=color, verticalalignment='top')

            ylim = ax.get_ylim()[1]*1.1

            if adsorbate == 'O' and len(metals) < 2:
                pass
            else:
                ax.text(pure_eads[sites[i][:2]], ylim / 12, sites[i][:2], family='monospace', fontsize=14,
                        verticalalignment='bottom', horizontalalignment='center',zorder=10)
                ax.arrow(pure_eads[sites[i][:2]], ylim  / 12, 0, -ylim  / 12 + 0.2,
                             head_width=(max_E - min_E) / 100, head_length=ylim / 30, length_includes_head=True,
                             ec='black', fill=False,zorder=10)


    ax.set(xlim=(min_E, max_E), ylim=(0,ax.get_ylim()[1]*1.3))
    ax.set_xlabel(r'$\Delta \mathrm{E}_{\mathrm{OH}}-\Delta \mathrm{E}_{\mathrm{OH}}^\mathrm{Pt}}$ [eV]', fontsize=20)
    ax.set_ylabel('Frequency [binsize: {:.3f} eV]'.format((max_E - min_E) / bins), fontsize=20)
    ax.text(0.01, 0.98, f'$^*${adsorbate} ' + alloy_label, family='monospace', transform=ax.transAxes, fontsize=18,
            color='black', verticalalignment='top')
    if len(metals) == 2 or not adsorbate == 'O':
        ax.text(0.01, 0.90, r'Ens.     $\mu_{\Delta E}$   ($\sigma_{\Delta E}$)  [eV]', family='monospace',
                transform=ax.transAxes, fontsize=14, color='black', verticalalignment='top')
    ax.yaxis.set_tick_params(labelsize=16)
    ax.xaxis.set_tick_params(labelsize=16, size=6, width=1.5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_tick_params(which='minor', size=3, width=1)
    plt.tight_layout()

    number_of_samples = len(np.array([item for sublist in ensemble_array for item in sublist])[:, -1])
    ax.text(0.98, 0.98, str(number_of_samples) + r' samples', family='monospace', transform=ax.transAxes, fontsize=16,
            color='black', verticalalignment='top', horizontalalignment='right')

    return fig


def count_elements(elements, n_elems):
    count = np.zeros(n_elems, dtype=int)
    for elem in elements:
        count[elem] += 1
    return count

def get_molar_fractions(step_size, n_elems, total=1., return_number_of_molar_fractions=False):
    'Get all molar fractions with the given step size'

    interval = int(total / step_size)
    n_combs = scipy.special.comb(n_elems + interval - 1, interval, exact=True)

    if return_number_of_molar_fractions:
        return n_combs

    counts = np.zeros((n_combs, n_elems), dtype=int)

    for i, comb in enumerate(it.combinations_with_replacement(range(n_elems), interval)):
        counts[i] = count_elements(comb, n_elems)

    return counts * step_size

def get_simplex_vertices(n_elems):
    # Initiate array of vertice coordinates
    vertices = np.zeros((n_elems, n_elems - 1))

    for idx in range(1, n_elems):
        # Get coordinate of the existing dimensions as the
        # mean of the existing vertices
        vertices[idx] = np.mean(vertices[:idx], axis=0)

        # Get the coordinate of the new dimension by ensuring it has a unit
        # distance to the first vertex at the origin
        vertices[idx][idx - 1] = (1 - np.sum(vertices[idx][:-1] ** 2)) ** 0.5

    return vertices

def molar_fractions_to_cartesians(fs):
    # Make into numpy
    fs = np.asarray(fs)

    if fs.ndim == 1:
        fs = np.reshape(fs, (1, -1))

    # Get vertices of the multidimensional simplex
    n_elems = fs.shape[1]
    vertices = get_simplex_vertices(n_elems)
    vertices_matrix = vertices.T

    # Get cartisian coordinates corresponding to the molar fractions
    return np.dot(vertices_matrix, fs.T)

def make_triangle_ticks(ax, start, stop, tick, n, offset=(0., 0.),
                        fontsize=16, ha='center', tick_labels=True):
    r = np.linspace(0, 1, n + 1)
    x = start[0] * (1 - r) + stop[0] * r
    x = np.vstack((x, x + tick[0]))
    y = start[1] * (1 - r) + stop[1] * r
    y = np.vstack((y, y + tick[1]))
    ax.plot(x, y, 'black', lw=1., zorder=0)

    if tick_labels:

        # Add tick labels
        for xx, yy, rr in zip(x[0], y[0], r):
            ax.text(xx + offset[0], yy + offset[1], f'{rr * 100.:.0f}',
                    fontsize=fontsize, ha=ha)

def prepare_triangle_plot(ax, elems, labels=True):
    # Set the number of ticks to make
    n_ticks = 5
    tick_labels = True
    fs_vertices = [[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]]

    # Get height of triangle
    height = 3 ** 0.5 / 2

    # Get cartesian coordinates of vertices
    xs_vertices, ys_vertices = molar_fractions_to_cartesians(fs_vertices)

    # Define padding to put the vertex text neatly
    pad = [[-0.06, -0.06],
           [0.06, -0.06],
           [0.00, 0.08]]
    has = ['right', 'left', 'center']
    vas = ['top', 'top', 'bottom']

    # Make ticks and tick labels on the triangle axes
    left, right, top = np.concatenate((xs_vertices.reshape(-1, 1), ys_vertices.reshape(-1, 1)), axis=1)

    tick_size = 0.025
    bottom_ticks = 0.8264 * tick_size * (right - top)
    right_ticks = 0.8264 * tick_size * (top - left)
    left_ticks = 0.8264 * tick_size * (left - right)

    # Set axis limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, height + 0.05)

    # Plot triangle edges
    ax.plot([0., 0.5], [0., height], '-', color='black', zorder=0)
    ax.plot([0.5, 1.], [height, 0.], '-', color='black', zorder=0)
    ax.plot([0., 1.], [0., 0.], '-', color='black', zorder=0)

    # Remove spines
    for direction in ['right', 'left', 'top', 'bottom']:
        ax.spines[direction].set_visible(False)

    # Remove tick and tick labels
    ax.tick_params(which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    ax.set_aspect('equal')

    if labels:
        make_triangle_ticks(ax, right, left, bottom_ticks, n_ticks, offset=(0.03, -0.08), ha='center',
                            tick_labels=tick_labels)
        make_triangle_ticks(ax, left, top, left_ticks, n_ticks, offset=(-0.03, -0.015), ha='right', tick_labels=tick_labels)
        make_triangle_ticks(ax, top, right, right_ticks, n_ticks, offset=(0.015, 0.02), ha='left', tick_labels=tick_labels)

        # Show axis labels (i.e. atomic percentages)
        ax.text(0.5, -0.14, f'{elems[0]} content (at.%)', rotation=0., fontsize=16, ha='center', va='center')
        ax.text(0.9, 0.5, f'{elems[1]} content (at.%)', rotation=-60., fontsize=16, ha='center', va='center')
        ax.text(0.1, 0.5, f'{elems[2]} content (at.%)', rotation=60., fontsize=16, ha='center', va='center')

        # Show the chemical symbol as text at each vertex
        for idx, (x, y, (dx, dy)) in enumerate(zip(xs_vertices, ys_vertices, pad)):
            ax.text(x + dx, y + dy, s=elems[idx], fontsize=24, ha=has[idx], va=vas[idx])

    return ax

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def plot_3Dsimplex(ax, elements):               #plot tetrahedral outline
    verts=[[0,0,0],
     [1,0,0],
     [0.5,np.sqrt(3)/2,0],
     [0.5,0.28867513, 0.81649658]]
    lines=it.combinations(verts,2)
    for x in lines:
        line=np.transpose(np.array(x))
        ax.plot3D(line[0],line[1],line[2],c='0',alpha=0.5,linestyle='--')

    c3d = get_cartesian_array_from_barycentric([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    ax.text(c3d[0][0], c3d[0][1], c3d[0][2], elements[0], size=32, va='center', ha='center')
    ax.text(c3d[1][0], c3d[1][1], c3d[1][2], elements[1], size=32, va='center', ha='center')
    ax.text(c3d[2][0], c3d[2][1], c3d[2][2], elements[2], size=32, va='center', ha='center')
    ax.text(c3d[3][0], c3d[3][1], c3d[3][2], elements[3], size=32, va='center', ha='center')

def get_cartesian_array_from_barycentric(b):  #tranform from "barycentric" composition space to cartesian coordinates
    verts=[[0,0,0],
         [1,0,0],
         [0.5,np.sqrt(3)/2,0],
         [0.5,0.28867513, 0.81649658]]

    #create transformation array vis https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    t = np.transpose(np.array(verts))

    t_array=np.array([t.dot(x) for x in b]) #apply transform to all points

    return t_array

