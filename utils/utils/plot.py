from matplotlib.ticker import AutoMinorLocator
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
import itertools as it
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import imageio
from tqdm import tqdm

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

