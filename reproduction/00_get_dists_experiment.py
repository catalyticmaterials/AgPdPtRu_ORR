import numpy as np
import pickle
from utils.surface import BruteForceSurface
from utils.regression import load_GCN
from tqdm import tqdm

# set adsorbate information
ads_atoms = ['O','H']  # adsorbate elements included
adsorbates = ['OH','O']  # adsorbates included
sites = ['ontop','fcc']  # sites of adsorption
coordinates = [([0,0,0],[0.65,0.65,0.40]),None]  # coordinates of multi-atom adsorbates
height = np.array([2,1.3])  # length of bond to surface
displace_e = [0.0, 0.0]
scale_e = [1, 0.5]
kwargs = {'n_conv_layers': 3,  # number of gated graph convolution layers
          'n_hidden_layers': 0,  # number of hidden layers
          'conv_dim': 18,  # number of fully connected layers
          'act': 'relu' # activation function in hidden layers.
         }

# load trained state
with open(f'regression/model_states/GC3H0reludim18BS64lr0.001.state', 'rb') as input:
    trained_state = pickle.load(input)

regressor = load_GCN(kwargs,trained_state=trained_state)

library_names = ['AgPdPtRu',
                 'AgPdPt',
                 'AgPdRu',
                 'AgPtRu',
                 'PdPtRu',
                 ]
EDX_files = ['AgPdPtRu/220829-K7-2 EDX.csv',
             'AgPdPt/0007033 EDX.csv',
             'AgPdRu/0006758 EDX.csv',
             'AgPtRu/220829-K7-1 EDX.csv',
             'PdPtRu/220826-K7-2 EDX.csv',
             ]
current_files = ['AgPdPtRu/currents_220829-K7-2_Ag-Pd-Pt-Ru_0-900mV - 500 steps_pH12.5.txt',
                 'AgPdPt/currents_0007033_Ag-Pd-Pt_0-900mV - 500 steps_pH12.5.txt',
                 'AgPdRu/currents_0006767_Ag-Pd-Ru_0-900mV - 500 steps_pH12.5.txt',
                 'AgPtRu/currents_220829-K7-1_Ag-Pt-Ru_0-900mV - 500 steps_pH12.5.txt',
                 'PdPtRu/currents_220826-K7-2_Pd-Pt-Ru_0-900mV - 500 steps_pH12.5.txt',
                ]

# iterate through material libraries
for i, library in enumerate(library_names):
    elements = np.genfromtxt(f'SDC_data/{EDX_files[i]}', delimiter=',', dtype=str)[0, 4:]
    raw_EDX = np.genfromtxt(f'SDC_data/{EDX_files[i]}', skip_header=1, delimiter=',')
    raw_currents = np.genfromtxt(f'SDC_data/{current_files[i]}')

    # iterate through measurements on library
    lib = []
    for k, row in tqdm(enumerate(raw_EDX),total=len(raw_EDX)):
        run_surface = True
        d = {}
        composition = dict(zip(elements, [0.0] * len(elements)))

        # get composition and do not run surface simulation if any component is negative
        for j, e in enumerate(elements):
            composition[e] = row[4 + j]
            if row[4 + j] < 0.0:
                run_surface = False

        # add remaining element for regressor
        for e in ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']:
            if e not in composition.keys():
                composition[e] = 0.0

        d['comp'] = composition
        d['curr'] = raw_currents[int(row[0])]

        # simulate surface and append measurement dictionary to library list
        if run_surface:
            surface_obj = BruteForceSurface(composition, adsorbates, ads_atoms, sites, coordinates, height,
                                            regressor, 'graphs', 2, 'fcc111', (96, 96), displace_e, scale_e)
            surface_obj.get_net_energies()
            d[('ontop', 'OH', 'gross')] = surface_obj.grid_dict_gross[('OH', 'ontop')].flatten()
            d[('fcc', 'O', 'gross')] = surface_obj.grid_dict_gross[('O', 'fcc')].flatten()
            d[('ontop', 'OH', 'net')] = surface_obj.grid_dict_gross[('OH', 'ontop')][surface_obj.ads_dict[('OH', 'ontop')]]
            d[('fcc', 'O', 'net')] = surface_obj.grid_dict_gross[('O', 'fcc')][surface_obj.ads_dict[('O', 'fcc')]]
            lib.append(d)
        else:
            lib.append(d)

    with open(f'dist_libraries/{library}_exp.pkl', 'wb') as output:
        pickle.dump(lib, output)