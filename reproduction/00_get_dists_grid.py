import numpy as np
import pickle
from utils.plot import get_molar_fractions
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

alloy = 'AgPdPtRu'
elements = ['Ag', 'Ir', 'Pd', 'Pt', 'Ru']

# Iterate through a 5% composition grid and predict adsorbtion energy distributions
frac_list = get_molar_fractions(0.05, 4)
dict_list = []

for frac in tqdm(frac_list,total=len(frac_list)):

    composition = dict(zip(elements,[0.0]*len(elements)))
    alloy_elem = [alloy[i:i+2] for i in range(0,len(alloy))[::2]]

    for j, e in enumerate(alloy_elem):
        composition[e] = frac[j]

    surface_obj = BruteForceSurface(composition, adsorbates, ads_atoms, sites, coordinates, height,
                                    regressor, 'graphs', 2, 'fcc111', (96, 96), displace_e, scale_e)
    surface_obj.get_net_energies()

    d = {'comp':composition}
    d[('ontop', 'OH', 'gross')] = surface_obj.grid_dict_gross[('OH', 'ontop')].flatten()
    d[('fcc', 'O', 'gross')] = surface_obj.grid_dict_gross[('O', 'fcc')].flatten()
    d[('ontop', 'OH', 'net')] = surface_obj.grid_dict_gross[('OH', 'ontop')][surface_obj.ads_dict[('OH', 'ontop')]]
    d[('fcc', 'O', 'net')] = surface_obj.grid_dict_gross[('O', 'fcc')][surface_obj.ads_dict[('O', 'fcc')]]

    dict_list.append(d)

with open(f'dist_libraries/{alloy}_equi.pkl', 'wb') as output:
    pickle.dump(dict_list, output)


# Iterate through a 1% composition grid of the ternary alloys and predict adsorbtion energy distributions
frac_list = get_molar_fractions(0.01, 4)
mask = np.all(frac_list != 0.0, axis=1)
frac_list = frac_list[~mask]

dict_list = []

for frac in tqdm(frac_list,total=len(frac_list)):

    composition = dict(zip(elements,[0.0]*len(elements)))
    alloy_elem = [alloy[i:i+2] for i in range(0,len(alloy))[::2]]

    for j, e in enumerate(alloy_elem):
        composition[e] = frac[j]

    surface_obj = BruteForceSurface(composition, adsorbates, ads_atoms, sites, coordinates, height,
                                    regressor, 'graphs', 2, 'fcc111', (96, 96), displace_e, scale_e)
    surface_obj.get_net_energies()

    d = {'comp':composition}
    d[('ontop', 'OH', 'gross')] = surface_obj.grid_dict_gross[('OH', 'ontop')].flatten()
    d[('fcc', 'O', 'gross')] = surface_obj.grid_dict_gross[('O', 'fcc')].flatten()
    d[('ontop', 'OH', 'net')] = surface_obj.grid_dict_gross[('OH', 'ontop')][surface_obj.ads_dict[('OH', 'ontop')]]
    d[('fcc', 'O', 'net')] = surface_obj.grid_dict_gross[('O', 'fcc')][surface_obj.ads_dict[('O', 'fcc')]]

    dict_list.append(d)

with open(f'dist_libraries/{alloy}_equi_ternaries.pkl', 'wb') as output:
    pickle.dump(dict_list, output)

