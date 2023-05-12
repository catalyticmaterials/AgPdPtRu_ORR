import ase.db
import pickle
import glob
from utils.features import db_to_graphs

# Reference energies
ref_dict_335 = {'ontop_OH':ase.db.connect('../DFT_data/refs/3x3x5_pt111_ontop_OH.db').get().energy,
			'fcc_O':ase.db.connect('../DFT_data/refs/3x3x5_pt111_fcc_O.db').get().energy,
			'slab':ase.db.connect('../DFT_data/refs/3x3x5_pt111_slab.db').get().energy,
				}
ref_dict_345 = {'ontop_OH':ase.db.connect('../DFT_data/refs/3x4x5_pt111_ontop_OH.db').get().energy,
			'fcc_O':ase.db.connect('../DFT_data/refs/3x4x5_pt111_fcc_O.db').get().energy,
			'slab':ase.db.connect('../DFT_data/refs/3x4x5_pt111_slab.db').get().energy,
			 }

surface_elements = ['Ag','Ir','Pd','Pt','Ru']
adsorbate_elements = ['O','H']

paths = glob.glob(f'../DFT_data/*.db')
existing_feats = [p.split("/")[-1][:-7] for p in glob.glob(f'../features/*.graphs')]

for p in paths:
	## load joined ASE datebase
	db = ase.db.connect(p)
	
	name = p.split("/")[-1][:-3]

	#if name in existing_feats:
	#	continue

	if name.split("_")[-1] == "3x4":
		ref_dict = ref_dict_345
	else:
		ref_dict = ref_dict_335

	## Construct graphs and pickle
	print(f'Performing graph construction of {name}') 

	samples = db_to_graphs(surface_elements, adsorbate_elements, 2, 0.1, db, ref_dict)
	with open(f'{name}.graphs', 'wb') as output:
		pickle.dump(samples, output)
