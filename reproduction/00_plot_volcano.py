import matplotlib.pyplot as plt
import numpy as np
import pickle

# define theory-derived activity model
def theo_act(dist, G_opt=0.10, a=-1, b=0.5, eU=0.85):
    kb, T = 8.617e-5, 298.15
    j_ki = b*np.exp((-np.abs(dist - G_opt) + 0.86 - eU) / (kb * T))
    j = a/96**2 * np.sum(1 / (1 / (1-b) + 1 / j_ki))
    return j

# get library and ads. e. dist.
library_name = 'AgPdPtRu'
n_point = 200
with open(f'dist_libraries/{library_name}_exp.pkl', 'rb') as input:
    dists = pickle.load(input)

# def parameters
G_opt=0.1
a= -2.062e+01
b= 9.751e-01
kb = 8.617e-5
T = 298.15
bin_width = 0.025
hist_range = (-1.2, 1.0)
dist_linspace = np.linspace(hist_range[0], hist_range[1], 250)
comp = np.array([dists[n_point]['comp'][e] for e in ['Ag','Pd','Pt','Ru']]) * 100
gd = [dists[n_point][('ontop', 'OH', 'gross')],dists[n_point][('fcc', 'O', 'gross')]]
nd = [dists[n_point][('ontop', 'OH', 'net')],dists[n_point][('fcc', 'O', 'net')]]
color, label = ['steelblue','firebrick'], ['*OH','*O']

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
for j in range(2):
    ax.hist(gd[j].flatten(), bins=int((hist_range[1] - hist_range[0]) / bin_width),
        range=(hist_range[0], hist_range[1]), histtype='step', color=color[j], zorder=0)
    ax.hist(gd[j].flatten(), bins=int((hist_range[1] - hist_range[0]) / bin_width),
        range=(hist_range[0], hist_range[1]), histtype='bar', color=color[j], alpha=0.2, label=f'gross {label[j]}')

twx = ax.twinx()
j_ki = b * np.exp((-np.abs(dist_linspace - G_opt) + 0.86 - 0.85) / (kb * T))
j = -a / (1 / (1-b) + 1 / j_ki)
twx.plot(dist_linspace, j, color='black', linewidth=2.0, label=r'$j_{i}$', linestyle='--')
twx.set(ylim=(0.0, 1.5))
ax.set(xlim=(-1.1,1.0))
ax.set_xlabel(r'$\Delta\mathrm{G}_{\mathrm{ads}}-\Delta\mathrm{G}^{\mathrm{Pt}}_{\mathrm{ads}}$ [eV]', fontsize=18)
ax.set_ylabel(r'Number of sites', fontsize=18)
twx.set_ylabel(r'-j [mA/cm$^2$]', fontsize=18, rotation=270, labelpad=28)
ax_handles, _ = ax.get_legend_handles_labels()
twx_handles, _ = twx.get_legend_handles_labels()
ax_handles.append(*twx_handles)
ax.legend(handles=ax_handles,fontsize=16,loc='upper right')
s = [f'a = {a:.2f}',
        f'b = {b:.2f}',
        r'$\Delta\mathrm{G}_{\mathrm{opt}}$' + f' = {G_opt}']
s = '\n'.join(s)
ax.set_yticks([])
ax.text(0.02, 0.98, s, family='monospace', fontsize=16, transform=ax.transAxes, va='top', color='k')
ax.text(0.02, 1.02, f'   Ag({comp[0]:.0f}%) Pd({comp[1]:.0f}%) Pt({comp[2]:.0f}%) Ru({comp[3]:.0f}%)', family='monospace', fontsize=18, transform=ax.transAxes, va='bottom', color='k')
ax.tick_params(labelsize=14)
twx.tick_params(labelsize=14)
plt.tight_layout()
filename = f'misc/volcano_example.png'
fig.savefig(filename)
print(f'[SAVED] {filename}')
