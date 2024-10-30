# %% Import modules
import numpy as np
import torch_geometric
import networkx as nx
import os 
import matplotlib.pyplot as plt
from dataset import MEGGraphs

# %% Retrieve dataset
# Define name of .ds folder
'''
Note: if you want to load multiple files at once, add more filenames to the list. 
'''
filenames = ["sub-PT06ses04_ses-20181204_task-somatosensory_run-06_meg.ds"]

# Define the duration of the subepochs and the amount of overlap between them
duration = 60
overlap = 0

# Define variables for node and edge features 
conn_method = 'pli'
freqs = {'fmin': 1, 
         'fmax': 40, 
         'freqs': np.linspace(1, 40, (40 - 1) * 4 + 1)}

dataset = MEGGraphs(root="data\\", 
                    filenames=filenames, 
                    duration=duration,
                    overlap=overlap,
                    conn_method=conn_method,
                    freqs=freqs)


# %%
filename = "sub-PT06ses04_ses-20181204_task-somatosensory_run-06_meg.ds"
raw_path = os.path.join("data\\", "raw\\", filename)
raw = dataset.load_raw_data(raw_path)

# %% PSD 
data_stim = dataset.get(0)
data_non_stim = dataset.get(9)

sfreq = 256
psd_stim = data_stim.x.numpy()
psd_non_stim = data_non_stim.x.numpy()
freqs = (sfreq / 2) * (np.arange(0, psd_stim.shape[1]) / psd_stim.shape[1])

channels = raw.info["ch_names"]
channels = [ch[:3] for ch in channels]

fig = plt.figure(figsize=(12,5))
plt.subplot(121)
for idx in range(psd_stim.shape[0]):
    plt.plot(freqs, psd_stim[idx,:], label=channels[idx])

plt.figlegend(channels)
plt.xscale("log")
plt.yscale("log")
plt.title('Power spectral density (stimulation ON)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
    
plt.subplot(122)
for idx in range(psd_non_stim.shape[0]):
    plt.plot(freqs, psd_non_stim[idx,:], label=channels[idx])

plt.figlegend(channels[:2])
plt.xscale("log")
plt.yscale("log")
plt.title('Power spectral density (stimulation OFF)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')

# %% PLI
pli = data_stim.edge_attr
print(pli)

# %% Visualize graph
# Retrieve names of channels  
channels_dict = {n:channel[:3] for n, channel in enumerate(channels)}

# Retrieve positions of channels
ch_pos = [ch['loc'][:2] for ch in raw.info['chs']]
ch_pos_dict = {idx: pos for idx, pos in enumerate(ch_pos)}

# Visualize graph
g = torch_geometric.utils.to_networkx(data_stim, to_undirected=True)
nx.draw_networkx(g, pos=ch_pos_dict, with_labels=False, node_color='r', edge_color='k', node_size=450)
nx.draw_networkx_labels(g, pos=ch_pos_dict, labels=channels_dict, font_color='k', font_weight='bold', font_size=9)

plt.show()
