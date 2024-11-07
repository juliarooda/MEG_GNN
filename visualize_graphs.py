import numpy as np
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt

def plot_PSD(dataset):
    '''
    Plots the power spectral density (PSD) of a graph where stimulation was turned ON and of a graph where stimulation was turned OFF. 
    INPUT:
        - dataset       : Dataset of graphs
    OUTPUT: N/A
    '''
    # Retrieve how many graphs there are in the dataset
    amount_graphs = dataset.len()

    # Retrieve a graph where stimulation was turned ON and a graph where stimulation was turned OFF. The first half of the created graphs are ON, the second half OFF. 
    data_stim = dataset.get(0)
    data_non_stim = dataset.get(amount_graphs-1)

    # Define the resampling frequency that is used in the 'process' function of 'MEGGraphs' (see dataset.py)
    sfreq = 256

    # Retrieve the node feature matrices from both graphs (this is the PSD)
    psd_stim = data_stim.x.numpy()
    psd_non_stim = data_non_stim.x.numpy()

    # Define the frequency axis 
    freqs = (sfreq / 2) * (np.arange(0, psd_stim.shape[1]) / psd_stim.shape[1])

    # Load the raw data 
    raw = dataset.load_raw_data(dataset.raw_paths[0])

    # Retrieve the channel names
    channels = raw.info["ch_names"]
    channels = [ch[:3] for ch in channels]

    # Plot the PSD of the stimulation ON graph
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    for idx in range(psd_stim.shape[0]):
        plt.plot(freqs, psd_stim[idx,:], label=channels[idx])

    plt.figlegend(channels)
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Power spectral density (stimulation ON)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

    # Plot the PSD of the stimulation OFF graph
    plt.subplot(122)
    for idx in range(psd_non_stim.shape[0]):
        plt.plot(freqs, psd_non_stim[idx,:], label=channels[idx])

    plt.figlegend(channels[:2])
    plt.xscale("log")
    plt.yscale("log")
    plt.title('Power spectral density (stimulation OFF)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')

def visualize_graph(dataset):
    '''
    Visualizes the network of one of the graphs in the dataset.
    INPUT:
        - dataset       : Dataset of graphs
    OUTPUT: N/A
    '''
    # Retrieve the first graph
    data = dataset.get(0)

    # Load the raw data
    raw = dataset.load_raw_data(dataset.raw_paths[0])

    # Retrieve the channel names
    channels = raw.info["ch_names"]
    channels_dict = {n:channel[:3] for n, channel in enumerate(channels)}

    # Retrieve the channel positions 
    ch_pos = [ch['loc'][:2] for ch in raw.info['chs']]
    ch_pos_dict = {idx: pos for idx, pos in enumerate(ch_pos)}

    # Visualize graph (note: node and edge features cannot be visualized)
    plt.figure()
    g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw_networkx(g, pos=ch_pos_dict, with_labels=False, node_color='r', edge_color='k', node_size=450)
    nx.draw_networkx_labels(g, pos=ch_pos_dict, labels=channels_dict, font_color='k', font_weight='bold', font_size=9)

    plt.show()
