import numpy as np
import os 
import mne
import mne_connectivity
import torch
from torch_geometric.data import Data, Dataset

class MEGGraphs(Dataset):
    def __init__(self, root, filenames, duration, overlap, conn_method, freqs):
        self.filenames = filenames
        self.duration = duration
        self.overlap = overlap 
        if self.overlap == 0:
            self.amount_epochs = int(60 / self.duration) * 10
        else:
            self.amount_epochs = int((self.duration / (self.duration - self.overlap) + 1) * 10) 
        self.conn_method = conn_method
        self.freqs = freqs
        super().__init__(root)

    @property
    def raw_file_names(self):
        '''
        If this file already exists in raw_dir, 'def download' is skipped.
        Since 'def download' is passed, make sure the raw (or preprocessed) 
        data file does exist in raw_dir.
        '''
        return self.filenames

    @property
    def processed_file_names(self):
        '''
        If these files already exist in processed_dir, 'def process' is skipped. 
        If you have made changes to the processing and want to test those without having to delete the already existing graph files, you can 
        comment this return statement and return a random string instead.
        '''
        file_names = []
        for idx_files in range(len(self.filenames)):
            for idx_epochs in range(self.amount_epochs):
                idx_save = idx_epochs + idx_files * self.amount_epochs
                file_names.append(f'graph_{idx_save}.pt')
            
        return file_names
        # return 'random'

    @property
    def raw_paths(self):
        '''
        Defines the path where the raw (or preprocessed) data can be found.
        '''
        # raw_paths = [os.path.abspath(os.path.join(self.raw_dir, filename)) for filename in self.filenames]
        return [os.path.join(self.raw_dir, filename) for filename in self.filenames]
    
    def download(self):
        '''
        Downloads data. 
        Since our data is already downloaded, this function is passed.
        '''
        pass

    def process(self):
        '''
        Performs all the processing steps needed to turn the raw (or preprocessed) data into graphs.
        '''
        
        for idx_files, filename in enumerate(self.filenames):
            self.filename = filename  # Update current filename
            raw_path = self.raw_paths[idx_files]
      
            # Load data and keep relevant channels
            self.raw = self.load_raw_data(raw_path)

            # Define events (stim or non_stim)
            self.events, self.event_id = self.create_events()

            # Create epochs during which stimulation took place
            self.epochs_stim = self.create_epochs(self.raw, 
                                                self.events, 
                                                self.event_id, 
                                                label='stim')    

            # Create epochs during which no stimulation took place
            self.epochs_non_stim = self.create_epochs(self.raw, 
                                                    self.events, 
                                                    self.event_id, label='non_stim')
            

            # Split epochs objects into lists of subepochs
            self.subepochs_stim = self.split_epochs(self.epochs_stim, self.duration, self.overlap, label='stim')
            self.subepochs_non_stim = self.split_epochs(self.epochs_non_stim, self.duration, self.overlap, label='non_stim')

            # Concatenate lists of subepochs together to an EpochsArray object
            self.subepochs = self.concatenate_subepochs(self.subepochs_stim, self.subepochs_non_stim)
            # else: 
            #     self.subepochs = self.concatenate_epochs(self.epochs_stim, self.epochs_non_stim)


            # Create a graph for each subepoch
            for idx_epochs in range(self.amount_epochs):
                epoch = self.subepochs.load_data()[idx_epochs]
            
                if self.subepochs.events[idx_epochs,2] == self.event_id['stim']:
                    label = 'stim' 
                elif self.subepochs.events[idx_epochs,2] == self.event_id['non_stim']:
                    label = 'non_stim'

                # Resample for shorter runtime 
                resample_freq = 256
                epoch_resampled = epoch.resample(sfreq=resample_freq)

                # Get nodes with features
                nodes = self._get_nodes(epoch_resampled, resample_freq)

                # Get edges with weights
                edge_index, edge_weight = self._get_edges(epoch_resampled,      
                                                        resample_freq, 
                                                        self.conn_method,
                                                        self.freqs)

                # Define label
                y = self._get_labels(label)

                # Create graph
                graph = Data(x=nodes, edge_index=edge_index, edge_attr=edge_weight, y=y)

                # Save graph
                idx_save = idx_epochs + idx_files * self.amount_epochs
                torch.save(graph, os.path.join(self.processed_dir, 
                                            f'graph_{idx_save}.pt'))


    def load_raw_data(self, raw_path):
        '''
        Loads the raw ctf data and picks the central channels from each brain region. 
        
        Note: once the data has been preprocessed before being passed into 
        this class, this function will need to be adjusted accordingly. 
        '''
        raw = mne.io.read_raw_ctf(raw_path, preload=False)
        central_channels = ['MLO33-4408', 'MZO02-4408', 'MRO33-4408', 
                            'MLT34-4408', 'MLF43-4408', 'MZF02-4408', 'MRF43-4408', 'MRT34-4408', 'MRC23-4408', 'MLC23-4408', 'MLP33-4408', 'MRP33-4408', 'MZP01-4408', 'MZC03-4408']
        raw.pick_channels(central_channels)
        return raw
    
    def create_events(self):
        '''
        Defines the events needed to create epochs. Events need three columns: (1) the sample of the event onset; (2) all zeros (in most cases); (3) the event_id labelling the type of event.

        Events_times represents the times at which the stimulation was switched off. Therefore the first item is skipped when creating 'stim events', and the last item is skipped when creating 'non stim events'. 

        Note: the events_times are now defined here manually, but this will probably become part of the pre-processing. In that case this function will need to be adjusted accordingly.  
        '''
        events_times = [11.767, 134.765, 257.772, 380.775, 503.781, 626.785]
        sfreq = 2400
        events_samples = [int(time * sfreq) for time in events_times]
        event_id = {'stim':1, 'non_stim':0}

        events = np.zeros((10,3))
        # Stim events
        for idx, event in enumerate(events_samples[1:]):
            events[idx] = [event, 0, event_id['stim']]

        # Non stim events
        for idx, event in enumerate(events_samples[:-1]):
            events[idx+5] = [event, 0, event_id['non_stim']]

        events = events.astype(int)
        return events, event_id

    def create_epochs(self, raw, events, event_id, label):
        '''
        Creates 60 second epochs based on the events defined in 'create_events'. 
        Stimulation took place in the 60 seconds prior to the event sample; stimulation was turned off in the 60 seconds following the event sample. 
        '''
        # Define duration of epoch around event (event = start_non_stim)
        if label == 'stim': 
            tmin, tmax = -60, 0
        elif label == 'non_stim':
            tmin, tmax = 0, 60

        # Create 60 sec epochs
        epochs = mne.Epochs(raw, 
                            events, 
                            event_id=event_id[label], 
                            tmin=tmin, 
                            tmax=tmax, 
                            preload=False, 
                            baseline=(0,0))
        epochs.drop_bad()

        return epochs

    def split_epochs(self, epochs, duration, overlap, label):
        '''
        Splits each epoch into subepochs of initialized duration and overlap and returns a list of all subepochs. 
        
        Note: since an epochs object cannot handle multiple epochs with the same event sample, the event samples of the subepochs are slightly altered using 'unique_event_samples'. 
        '''
        # Create subepochs
        if overlap:    # Overlap is not zero, so cropping is needed
            if label == 'stim': 
                start = -60
                stop = int(0 - duration)
                num = int(duration / (duration - overlap) + 1)
                all_tmin = np.linspace(start, stop, num)
                all_tmax = all_tmin + duration
                unique_event_samples = list(range(0, 2*num, 2))
            elif label == 'non_stim':
                start = 0
                stop = int(60 - duration)
                num = int(duration / (duration - overlap) + 1)
                all_tmin = np.linspace(start, stop, num)
                all_tmax = all_tmin + duration
                unique_event_samples = list(range(1, 2*num, 2))
            
            subepochs_list = []
            for idx, _ in enumerate(range(epochs.__len__())):
                for i, (tmin, tmax) in enumerate(zip(all_tmin, all_tmax)):
                    epoch = epochs.load_data()[idx]
                    subepoch = epoch.crop(tmin=tmin, tmax=tmax)
                    subepoch.events[:, 0] = subepoch.events[:, 0] + unique_event_samples[i]
                    subepochs_list.append(subepoch)

        else:       # Overlap = 0, so no cropping is needed
            subepochs_list = []
            for idx, _ in enumerate(range(epochs.__len__())):
                epoch = epochs.load_data()[idx]
                if label == 'non_stim':
                    epoch.events[:, 0] = epoch.events[:, 0] + 1 # Creating unique events
                subepochs_list.append(epoch)

        return subepochs_list
    
    def concatenate_subepochs(self, subepochs_list_stim, subepochs_list_non_stim):
        '''
        Concatenates all subepochs in the subepochs lists to one EpochsArray object. 
        '''
        # Extract the data and events from the list of epochs
        all_data_stim = [epochs.get_data() for epochs in subepochs_list_stim]
        all_events_stim = [epochs.events for epochs in subepochs_list_stim]  

        all_data_non_stim = [epochs.get_data() for epochs in subepochs_list_non_stim] 
        all_events_non_stim = [epochs.events for epochs in subepochs_list_non_stim]   

        # Concatenate stim and non_stim data and events
        all_data = np.concatenate((all_data_stim, all_data_non_stim), axis=0)
        all_events = np.concatenate((all_events_stim, all_events_non_stim), axis=0)

        combined_data = np.concatenate(all_data, axis=0)
        combined_events = np.concatenate(all_events, axis=0)

        # Use the info from one of the original epochs
        info = subepochs_list_stim[0].info

        # Create the combined Epochs object
        combined_epochs = mne.EpochsArray(combined_data, 
                                          info, 
                                          events=combined_events)

        return combined_epochs

    def _get_nodes(self, epoch, sfreq):
        '''
        Calculates the Power Spectral Density (PSD) for each of the central channels. This PSD can then be used as a node feature matrix, with each central channel as a node and its PSD as the node features. 
        '''
        psd, _ = mne.time_frequency.psd_array_welch(epoch.get_data(), 
                                                    sfreq=sfreq)
        # psd_normalize = np.float64(np.multiply(psd, 10**28))
        return torch.tensor(np.squeeze(psd), dtype=torch.float)
    
    def _get_edges(self, epoch, sfreq, method, freqs): 
        '''
        Calculates a connectivity metric between each of the nodes, based on the method you provide as an input. 

        Based on the non-zero indices of the resulting connectivity matrix, the edges are defined. The actual values of the resulting connectivity matrix represent the edge weights. 
        ''' 

        conn = mne_connectivity.spectral_connectivity_time(
            epoch,
            freqs=freqs['freqs'],
            method=method,
            sfreq=sfreq,
            fmin=freqs['fmin'],
            fmax=freqs['fmax'],
            faverage=True,
        )

        # Get data as connectivity matrix
        conn_data = conn.get_data(output="dense")
        conn_data = np.squeeze(conn_data.mean(axis=-1))

        # Retrieve all non-zero elements from pli with in each column the start and end node of the edge
        edges = np.array(np.nonzero(conn_data))

        # Convert edges to tensor
        edge_index = torch.tensor(edges, dtype=torch.long)

        # Retrieve the value of the edges from pli
        edge_weight = torch.tensor(conn_data[edges[0], edges[1]], dtype=torch.float)

        return edge_index, edge_weight
    
    def _get_labels(self, label):
        label = np.asarray([self.event_id[label]])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        '''
        Returns the number of data objects stored in the dataset. This is equivalent to the amount of files times the amount of epochs per file. 
        '''
        return len(self.filenames) * self.amount_epochs 

    def get(self, idx):
        '''
        Gets the data object at index 'idx'
        This is equivalent to __getitem__ in pytorch
        '''
        # path_graph = os.path.join(self.processed_dir, f'graph_{idx}.pt')
        graph = torch.load(os.path.join(self.processed_dir, f'graph_{idx}.pt'))
        # graph_abs = torch.load(os.path.abspath(path_graph))
        return graph
    