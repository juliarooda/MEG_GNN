import numpy as np
import os 
import mne
import mne_connectivity
import torch
from torch_geometric.data import Data, Dataset

class MEGGraphs(Dataset):
    '''
    Creates a Dataset object of graphs out of raw MEG data. 
    '''
    def __init__(self, root, filenames, duration, overlap, conn_method, freqs):
        '''
        Initializes all the inputs given to the class. 
        INPUTS: 
            - root          : relative path where folders 'raw' and 'processed' are located
            - filenames     : list of filenames of raw data 
            - duration      : duration of final epochs in seconds
            - overlap       : overlap between final epochs in seconds
            - conn_method   : string of connectivity metric 
            - freqs         : dictionary of frequencies for connectivity calculation 

        OUTPUT: N/A
        '''
        # Initialize inputs
        self.filenames = filenames
        self.duration = duration
        self.overlap = overlap 
        self.conn_method = conn_method
        self.freqs = freqs

        # Define the total amount of epochs based on the duration, overlap, and amount of files (assuming there are 10 epochs per data file)
        if self.overlap == 0:
            self.amount_epochs = int(60 / self.duration) * 10 * len(self.filenames)
        else:
            self.amount_epochs = int((self.duration / (self.duration - self.overlap) + 1) * 10) * len(filenames)

        # Retrieve the basic functionality from torch_geometric.data.Dataset
        super().__init__(root)

    @property
    def raw_file_names(self):
        '''
        If this file already exists in raw_dir, 'def download' is skipped. Since 'def download' is passed, make sure the raw (or preprocessed) data file does exist in raw_dir.
        INPUT: N/A
        OUTPUT: N/A
        '''
        return self.filenames

    @property
    def processed_file_names(self):
        '''
        If these files already exist in processed_dir, 'def process' is skipped. 
        If you have made changes to the processing and want to test those without having to delete the already existing graph files, you can comment the return statement and return a random string instead.
        INPUT: N/A
        OUTPUT: N/A
        '''
        # Define the names of saved graphs
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
        Defines the paths where the (raw) data can be found.
        INPUT: N/A
        OUTPUT: N/A
        '''
        raw_paths = [os.path.join(self.raw_dir, filename) for filename in self.filenames]
        return raw_paths
    
    def download(self):
        '''
        Downloads data. 
        Since our data is already downloaded, this function is passed.
        INPUT: N/A
        OUTPUT: N/A
        '''
        pass

    def process(self):
        '''
        Performs all the processing steps needed to turn the raw (or preprocessed) data into graphs.
        INPUT: N/A
        OUTPUT: N/A
        '''
        
        # Iterate over each of the (raw) data files
        for idx_files, filename in enumerate(self.filenames):
            # Update current filename
            self.filename = filename  
            raw_path = self.raw_paths[idx_files]
      
            # Load data and keep relevant channels
            self.raw = self.load_raw_data(raw_path)

            # Define events stating when stimulation did or did not take place
            self.events, self.event_id = self.create_events()

            # Create 60-second epochs during which stimulation took place
            self.epochs_stim = self.create_epochs(self.raw, 
                                                self.events, 
                                                self.event_id, 
                                                label='stim')    

            # Create 60-second epochs during which no stimulation took place
            self.epochs_non_stim = self.create_epochs(self.raw, 
                                                    self.events, 
                                                    self.event_id, label='non_stim')
            

            # Split epochs into lists of subepochs with the initialized duration and overlap
            self.subepochs_stim = self.split_epochs(self.epochs_stim, self.duration, self.overlap, label='stim')
            self.subepochs_non_stim = self.split_epochs(self.epochs_non_stim, self.duration, self.overlap, label='non_stim')

            # Concatenate lists of subepochs together into an EpochsArray object
            self.subepochs = self.concatenate_subepochs(self.subepochs_stim, self.subepochs_non_stim)

            # Create a graph for each subepoch
            for idx_epochs in range(self.amount_epochs):
                # Load data of current epoch
                epoch = self.subepochs.load_data()[idx_epochs]

                # Define correct label 
                if self.subepochs.events[idx_epochs, 2] == self.event_id['stim']:
                    label = 'stim' 
                elif self.subepochs.events[idx_epochs, 2] == self.event_id['non_stim']:
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

                # Save graph with correct index
                idx_save = idx_epochs + idx_files * self.amount_epochs
                torch.save(graph, os.path.join(self.processed_dir, 
                                            f'graph_{idx_save}.pt'))


    def load_raw_data(self, raw_path):
        '''
        Loads the raw ctf data and picks the central channels from each brain region. 
        INPUT: 
            - raw_path      : path to the data file currently being processed
        
        OUTPUT:
            - raw           : RawCTF object of raw time series data of central channels
        '''
        # Load raw data file
        raw = mne.io.read_raw_ctf(raw_path, preload=False)

        # Define which channels you want to keep
        central_channels = ['MLO33-4408', 'MZO02-4408', 'MRO33-4408', 
                            'MLT34-4408', 'MLF43-4408', 'MZF02-4408', 'MRF43-4408', 'MRT34-4408', 'MRC23-4408', 'MLC23-4408', 'MLP33-4408', 'MRP33-4408', 'MZP01-4408', 'MZC03-4408']
        
        # Pick only the time series of the defined channels 
        raw.pick_channels(central_channels)
        return raw
    
    def create_events(self):
        '''
        Defines the events needed to create epochs. Events need three columns: (1) the sample of the event onset; (2) all zeros (in most cases); (3) the event_id labelling the type of event.

        Events_times represents the times at which the stimulation was switched off. When creating epochs, we therefore take the 60 seconds before this point to define stimulation ON, and the 60 seconds after to define stimulation OFF. Therefore the first item is skipped when creating 'stim events', and the last item is skipped when creating 'non stim events'. 

        INPUT: N/A
        OUTPUTS:
            - events        : array with events information needed for epoch creation
            - event_id      : dictionary with labels for stimulation ON and OFF

        '''
        # Define time points where stimulation was turned OFF (in seconds)
        events_times = [11.767, 134.765, 257.772, 380.775, 503.781, 626.785]

        # Define sampling frequency of raw data 
        sfreq = 2400

        # Turn time points into samples 
        events_samples = [int(time * sfreq) for time in events_times]

        # Define dictionary with labels for stimulation ON and OFF
        event_id = {'stim': 1, 'non_stim': 0}

        # Create array of zeroes with 10 rows (for 10 epochs) and 3 columns 
        events = np.zeros((10,3))

        # Fill first 5 rows with event samples and labels for stimulation ON
        for idx, event in enumerate(events_samples[1:]):
            events[idx] = [event, 0, event_id['stim']]

        # Fill last 5 rows with event samples and labels for stimulation OFF 
        for idx, event in enumerate(events_samples[:-1]):
            events[idx+5] = [event, 0, event_id['non_stim']]

        # Turn all values in events array into integers 
        events = events.astype(int)
        return events, event_id

    def create_epochs(self, raw, events, event_id, label):
        '''
        Creates 60-second epochs based on the events defined in 'create_events'. 
        Stimulation was turned ON in the 60 seconds prior to the event sample; stimulation was turned OFF in the 60 seconds following the event sample. 
        INPUTS:
            - raw           : RawCTF object of raw time series data of central channels
            - events        : array with events information needed for epoch creation
            - event_id      : dictionary with labels for stimulation ON and OFF
            - label         : string defining whether this epoch is for stimulation ON or OFF

        OUTPUT:
            - epochs        : Epochs object containing five 60-second stimulation ON epochs, and five stimulation OFF epochs
        '''
        # Define tmin and tmax of the epoch you want to create relative to the event sample 
        if label == 'stim': 
            tmin, tmax = -60, 0
        elif label == 'non_stim':
            tmin, tmax = 0, 60

        # Create 60-sec epochs
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
        INPUTS:
            - epochs            : Epochs object containing five 60-second stimulation ON epochs, and five stimulation OFF epochs
            - duration          : duration of final epochs in seconds
            - overlap           : overlap between final epochs in seconds
            - label             : string defining whether this epoch is for stimulation ON or OFF

        OUTPUT: 
            - subepochs_list    : list of subepochs of length 'duration'

        NB: since an epochs object cannot handle multiple epochs with the exact same event sample, the event samples of the subepochs are slightly altered using 'unique_event_samples'. 
        '''
        # Duration is not 60 seconds, so cropping is needed
        if duration != 60:
            # Since tmin and tmax are different for stimulation ON and OFF epochs, these need to be split    
            if label == 'stim':
                # First subepoch needs to start at -60 seconds 
                start = -60

                # Last subepoch needs to start at 'duration' before 0 
                stop = int(0 - duration)

                # Calculate how many subepochs you will get out of the 60 seconds based on 'duration' and 'overlap' 
                num = int(duration / (duration - overlap) + 1)

                # Create list of all start time points of subepochs
                all_tmin = np.linspace(start, stop, num)

                # Create list of all end time points of subepochs 
                all_tmax = all_tmin + duration

                # All subepochs will have the same event sample, but is not possible in an Epochs object. Therefore, create a list of unique numbers to make each event sample unique. 
                unique_event_samples = list(range(0, 2*num, 2))

            elif label == 'non_stim':
                # First subepoch needs to start at -60 seconds 
                start = 0

                # Last subepoch needs to start at 'duration' before 60
                stop = int(60 - duration)

                # Calculate how many subepochs you will get out of the 60 seconds based on 'duration' and 'overlap' 
                num = int(duration / (duration - overlap) + 1)

                # Create list of all start time points of subepochs
                all_tmin = np.linspace(start, stop, num)

                # Create list of all end time points of subepochs 
                all_tmax = all_tmin + duration

                # Create list of OTHER unique numbers than for stimulation ON
                unique_event_samples = list(range(1, 2*num, 2))
            
            # Define empty list to fill with subepochs
            subepochs_list = []

            # Iterate over all epochs
            for idx, _ in enumerate(range(epochs.__len__())):
                # Iterate over all tmin and tmax
                for i, (tmin, tmax) in enumerate(zip(all_tmin, all_tmax)):
                    # Load data from epoch
                    epoch = epochs.load_data()[idx]

                    # Crop epoch with tmin and tmax
                    subepoch = epoch.crop(tmin=tmin, tmax=tmax)

                    # Create unique event sample
                    subepoch.events[:, 0] = subepoch.events[:, 0] + unique_event_samples[i]

                    # Add subepoch to list
                    subepochs_list.append(subepoch)

        # Duration is 60 seconds, so no cropping is needed
        else:       
            # Define empty list to fill with subepochs
            subepochs_list = []
            
            # Iterate over all epochs
            for idx, _ in enumerate(range(epochs.__len__())):
                # Load data from epoch
                epoch = epochs.load_data()[idx]

                # Make sure stimulation OFF epochs have a slightly different event sample than stimulation ON epochs
                if label == 'non_stim':
                    epoch.events[:, 0] = epoch.events[:, 0] + 1 
                
                # Add subepoch to list
                subepochs_list.append(epoch)

        return subepochs_list
    
    def concatenate_subepochs(self, subepochs_list_stim, subepochs_list_non_stim):
        '''
        Concatenates all subepochs in the subepochs lists into one EpochsArray object. 
        INPUTS:
            - subepochs_list_stim       : list of subepochs of length 'duration' of stimulation ON
            - subepochs_list_non_stim   : list of subepochs of length 'duration' of stimulation OFF 

        OUTPUT:
            - combined_epochs           : EpochsArray object of all subepochs
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
        INPUTS: 
            - epoch         : the epoch currently being processed
            - sfreq         : the defined resampling frequency
        
        OUTPUT:
            - nodes         : Torch tensor object of the node feature matrix 
        '''
        # Perform PSD calculation
        fmax = 75
        psd, _ = mne.time_frequency.psd_array_welch(epoch.get_data(),
                                                    fmax=fmax, 
                                                    sfreq=sfreq)
        
        # 'Normalize' by multiplying with 10^28 
        psd_normalize = np.float64(np.multiply(psd, 10**28))

        # Turn PSD into a torch tensor to get the node feature matrix 
        nodes = torch.tensor(np.squeeze(psd_normalize), dtype=torch.float)
        return nodes
    
    def _get_edges(self, epoch, sfreq, method, freqs): 
        '''
        Calculates a connectivity metric between each of the nodes, based on the method you provide as an input. 
        Based on the non-zero indices of the resulting connectivity matrix, the edges are defined. The actual values of the resulting connectivity matrix represent the edge weights. 
        INPUTS:
            - epoch         : the epoch currently being processed
            - sfreq         : the defined resampling frequency
            - method        : string of connectivity metric 
            - freqs         : dictionary of frequencies for connectivity calculation
        
        OUTPUT:
            - edge_index    : Torch tensor object of indices of connected nodes (edges)
            - edge_weight   : Torch tensor object of PLI-values (edge features)

        ''' 

        # Perform connectivity calculation
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

        # Retrieve all non-zero elements from PLI with in each column the start and end node of the edge
        edges = np.array(np.nonzero(conn_data))

        # Convert edges to tensor
        edge_index = torch.tensor(edges, dtype=torch.long)

        # Retrieve the value of the edges from pli
        edge_weight = torch.tensor(conn_data[edges[0], edges[1]], dtype=torch.float)

        return edge_index, edge_weight
    
    def _get_labels(self, label):
        '''
        Defines the label of the graph: stimulation ON or OFF.
        INPUT:
            - label         : string defining whether this epoch is for stimulation ON or OFF
        
        OUTPUT:
            - label_tensor  : Torch tensor object of the label (0 for OFF; 1 for ON)
        '''
        label = np.asarray([self.event_id[label]])
        label_tensor = torch.tensor(label, dtype=torch.int64)
        return label_tensor

    def len(self):
        '''
        Returns the number of data objects stored in the dataset. This is equivalent to the amount of files times the amount of epochs per file.
        INPUT: N/A
        OUTPUT:
            - total     : integer representng the amount of graphs in the dataset
        '''
        total = len(self.filenames) * self.amount_epochs
        return total

    def get(self, idx):
        '''
        Gets the data object at index 'idx'
        This is equivalent to __getitem__ in pytorch
        INPUT:
            - idx       : integer defining which graph you want to retrieve
        
        OUTPUT:
            - graph     : Data object of graph number 'idx'
        '''
        graph = torch.load(os.path.join(self.processed_dir, f'graph_{idx}.pt'))
        return graph
    