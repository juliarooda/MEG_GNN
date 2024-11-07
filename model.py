import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    '''
    Defines the graph neural network (GNN) model that is trained and tested by train.py
    '''
    def __init__(self, hidden_channels, dataset):
        '''
        Initializes the layers of the GNN.
        INPUTS:
            - hidden_channels       : integer defining the amount of features the convolutional layers should diverge/converge into
            - dataset               : Dataset of graphs
        OUTPUT: N/A
        '''
        # Retrieve the basic functionality from torch.nn.Module
        super(GNN, self).__init__()

        # Define the convolutional layers
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        # Define the final linear layer to compromise features into 2 classes (ON or OFF)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        '''
        Performs the layers initialized above. 
        INPUTS:
            - x             : Torch tensor object of node feature matrix (PSD)
            - edge_index    : Torch tensor object of edges (indices of connected nodes) 
            - edge_attr     : Torch tensor object of edge features (PLI-values)
        
        OUTPUT:
            - x             : Torch tensor object of matrix containing the model's predictions for each graph
        '''
        # Node embeddings (message passing)
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)

        # Readout layer
        x = global_mean_pool(x, batch)

        # Apply final classifier 
        x = F.dropout(x)
        x = self.lin(x)

        return x



