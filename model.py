import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # Node embeddings
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



