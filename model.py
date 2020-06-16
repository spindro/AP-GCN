__author__ = "Indro Spinelli and Simone Scardapane"
__license__ = "MIT"

import math
import torch
from typing import List
from torch.nn import ModuleList, Dropout, ReLU, Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.dropout import dropout_adj 


class AdaptivePropagation(MessagePassing):
    def __init__(self, niter: int,  h_size: int, bias = True, **kwargs):
        super(AdaptivePropagation, self).__init__(aggr='add', **kwargs)

        self.niter = niter
        self.halt = Linear(h_size,1)
        self.reg_params = list(self.halt.parameters())
        self.dropout = Dropout()
        self.reset_parameters()
    
    def reset_parameters(self):
        self.halt.reset_parameters()
        x = (self.niter+1) // 1
        b = math.log((1/x)/(1-(1/x)))
        self.halt.bias.data.fill_(b)

    def forward(self, local_preds: torch.FloatTensor, edge_index):
        sz = local_preds.size(0)
        steps = torch.ones(sz).to(local_preds.device)
        sum_h = torch.zeros(sz).to(local_preds.device)
        continue_mask = torch.ones(sz, dtype=torch.bool).to(local_preds.device)
        x = torch.zeros_like(local_preds).to(local_preds.device)

        prop = self.dropout(local_preds)
        for _ in range(self.niter):
            
            old_prop = prop
            continue_fmask = continue_mask.type('torch.FloatTensor').to(local_preds.device)
           
            drop_edge_index, _ = dropout_adj(edge_index, training=self.training)
            drop_edge_index, drop_norm = GCNConv.norm(drop_edge_index,sz)  

            prop = self.propagate(drop_edge_index, x=prop, norm=drop_norm) 

            h = torch.sigmoid(self.halt(prop)).t().squeeze()
            prob_mask = (((sum_h+h) < 0.99) & continue_mask).squeeze()
            prob_fmask = prob_mask.type('torch.FloatTensor').to(local_preds.device)

            steps = steps + prob_fmask 
            sum_h = sum_h + prob_fmask * h 

            final_iter = steps < self.niter
            
            condition = prob_mask & final_iter
            p = torch.where(condition, sum_h, 1-sum_h)
    
            to_update = self.dropout(continue_fmask)[:,None]
            x = x + (prop * p[:,None] +
                old_prop *  (1-p)[:,None])*to_update
            
            continue_mask = continue_mask & prob_mask

            if (~continue_mask).all():
                break

        x = x / steps[:,None]
        
        return x, steps, (1-sum_h)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

   

class APGCN(torch.nn.Module):
    def __init__(self,
                 dataset: InMemoryDataset,
                 niter: float = 10,
                 prop_penalty: float = 0.005,
                 hidden: List[int] = [64],
                 dropout: float = 0.5):
        super(APGCN, self).__init__()

        num_features = [dataset.data.x.shape[1]] + hidden + [dataset.num_classes]
        
        layers = []
        for in_features, out_features in zip(num_features[:-1], num_features[1:]):
            layers.append(Linear(in_features, out_features))
        
        self.prop = AdaptivePropagation(niter,dataset.num_classes)
        self.prop_penalty = prop_penalty
        self.layers = ModuleList(layers)
        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=dropout)
        self.act_fn = ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.prop.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data: Data):
        
        x, edge_index = data.x, data.edge_index
        for i, layer in enumerate(self.layers):
            x = layer(self.dropout(x))

            if i == len(self.layers) - 1:
                break

            x = self.act_fn(x)

        x, steps, reminders = self.prop(x, edge_index)

        return torch.nn.functional.log_softmax(x, dim=1), steps, reminders

