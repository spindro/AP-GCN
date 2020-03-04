# Adaptive Propagation Graph Convolutional Network (AP-GCN)

This is the companion code for the paper:
Spinelli I, Scardapane S, Uncini A, [Adaptive Propagation Graph Convolutional Network](https://arxiv.org/abs/2002.10306), *arXiv:2002.10306*, 2020.

### The Spectral K

We introduce the adaptive propagation graph convolutional network (AP-GCN), a variation of GCN wherein each node selects automatically the number of propagation steps performed across the graph.

![Schematics of the proposed framework.](https://github.com/spindro/AP-GCN/blob/master/apgcn.png)

### Organization of the code

All the code for the models described in the paper can be found in *model.py*. An example of use which can be quickly extendet to the full experimental evaluation is provided in *AP-GCN_demo.ipynb*.

### References

[1] Kipf, T.N. and Welling, M., 2016. **Semi-supervised classification with graph convolutional networks**. arXiv preprint arXiv:1609.02907.


[2] Klicpera J., Bojchevski A., Günnemann S. **Predict then Propagate: Graph Neural Networks meet Personalized PageRank**. arXiv preprint arXiv:1810.05997


[3] Fey M., Lenssen J.E. **Fast Graph Representation Learning with PyTorch Geometric**. arXiv preprint arXiv:1903.02428


### Acknowledgments

Many thanks to the authors of [[2]](https://github.com/klicperajo/ppnp) for making their code public. Our repo is based on their work for a fair comparison between algorithms.
Many thanks also to the maintainers [[3]](https://github.com/rusty1s/pytorch_geometric) for such an awesome open-source library.


