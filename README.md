# Deep Learning on Graphs: a roadmap 

This roadmap explores the following topics:

- Graph Convolutional Networks
- Graph Neural Networks
- Relation Networks

I would continue adding papers to this roadmap.

---------------------------------------

# 1. Impactful Graph Neural Networks (chronological order):

**[1]** M. Defferrard, X. Bresson, and P. Vandergheynst, **Convolutional Neural Networks on Graphs with Fast
Localized Spectral Filtering,** NIPS, 2016 [[pdf]](https://arxiv.org/pdf/1606.09375.pdf) [[code TensorFlow]](https://github.com/mdeff/cnn_graph)

**[2]** N. Kipf and M. Welling, **Semi supervised classification with graph convolutional networks,** 2017, ICLR [[pdf]](https://arxiv.org/pdf/1609.02907.pdf)[[code TensorFlow]](https://github.com/tkipf/gcn)

**[3]** A. Santoro, D. Raposo, D. G. T. Barrett, M. Malinowski, R. Pascanu, P. Battaglia, and T. Lillicrap,
**A simple neural network module for relational reasoning,** NIPS, 2017 [[pdf]](https://arxiv.org/pdf/1706.01427.pdf) [[code PyTorch]](https://github.com/kimhc6028/relational-networks) [[code TensorFlow]](https://github.com/gitlimlab/Relation-Network-Tensorflow)

**[4]** J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, **Neural Message Passing for
Quantum Chemistry,** ICML, 2017 [[pdf]](https://arxiv.org/pdf/1704.01212.pdf)

**[5]** P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, **Graph Attention Networks,**
ICLR, 2018 [[pdf]](https://arxiv.org/pdf/1710.10903.pdf) [[code TensorFlow]](https://github.com/PetarV-/GAT).

**[6]** K. Xu, W. Hu, J. Leskovec, S. Jegelka, **How Powerful are Graph Neural Networks ?,**
ICLR, 2019 [[pdf]](https://arxiv.org/pdf/1810.00826.pdf) [[code PyTorch]](https://github.com/weihua916/powerful-gnns)

**[7]** Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm, **Deep Graph Infomax** ICLR 2019 [[pdf]](https://arxiv.org/pdf/1809.10341.pdf) [[code PyTorch]](https://github.com/PetarV-/DGI).

# 2. Literature Reviews 

**[1]** P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, C.  Gulcehre,  F.  Song,  A.  Ballard,  J.  Gilmer,  G.  Dahl,  A.  Vaswani, K.  Allen,  C.  Nash,  V.  Langston,  C.  Dyer,  N.  Heess,  D.  Wierstra, P. Kohli, M. Botvinick, O. Vinyals, Y. Li, and R. Pascanu, **Relational
inductive  biases,  deep  learning,  and  graph  networks** [[pdf]](https://arxiv.org/pdf/1806.01261.pdf) [[code]](https://github.com/deepmind/graph_nets)

**[2]** J. Zhou, G. Cui, Z. Zhang, C. Yang, Z. Liu, and M. Sun, **Graph Neural Networks: A Review of Methods and Applications** [[pdf]](https://arxiv.org/pdf/1812.08434.pdf) 

**[3]** Z. Zhang, P. Cui, and W. Zhu, **Deep Learning on Graphs: A Survey** [[pdf]](https://arxiv.org/pdf/1812.04202v1.pdf)

**[4]** W. Zonghan, P. Shirui, C. Fengwen, L. Guodong, Z. Chengqi, Y. Philip, **A Comprehensive Survey on Graph Neural Networks** [[pdf]](https://arxiv.org/pdf/1901.00596.pdf)

# 3. Let's dig by topic:

## 3.1 Pooling on graphs:

**[1]** R. Ying, J. You, C. Morris, X. Ren, W. L. Hamilton, and J. Leskovec, **Hierarchical Graph Represen-
tation Learning with Differentiable Pooling,** 2018, NIPS. [[pdf]](https://arxiv.org/pdf/1806.08804.pdf) [[code]](https://github.com/RexYing/diffpool)

**[2]** Anonymous Review, **GRAPH U-NET,** to be presented at ICLR 2019 [[pdf]](https://openreview.net/pdf?id=HJePRoAct7) 

**[3]** M. Zhang, Z. Cui, M. Neumann, Y. Chen, **An End-to-End Deep Learning Architecture for Graph Classification**, AAAI, 2018 [[pdf]](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf)

## 3.2 Relational Reinforcement Learning:

**[1]** V. Zambaldi, D. Raposo, A. Santoro, V. Bapst, Y. Li, I. Babuschkin, K. Tuyls, D. Reichert, T. Lillicrap,
E. Lockhart, M. Shanahan, V. Langston, R. Pascanu, M. Botvinick, O. Vinyals, and P. Battaglia,
**Relational Deep Reinforcement Learning,** 2018 [[pdf]](https://arxiv.org/pdf/1806.01830.pdf)

## 3.3 Generative models of graphs:

**[1]** N. De Cao and T. Kipf, **MolGAN: An implicit generative model for small molecular graphs,** 2018 [[pdf]](https://arxiv.org/pdf/1805.11973.pdf)

**[2]**  J. You, B. Liu, R. Ying, V. Pande, and J. Leskovec, **Graph Convolutional Policy Network for Goal-
Directed Molecular Graph Generation,**, NIPS, 2018 [[pdf]](https://arxiv.org/pdf/1806.02473.pdf)

## 3.4 Knowledge Graphs

**[1]** M. Nickel, K. Murphy, V. Tresp, E. Gabrilovich, **A Review of Relational Machine Learning for Knowledge Graphs** 2015 [[pdf]](https://arxiv.org/pdf/1503.00759.pdf)

## 3.5 More to come 

# 4. Libraries 

- **GraphNets** by DeepMind (written in TensorFlow/Sonnet) [[code]](https://github.com/deepmind/graph_nets)
- **Deep Graph Library (DGL)** (written in PyTorch/MXNet) [[code]](https://github.com/dmlc/dgl)
- **pytorch-geometric** (written in PyTorch) [[code]](https://github.com/rusty1s/pytorch_geometric). Provides implementation for:
  * GCN
  * GAT
  * Graph U-Net
  * Deep Graph InfoMax
  * GIN
- **Graph Embedding Methods (GEM)** (written in Kera) [[code]](https://github.com/palash1992/GEM) 
  * Node2vec
  * Laplacian Eigenmaps
- **Spektral** (written in Keras) [[code]](https://github.com/danielegrattarola/spektral/)
- **Chainer Chemistry** (written in Chainer) [[code]](https://github.com/pfnet-research/chainer-chemistry)


# 5. Presentation/Slides 

* Weekly presentation on **Graph Neural Networks at MILA** [[presentation]](https://github.com/shagunsodhani/Graph-Reading-Group)
* Presentation on **Relational Learning** by Petar Veličković [[presentation]](https://www.cl.cam.ac.uk/~pv273/slides/MILA-RN.pdf)
* Presentation on **Adversarial learning meets graphs** by Petar Veličković [[presentation]](https://www.cl.cam.ac.uk/~pv273/slides/MILA-AdvGraph.pdf)
* Presentation on **How Powerful are Graph Neural Networks?** by Jure 
Leskovec [[presentation]](http://i.stanford.edu/~jure/pub/talks2/graphsage_gin-ita-feb19.pdf?fbclid=IwAR1rM7xiVmHpk6PJ1dMntYPz1odn1TQiOGYOZKpaLuuBjVb34LxwupioABw)
  - Graph SAGE
  - Hierarchical Graph Pooling
  - Graph Isomorphism Networks


# 6. Videos 

- **Geometric Deep Learning** by Siraj Raval [[video]](https://www.youtube.com/watch?v=D3fnGG7cdjY)
- **Graph neural networks: Variations and applications** by Alexander Gaunt [[video]](https://www.youtube.com/watch?v=cWIeTMklzNg)
- **Large-scale Graph Representation Learning** by Jure Leskovec [[video]](https://www.youtube.com/watch?v=oQL4E1gK3VU)

# 7. Workshops

* **Representation Learning on Graphs and Manifolds** at ICLR 2019 [[link]](https://rlgm.github.io/) 

* **Relational Representation Learning** at NeurIPS 2018 [[link]](https://r2learning.github.io/)
  - My paper: Image-Level Attentional Context Modeling using Nest Graph Neural Networks [[link]](https://arxiv.org/abs/1811.03830)

* **Representation Learning on Networks** by Jure Leskovec [[link]](http://snap.stanford.edu/proj/embeddings-www/)
  - Network/Relational Representation Learning
  - Node embeddings
  - Graph Neural Networks
  - Applications in recommender systems and computational biology 
  
* **Deep Learning for Network Biology** by Jure Leskovec [[link]](http://snap.stanford.edu/deepnetbio-ismb/)
  - Network propagation and node embeddings
  - Graph autoencoders and deep representation learning
  - Heterogeneous networks
  - Tensorflow examples
