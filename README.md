# Deep Learning on Graphs: a roadmap 

This roadmap explores the latest advances made in the field of deep learning on graphs. After listing the main papers that set the foundations of DL on graphs and Graph Neural Networks, we dig in each sub-topic. Sub-topics include graph VAE, generative model of graphs, theoretical studies of the expressiveness power of GNNs, edge-informative graphs etc.. 

I would continue adding papers to this roadmap. Feel free to suggest new papers that are missing in this list. 

---------------------------------------

# 1. Impactful Graph Neural Networks (chronological order):

**[1]** M. Defferrard, X. Bresson, and P. Vandergheynst, **Convolutional Neural Networks on Graphs with Fast
Localized Spectral Filtering,** NeurIPS, 2016 [[pdf]](https://arxiv.org/pdf/1606.09375.pdf) [[code TensorFlow]](https://github.com/mdeff/cnn_graph)

**[2]** N. Kipf and M. Welling, **Semi supervised classification with graph convolutional networks,** 2017, ICLR [[pdf]](https://arxiv.org/pdf/1609.02907.pdf)[[code TensorFlow]](https://github.com/tkipf/gcn)

**[3]** A. Santoro, D. Raposo, D. G. T. Barrett, M. Malinowski, R. Pascanu, P. Battaglia, and T. Lillicrap,
**A simple neural network module for relational reasoning,** NeurIPS, 2017 [[pdf]](https://arxiv.org/pdf/1706.01427.pdf) [[code PyTorch]](https://github.com/kimhc6028/relational-networks) [[code TensorFlow]](https://github.com/gitlimlab/Relation-Network-Tensorflow)

**[4]** J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, **Neural Message Passing for
Quantum Chemistry,** ICML, 2017 [[pdf]](https://arxiv.org/pdf/1704.01212.pdf)

**[5]** P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, **Graph Attention Networks,**
ICLR, 2018 [[pdf]](https://arxiv.org/pdf/1710.10903.pdf) [[code TensorFlow]](https://github.com/PetarV-/GAT).

# 2. Literature Reviews 

**[1]** P. W. Battaglia, J. B. Hamrick, V. Bapst, A. Sanchez-Gonzalez, V. Zambaldi, M. Malinowski, A. Tacchetti, D. Raposo, A. Santoro, R. Faulkner, C.  Gulcehre,  F.  Song,  A.  Ballard,  J.  Gilmer,  G.  Dahl,  A.  Vaswani, K.  Allen,  C.  Nash,  V.  Langston,  C.  Dyer,  N.  Heess,  D.  Wierstra, P. Kohli, M. Botvinick, O. Vinyals, Y. Li, and R. Pascanu, **Relational
inductive  biases,  deep  learning,  and  graph  networks** [[pdf]](https://arxiv.org/pdf/1806.01261.pdf) [[code]](https://github.com/deepmind/graph_nets)

**[2]** J. Zhou, G. Cui, Z. Zhang, C. Yang, Z. Liu, and M. Sun, **Graph Neural Networks: A Review of Methods and Applications** [[pdf]](https://arxiv.org/pdf/1812.08434.pdf) 

**[3]** Z. Zhang, P. Cui, and W. Zhu, **Deep Learning on Graphs: A Survey** [[pdf]](https://arxiv.org/pdf/1812.04202v1.pdf)

**[4]** W. Zonghan, P. Shirui, C. Fengwen, L. Guodong, Z. Chengqi, Y. Philip, **A Comprehensive Survey on Graph Neural Networks** [[pdf]](https://arxiv.org/pdf/1901.00596.pdf)

# 3. Let's dig by topic:

## 3.1 GNN for Edge-Informative graphs:

**[1]** M. Simonovsky and N. Komodakis. **Dynamic edge-conditioned filters in convolutional neu-
ral networks on graphs**, 2017, CVPR.[[pdf]](https://arxiv.org/pdf/1704.02901.pdf)

**[2]** M. Schlichtkrull, T. N. Kipf, P. Bloem, R. van den Berg, I. Titov, M.
Welling. **Modeling Relational Data with Graph Convolutional Networks**, 2018, In
Extended Semantic Web Conference. [[pdf]](https://arxiv.org/pdf/1703.06103.pdf)

**[3]** G. Jaume, A. Nguyen, M. Rodriguez, J-P. Thiran, M. Gabrani, **edGNN: A simple and powerful GNN for directed labeled graphs**, 2019, ICLR workshop on graphs and manifolds. [[pdf]](https://arxiv.org/pdf/1904.08745.pdf) [[code]](https://github.com/guillaumejaume/edGNN)

## 3.2 Unsupervised Graph Neural Networks

**[1]** Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm, **Deep Graph Infomax** ICLR 2019 [[pdf]](https://arxiv.org/pdf/1809.10341.pdf) [[code PyTorch]](https://github.com/PetarV-/DGI).

**[2]** Graph auto-encoder and generative graph modeling. See Section 3.6 & 3.7

## 3.3 Characterization of Graph Neural Networks

**[1]** K. Xu, W. Hu, J. Leskovec, S. Jegelka, **How Powerful are Graph Neural Networks ?,**
ICLR, 2019 [[pdf]](https://arxiv.org/pdf/1810.00826.pdf) [[code PyTorch]](https://github.com/weihua916/powerful-gnns)

**[2]** C. Morris, M. Ritzert, M. Fey , W. L. Hamilton, J. Lenssen, G. Rattan, M. Grohe, **Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks,**
AAAI, 2018 [[pdf]](https://arxiv.org/pdf/1810.02244.pdf) [[code PyTorch]](https://github.com/chrsmrrs/k-gnn)

**[3]** F. Wu, T. Zhang, A. Holanda de Souza Jr., C. Fifty, T. Yu, K. Q. Weinberger, **Simplifying Graph Convolutional Networks,**
ICML, 2019 [[pdf]](https://arxiv.org/pdf/1902.07153.pdf)

**[4]** H. NT, T. Maehara, **Revisiting Graph Neural Networks:
All We Have is Low-Pass Filters,**
submitted to NeurIPS, 2019 [[pdf]](https://arxiv.org/pdf/1905.09550.pdf)

## 3.4 Pooling on graphs:

**[1]** M. Zhang, Z. Cui, M. Neumann, Y. Chen, **An End-to-End Deep Learning Architecture for Graph Classification**, AAAI, 2018 [[pdf]](https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf)

**[2]** R. Ying, J. You, C. Morris, X. Ren, W. L. Hamilton, and J. Leskovec, **Hierarchical Graph Representation Learning with Differentiable Pooling,** 2018, NeurIPS. [[pdf]](https://arxiv.org/pdf/1806.08804.pdf) [[code]](https://github.com/RexYing/diffpool)

**[3]** Anonymous Review, **GRAPH U-NET,** to be presented at ICLR 2019 [[pdf]](https://openreview.net/pdf?id=HJePRoAct7) 

**[4]** J. Lee, I. Lee, J. Kang, **Self-Attention Graph Pooling,** ICML, 2019 [[pdf]](https://arxiv.org/pdf/1904.08082.pdf) 

## 3.5 Relational Reinforcement Learning:

**[1]** V. Zambaldi, D. Raposo, A. Santoro, V. Bapst, Y. Li, I. Babuschkin, K. Tuyls, D. Reichert, T. Lillicrap,
E. Lockhart, M. Shanahan, V. Langston, R. Pascanu, M. Botvinick, O. Vinyals, and P. Battaglia,
**Relational Deep Reinforcement Learning,** 2018 [[pdf]](https://arxiv.org/pdf/1806.01830.pdf)

## 3.6 Generative models of graphs:

**[1]** N. De Cao and T. Kipf, **MolGAN: An implicit generative model for small molecular graphs,** 2018 [[pdf]](https://arxiv.org/pdf/1805.11973.pdf)

**[2]**  J. You, B. Liu, R. Ying, V. Pande, and J. Leskovec, **Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation**, NeurIPS, 2018 [[pdf]](https://arxiv.org/pdf/1806.02473.pdf)

**[3]** M. Simonovsky, N. Komodakis **GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders**, 2018. [[pdf]](https://arxiv.org/pdf/1802.03480.pdf)

**[4]** W. Jin, R. Barzilay, T. Jaakkola, **Junction Tree Variational Autoencoder for Molecular Graph Generation**, ICML, 2018. [[pdf]](https://arxiv.org/pdf/1802.04364.pdf) [[code PyTorch]](https://github.com/wengong-jin/icml18-jtnn)

**[5]** Q. Liu, M. Allamanis, M. Brockschmidt, A. L. Gaunt **Constrained Graph Variational Autoencoders for Molecule Design**, NeuIPS, 2018. [[pdf]](https://arxiv.org/pdf/1802.03480.pdf)

**[6]** A. Grover, A. Zweig, S. Ermon, **Graphite: Iterative Generative Modeling of Graphs**, ICML, 2019. [[pdf]](https://arxiv.org/pdf/1803.10459.pdf) [[code TensorFlow]](https://github.com/ermongroup/graphite)

## 3.7 Graph Encoder-Decoder 

**[1]** T. N. Kipf and M. Welling., **Variational graph auto-encoders,** 2016, Bayesian DL, NeurIPS [[pdf]](https://arxiv.org/pdf/1611.07308.pdf)

**[2]** T. N. Kipf, E. Fetaya, K. Wang, M. Welling, R. Zemel, **Neural Relational Inference for Interacting Systems,** ICML, 2018 [[pdf]](https://arxiv.org/pdf/1611.07308.pdf)

## 3.8 Scene Understanding (e.g., Scene Graph Generation)

**[1]** D. Xu, Y. Zhu, C. B. Choy, and L. Fei-Fei. **Scene graph generation by iterative message passing**,
CVPR,2017. [[pdf]](https://arxiv.org/pdf/1701.02426.pdf)

**[2]** J. Yang, J. Lu, S. Lee, D. Batra, and D. Parikh. **Graph R-CNN for Scene Graph Generation**,
ECCV, 2018. [[pdf]](https://arxiv.org/pdf/1808.00191.pdf)

**[3]** G. Jaume, B. Bozorgtabar, H. Ekenel, J-P. Thiran, M. Gabrani, **Image-Level Attentional Context Modeling using Nest Graph Neural Networks**, NeuIPS workshop on Relational Representation Learning [[pdf]](https://arxiv.org/abs/1811.03830)

## 3.9 Knowledge Graphs

**[1]** M. Nickel, K. Murphy, V. Tresp, E. Gabrilovich, **A Review of Relational Machine Learning for Knowledge Graphs** 2015 [[pdf]](https://arxiv.org/pdf/1503.00759.pdf)

## 3.10 Neural Architecture Search with GNNs

**[1]** C. Zhang, M. Ren, R. Urtasun, **Graph Hyper Networks for Neural Architecture Search**, ICLR 2019 [[pdf]](https://openreview.net/pdf?id=rkgW0oA9FX)

## 3.xx More to come 

# 4. Libraries 

- **GraphNets** by DeepMind (written in TensorFlow/Sonnet) [[code]](https://github.com/deepmind/graph_nets)
- **Deep Graph Library (DGL)** (written in PyTorch/MXNet) [[code]](https://github.com/dmlc/dgl)
- **pytorch-geometric** (written in PyTorch) [[code]](https://github.com/rusty1s/pytorch_geometric). Provides implementation for:
  * GCN
  * GAT
  * Graph U-Net
  * Deep Graph InfoMax
  * GIN
- **Graph Embedding Methods (GEM)** (written in Keras) [[code]](https://github.com/palash1992/GEM) 
  * Node2vec
  * Laplacian Eigenmaps
- **Spektral** (written in Keras) [[code]](https://github.com/danielegrattarola/spektral/)
- **Chainer Chemistry** (written in Chainer) [[code]](https://github.com/pfnet-research/chainer-chemistry)
- **PyTorch Big-Graph** (written in PyTorch) [[code]](https://github.com/facebookresearch/PyTorch-BigGraph?fbclid=IwAR0QbWESbygic26TW1b3pxHf0Jr2XBvQTI3kVYCIs7iRP-sE1DIubpAFwTo)

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

* **Geometry and Learning from Data in 3D and Beyond** at IPAM 2019 [[link]](http://www.ipam.ucla.edu/programs/long-programs/geometry-and-learning-from-data-in-3d-and-beyond/)
  - [Geometry and Learning from Data Tutorials](https://www.ipam.ucla.edu/programs/workshops/geometry-and-learning-from-data-tutorials/)
  - [Workshop I: Geometric Processing](https://www.ipam.ucla.edu/programs/workshops/workshop-i-geometric-processing/)
  - [Workshop II: Shape Analysis](https://www.ipam.ucla.edu/programs/workshops/workshop-ii-shape-analysis/)
  - [Workshop III: Geometry of Big Data](https://www.ipam.ucla.edu/programs/workshops/workshop-iii-geometry-of-big-data/)
  - [Workshop IV: Deep Geometric Learning of Big Data and Applications](https://www.ipam.ucla.edu/programs/workshops/workshop-iv-deep-geometric-learning-of-big-data-and-applications/)

* **Geometry meets deep learning** at ICCV 2019 [[link]](http://geometricdeeplearning.com/)

* **Learning and Reasoning with Graph-Structured Data** at ICML 2019 [[link]](https://graphreason.github.io/)

* **Representation Learning on Graphs and Manifolds** at ICLR 2019 [[link]](https://rlgm.github.io/) 

* **Relational Representation Learning** at NeurIPS 2018 [[link]](https://r2learning.github.io/)

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
