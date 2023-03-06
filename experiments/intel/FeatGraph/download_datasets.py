import os
import scipy.sparse
import dgl
from dgl.data import RedditDataset
from ogb.nodeproppred import * 

if not os.path.isdir('data'):
    os.mkdir('data')

data = RedditDataset()

adj_scipy_csr = data._graph.adjacency_matrix(scipy_fmt = 'csr')
adj_scipy_csr.sort_indices()
adj_scipy_csr.sum_duplicates()
assert adj_scipy_csr.has_canonical_format  # the matrix has sorted indices and no duplicates
adj_scipy_csr.data = adj_scipy_csr.data.astype('float32')
scipy.sparse.save_npz('data/reddit_csr_float32.npz', adj_scipy_csr)

adj_scipy_coo = adj_scipy_csr.tocoo()
assert adj_scipy_csr.has_canonical_format  # the matrix has sorted indices and no duplicates
scipy.sparse.save_npz('data/reddit_coo_float32.npz', adj_scipy_coo)



dataset = DglNodePropPredDataset(name='ogbn-proteins')
adj_scipy_csr = dataset.graph[0].adjacency_matrix(scipy_fmt = 'csr')
adj_scipy_csr.sort_indices()
adj_scipy_csr.sum_duplicates()
assert adj_scipy_csr.has_canonical_format  # the matrix has sorted indices and no duplicates
adj_scipy_csr.data = adj_scipy_csr.data.astype('float32')
scipy.sparse.save_npz('data/ogbn-proteins_csr.npz', adj_scipy_csr)
