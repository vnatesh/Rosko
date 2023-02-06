import os
import scipy.sparse
import dgl
from dgl.data import RedditDataset


data = RedditDataset()

adj_scipy_csr = data.graph.adjacency_matrix_scipy(fmt='csr')
adj_scipy_csr.sort_indices()
adj_scipy_csr.sum_duplicates()
assert adj_scipy_csr.has_canonical_format  # the matrix has sorted indices and no duplicates
adj_scipy_csr.data = adj_scipy_csr.data.astype('float32')


M = adj_scipy_csr.shape[0]
K = adj_scipy_csr.shape[1]
nnz = adj_scipy_csr.nnz


from array import array
output_file = open('reddit_data', 'wb')

header = array('i', [M, K, nnz])
indptr = array('i', adj_scipy_csr.indptr)
indices = array('i', adj_scipy_csr.indices)
data = array('f', adj_scipy_csr.data)

header.tofile(output_file)
indptr.tofile(output_file)
indices.tofile(output_file)
data.tofile(output_file)

output_file.close()


