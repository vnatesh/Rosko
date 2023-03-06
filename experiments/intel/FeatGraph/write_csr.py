import os
import scipy.sparse
import dgl
from array import array
from dgl.data import RedditDataset
from ogb.nodeproppred import * 


# write csr in binary format M,K,nnz,indptr,indices,data
def write_csr(csr, fname):
	print("writing %s" % fname)
	csr.sort_indices()
	csr.sum_duplicates()
	assert csr.has_canonical_format  # the matrix has sorted indices and no duplicates
	csr.data = csr.data.astype('float32')
	M = csr.shape[0]
	K = csr.shape[1]
	nnz = csr.nnz
	output_file = open(fname, 'wb')
	header = array('i', [M, K, nnz])
	indptr = array('i', csr.indptr)
	indices = array('i', csr.indices)
	data = array('f', csr.data)
	header.tofile(output_file)
	indptr.tofile(output_file)
	indices.tofile(output_file)
	data.tofile(output_file)
	output_file.close()



data = RedditDataset()
adj_scipy_csr = data._graph.adjacency_matrix(scipy_fmt = 'csr')
write_csr(adj_scipy_csr, 'reddit_data')


dataset = DglNodePropPredDataset(name='ogbn-proteins')
adj_scipy_csr = dataset.graph[0].adjacency_matrix(scipy_fmt = 'csr')
write_csr(adj_scipy_csr, 'ogbn-proteins')




# input_file = open('file', 'rb')
# header1 = array('Q')
# indptr1 = array('i')
# indices1 = array('i')
# data1 = array('f')

# header1.fromstring(input_file.read(3*8))
# indptr1.fromstring(input_file.read(4*(M+1)))
# indices1.fromstring(input_file.read(nnz*4))
# data1.fromstring(input_file.read(nnz*4))
# input_file.close()


# comparison = header == header1
# equal_arrays = comparison.all()
 
# print(equal_arrays)


# f = open('reddit_data.txt', 'w')
# f.write("%d %d %d\n" % (M, K, nnz))


# ptrlen = M+1
# indlen = nnz
# datalen = nnz

# rem = ptrlen % 1000;
# ptrlen -= rem;

# for i in range(0, ptrlen, 1000):
# 	line = ' '.join(map(str, adj_scipy_csr.indptr[i : i+1000]))
# 	f.write(line + ' ')

# line = ' '.join(map(str, adj_scipy_csr.indptr[ptrlen : ptrlen + rem]))
# f.write(line + ' ')



# rem = indlen % 1000;
# indlen -= rem;

# for i in range(0, indlen, 1000):
# 	line = ' '.join(map(str, adj_scipy_csr.indices[i : i+1000]))
# 	f.write(line + ' ')

# line = ' '.join(map(str, adj_scipy_csr.indices[indlen : indlen + rem]))
# f.write(line + ' ')



# rem = datalen % 1000;
# datalen -= rem;

# for i in range(0, datalen, 1000):
# 	line = ' '.join(map(str, adj_scipy_csr.data[i : i+1000]))
# 	f.write(line + ' ')

# line = ' '.join(map(str, adj_scipy_csr.data[datalen : datalen + rem]))
# f.write(line + ' ')



# adj_scipy_csr.nnz   
# 114615892
# adj_scipy_csr.shape 
# (232965, 232965)
