import torch
import torch.nn as nn
import numpy as np
import time 
import matplotlib.pyplot as plt
import sys



def print_mat(A, M, K):
    a = ''
    for i in range(M):
        for j in range(K):
            a += str(float(A[i][j])) + ' '
        a += '\n'
    print(a)



def convert_ebag_to_sp_pack(ebag, indices, offsets):
    M = len(offsets)
    K = ebag.shape[0]
    A = [0]*M*K
    #
    for m in range(len(offsets) - 1):
        s = offsets[m]
        nrows = offsets[m+1] - s
        for k in range(nrows):
            A[m*K + indices[s+k]] += 1
    #
    A[(M-1)*K + indices[offsets[-1]]] = 1
    return torch.reshape(torch.tensor(A).float(),(M,K))




def nnz_bins(ebag, indices, offsets):
    M = len(offsets)
    K = ebag.shape[0]
    bins = {}
    #
    for m in range(len(offsets) - 1):
        s = offsets[m]
        nrows = offsets[m+1] - s
        tmp = set()
        tmp1 = []
        for k in range(nrows):
            tmp.add(indices[s+k].item())
            tmp1.append(indices[s+k].item())
        try:
            bins[len(tmp)].append(tmp1)
        except KeyError:
            bins[len(tmp)] = [tmp1]
    # bins[1].append([indices[offsets[-1]].item()])
    return bins
    # A[(M-1)*K + indices[offsets[-1]]] = 1
    # return torch.reshape(torch.tensor(A).float(),(M,K))



def get_reordered_ind_off(bins, offsets):
    M = len(offsets)
    inds = []
    start = 0
    while M > 1:
        for i in sorted(bins.keys()):
            if len(bins[i]):
                inds.append(bins[i].pop())
                M -= 1
        for i in list(reversed(sorted(bins.keys()))):
            if len(bins[i]):
                inds.append(bins[i].pop())
                M -= 1
    return inds



def gen_reordered_offsets(inds, indices, offsets):
    off = [0]
    start = 0
    x = 0
    for i in inds:
        off.append(off[x] + len(i))
        x += 1
        start += len(i)
    inds = [item for sublist in inds for item in sublist]
    inds.append(indices[offsets[-1]-1].item())
    return (torch.tensor(inds),torch.tensor(off))

    # [item for sublist in inds for item in sublist]




def reorder_test(reorder, ncores,inp):

    # os.environ['OMP_NUM_THREADS'] = "%d" % (1) 

    # torch.Size([468084223])
    # tables = range(1)
    # tables = range(5)

# for z in range(1,20):
    tables = [inp]
# tables = range(lengths_tensor.shape[0])
        # for t in tables:
    indices_tensor, offsets_tensor, lengths_tensor = torch.load("embedding_bag/fbgemm_t856_bs65536_7.pt")
    lengths_tensor.shape
    # torch.Size([856, 65536])
    offsets_tensor.shape
    # torch.Size([56098817])
    indices_tensor.shape

    B = 65536
    total_L = 0
    indices = torch.zeros(0, dtype=indices_tensor.dtype)
    offsets = torch.zeros(1, dtype=offsets_tensor.dtype)

    t = inp
    t_offsets = offsets_tensor[B * t : B * (t + 1) + 1]
    try:
        total_L += t_offsets[-1] - t_offsets[0]
    except IndexError:
        return
    indices = torch.cat(
        (
            indices,
            indices_tensor[t_offsets[0] : t_offsets[-1]],
        )
    )
    offsets = torch.cat(
        (
            offsets,
            t_offsets[1:] - t_offsets[0] + offsets[-1],
        )
    )


    print(indices_tensor)
    indices_tensor = indices
    offsets_tensor = offsets
    average_L = int(total_L / B)

    # print(indices_tensor)
    # print(offsets_tensor)
    # print(average_L)

    # print(indices_tensor.shape)
    try:
        max_indices = int(indices_tensor.max())
    except RuntimeError:
        return


    m = 128
    n = max_indices+1
    emb_bag = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
    W = np.random.uniform(
        low=-np.sqrt(1. / n), high=np.sqrt(1. / n), size=(n, m)
    ).astype(np.float32)
    emb_bag.weight.data = torch.tensor(W)

    indices_tensor = indices_tensor.int()
    # print(indices_tensor.type())
    # print(offsets_tensor.type())

    f = open("results", 'a')
    # if reorder:
    #     bins = nnz_bins(W, indices_tensor,offsets_tensor)
    #     inds = get_reordered_ind_off(bins, offsets_tensor)
    #     indices_tensor1, offsets_tensor1 = gen_reordered_offsets(inds, indices_tensor,offsets_tensor[:-1])
    #     start = time.time_ns()
    #     V = emb_bag(indices_tensor1,
    #                 offsets_tensor1[:-1])
    #     dur = time.time_ns() - start
    #     f.write("reorder,%d,%d,%d\n" % (t,ncores,dur))

    # else:
    start = time.time_ns()
    V = emb_bag(indices_tensor,
            offsets_tensor[:-1])
    dur = time.time_ns() - start
    f.write("non-reorder,%d,%d,%d\n" % (t,ncores,dur))

    # f.close()
# print(V)
# print(V.shape)




if __name__ == '__main__':

    for i in [ 6,  7,  9, 12, 27, 29, 34, 46, 51, 53, 59, 69, 71, 72, 74, 86, 89]:
        reorder_test(0, int(sys.argv[1]),i)
        reorder_test(1, int(sys.argv[1]),i) 