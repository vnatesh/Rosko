import numpy as np
import torch
from torch import nn

init_mat = torch.tensor([[1.0, 1.0,  3.2],
                         [2.0, 1.0,  2.2],
                         [3.0, 1.0, 23.1],
                         [4.0, 1.0,  3.1],
                         [5.0, 1.0, 1.78],
                         [6.0, 1.0, 0.23],
                         [7.0, 1.0, 1.11],
                         [8.0, 1.0, 0.12],
                         [9.0, 1.0, 9.00],
                         [10.0,1.0, 1.02]])

indices  = torch.tensor([1,4,5,2,6,1,1,2,3,4,1,1,1,1,1,1,1,1,1])
offsets  = torch.tensor([0,0,2,3,5,6,7,9,13,15,18,18,18])

n = init_mat.shape[0]
m = init_mat.shape[1]

ebag = nn.EmbeddingBag(n, m, mode = "sum")
ebag.weight.data = torch.tensor(init_mat)

print(init_mat.shape)
print(indices.shape)
print(offsets.shape)

V = ebag(indices, offsets)


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




A_sp = convert_ebag_to_sp_pack(init_mat, indices, offsets)
C = torch.matmul(A_sp,init_mat)
print_mat(A_sp, len(offsets), init_mat.shape[0])


torch.equal(C, V)









# def test():
#     for ee in [0,2,10]:
#         indices_tensor, offsets_tensor, lengths_tensor = torch.load("embedding_bag/fbgemm_t856_bs65536_%d.pt" % ee)
#         lengths_tensor.shape
#         # torch.Size([856, 65536])
#         offsets_tensor.shape
#         # torch.Size([56098817])
#         indices_tensor.shape
#         # torch.Size([468084223])
#         B = 65536
#         total_L = 0
#         indices = torch.zeros(0, dtype=indices_tensor.dtype)
#         offsets = torch.zeros(1, dtype=offsets_tensor.dtype)
#         tables = [0]
#         for t in tables:
#             t_offsets = offsets_tensor[B * t : B * (t + 1) + 1]
#             total_L += t_offsets[-1] - t_offsets[0]
#             indices = torch.cat(
#                 (
#                     indices,
#                     indices_tensor[t_offsets[0] : t_offsets[-1]],
#                 )
#             )
#             offsets = torch.cat(
#                 (
#                     offsets,
#                     t_offsets[1:] - t_offsets[0] + offsets[-1],
#                 )
#             )
#         indices_tensor = indices
#         offsets_tensor = offsets
#         average_L = int(total_L / B)
#         # print(indices_tensor)
#         # print(offsets_tensor)
#         # print(average_L)
#         # print(indices_tensor.shape)
#         max_indices = int(indices_tensor.max())
#         m = 128
#         n = max_indices+1
#         emb_bag = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
#         W = np.random.uniform(
#             low=-np.sqrt(1. / n), high=np.sqrt(1. / n), size=(n, m)
#         ).astype(np.float32)
#         emb_bag.weight.data = torch.tensor(W)
#         indices_tensor = indices_tensor.int()
#         # print(indices_tensor.type())
#         # print(offsets_tensor.type())
#         V = emb_bag(indices_tensor,
#                     offsets_tensor[:-1])
#         # print(V)
#         # print(V.shape)
#         q = get_lookup_sparsity(W, indices,offsets)




# test()



indices_tensor, offsets_tensor, lengths_tensor = torch.load("embedding_bag/fbgemm_t856_bs65536_7.pt")
lengths_tensor.shape
# torch.Size([856, 65536])
offsets_tensor.shape
# torch.Size([56098817])
indices_tensor.shape
# torch.Size([468084223])

B = 65536
total_L = 0
indices = torch.zeros(0, dtype=indices_tensor.dtype)
offsets = torch.zeros(1, dtype=offsets_tensor.dtype)
tables = [0]
for t in tables:
    t_offsets = offsets_tensor[B * t : B * (t + 1) + 1]
    total_L += t_offsets[-1] - t_offsets[0]
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



indices_tensor = indices
offsets_tensor = offsets
average_L = int(total_L / B)

print(indices_tensor)
print(offsets_tensor)
print(average_L)

print(indices_tensor.shape)
max_indices = int(indices_tensor.max())

m = 128
n = max_indices+1
emb_bag = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
W = np.random.uniform(
    low=-np.sqrt(1. / n), high=np.sqrt(1. / n), size=(n, m)
).astype(np.float32)
emb_bag.weight.data = torch.tensor(W)

indices_tensor = indices_tensor.int()
print(indices_tensor.type())
print(offsets_tensor.type())
V = emb_bag(indices_tensor,
            offsets_tensor[:-1])
print(V)
print(V.shape)




def get_lookup_sparsity(ebag, indices, offsets):
    M = len(offsets)
    K = ebag.shape[0]
    # A = [0]*M*K
    nnz = 0
    nnz_inds = {}
    nnz_row = []
    #
    for m in range(len(offsets) - 1):
        s = offsets[m]
        nrows = offsets[m+1] - s
        nnz_r = 0
        for k in range(nrows):
            try:
                nnz_inds[m*K + int(indices[s+k])] += 1
            except KeyError:
                nnz += 1
                nnz_r += 1
                nnz_inds[m*K + int(indices[s+k])] = 1
            # if (m*K + int(indices[s+k])) not in nnz_inds:
            #     nnz_inds.append(m*K + int(indices[s+k]))
        nnz_row.append(nnz_r)
    print("min nnz, max nnz, nnz_tot, M, K, sparsity %")
    print(min(nnz_row),max(nnz_row), nnz, M, K, (1 - (float(nnz) / (M*K))) * 100)



q = get_lookup_sparsity(W, indices,offsets)

