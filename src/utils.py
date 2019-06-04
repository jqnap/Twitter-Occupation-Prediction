import numpy as np
import scipy.sparse as sp
import torch
import pickle as pkl
from scipy.sparse import identity

def encode_label(labels):
    """9-way job classification"""
    classes_dict = {str(i): int(i-1) for i in range(1, 10)} # class 1-9 are encoded as 0-8
    classes_dict.update({'-1': 8})

    labels = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels


def load_data(path='../Twi_data/', dataset="twi", onehot=False):
    """Load twitter follow network
    load bag-of-words features (onehot is False) or one-hot node encoding (onehot is True)
    """
    print('Loading {} dataset...'.format(dataset))
    user_label_arr = np.genfromtxt("{}{}.user_label".format(path, dataset), dtype=str)
    labels = encode_label(user_label_arr[:, 1])
    print(labels)

    if onehot is False:
        # BoW features
        bow_features = pkl.load(open("{}{}.bio_content_csr".format(path, dataset), 'rb'))
        features = sp.csr_matrix(bow_features, dtype=np.float32)

    if onehot is True:
        # onehot features
        dim = user_label_arr.shape[0]
        I = identity(dim)
        features = I

    print('features shape: ', features.shape)

    # build graph
    idx = np.array(user_label_arr[:, 0], dtype=str)
    print('total number of users: ', len(idx))
    idx_map = {j: i for i, j in enumerate(idx)}
    print('idx_map', len(idx_map))
    edges_unordered = np.genfromtxt("{}{}.follow".format(path, dataset), dtype=str)

    print("total number of edges: ", edges_unordered.shape[0])
    edge_list = list(map(idx_map.get, edges_unordered.flatten()))
    edges = np.array(edge_list, dtype=np.int32).reshape(edges_unordered.shape)
    print("edges after reindex: ", edges.shape[0])

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    features = normalize(features)

    # load train, val and test index
    ids = np.genfromtxt("{}train_val_test_id".format(path), dtype=int)
    idx_train = ids[:3645]
    idx_val = ids[3645:4101]
    idx_test = ids[4101:]
    # with open('../../data_processed/train_val_test_index.pkl', 'rb') as f:
    #     idx_train, idx_val, idx_test = pkl.load(f)


    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(np.array(idx_train).astype(np.int32))
    idx_val = torch.LongTensor(np.array(idx_val).astype(np.int32))
    idx_test = torch.LongTensor(np.array(idx_test).astype(np.int32))

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

