#!/usr/bin/env python
# coding: utf-8

# In[263]:


import torch
import torch.nn as nn
import numpy as np
import itertools
from torch_scatter import scatter
from torch_geometric.data import Data as GeometricData
from torch_geometric.data import InMemoryDataset
from functools import partial
import math



class EdgeSelfAttention(nn.Module):
    """
    Simple implementation of self attention where we skip the 'value' projection and return only the attention logits (positive).
    Supports multiple heads.
    """

    def __init__(self,
                 node_embed_dim: int,
                 n_nodes: int,
                 graph_input: bool = True,
                 gat_edge: bool = True,
                 feature_name: str = "invariant_node_features",
                 bias: bool = False,
                 gat: bool = False,
                 ):

        super(EdgeSelfAttention, self).__init__()

        self.query = nn.Linear(node_embed_dim, node_embed_dim, bias=bias)
        self.key = nn.Linear(node_embed_dim, node_embed_dim, bias=bias)

        if gat or gat_edge:
            assert not (gat and gat_edge), "Can only choose gat or get edge"
            graph_input = True
            scaled_embed = int(node_embed_dim * (2 if gat else 3))
            self.attention_net = nn.Sequential(nn.Linear(scaled_embed, scaled_embed, bias=bias),
                                               nn.GELU(),
                                               nn.Linear(scaled_embed, 1, bias=bias),
                                               )
            self.query = nn.Identity()
            self.key = nn.Identity()

        else:

            self.div_term = math.sqrt(node_embed_dim)
            self.softmax = nn.Softmax(dim=-1)
            self.query = nn.Linear(node_embed_dim, node_embed_dim, bias=bias)
            self.key = nn.Linear(node_embed_dim, node_embed_dim, bias=bias)


        self.gat, self.gat_edge = gat, gat_edge
        # self.split_dim = node_embed_dim // n_heads
        # self.div_term = math.sqrt(self.split_dim)

        self.feature_name = feature_name
        self.graph_input = graph_input

        self.register_buffer("n_nodes_", torch.LongTensor([n_nodes]))
        self.register_buffer("node_embed_dim_", torch.LongTensor([node_embed_dim]))

    def matrix_attention(self, x, adj_matrix):
        query, key = (fxn(x).reshape(-1, self.n_nodes, self.node_embed_dim) for fxn in (self.query, self.key))
        attn = self.softmax(torch.matmul(query, key.transpose(-2, -1)) / self.div_term).unsqueeze(-1)
        return (adj_matrix * attn).sum(-2).reshape(-1, self.n_nodes, self.node_embed_dim)

    def graph_attention(self, batch, adj_matrix):

        # get pair list and extract features from adj_matrix
        src_node, dst_node = (i.squeeze() for i in batch.edge_index)
        index = batch.edge_index[0] // self.n_nodes
        index_src, index_dst = (i % self.n_nodes for i in (src_node, dst_node))
        edges = adj_matrix[index, index_src, index_dst]  # .reshape(len(src_node), self.n_heads, self.split_dim)

        # prune node pairs with zero writhe
        where = abs(src_node - dst_node) > 1
        src_node, dst_node = (i[where] for i in (src_node, dst_node))
        edges = edges[where]

        # project node features
        query, key = (fxn(getattr(batch, self.feature_name)) for fxn in (self.query, self.key))  # .reshape(-1, self.n_heads, self.split_dim)

        # compute attention chosen in __init__
        if self.gat:
            attn = torch.exp(self.attention_net(torch.cat([query[dst_node], key[src_node]], dim=-1))).squeeze()

        elif self.gat_edge:
            attn = torch.exp(self.attention_net(torch.cat([query[dst_node], key[src_node], edges], dim=-1))).squeeze()

        else:
            attn = torch.exp((query[dst_node] * key[src_node]).sum(-1) / self.div_term).squeeze()

        # normalization is always the same
        attn = (attn / scatter(attn, dst_node)[dst_node]).unsqueeze(-1)

        return scatter(edges * attn, dst_node, dim=0)  # .reshape(-1, self.n_nodes, self.node_embed_dim)

    @property
    def n_nodes(self):
        return self.n_nodes_.item()

    @property
    def node_embed_dim(self):
        return self.node_embed_dim_.item()

    def forward(self,
                x: "node features or batch of graphs",
                adj_matrix: "batch, n_nodes, n_node, node_embed_dim"):

        if self.graph_input:

            return self.graph_attention(x, adj_matrix)

        else:

            return self.matrix_attention(x, adj_matrix)


class _SoftUnitStep(torch.autograd.Function):
    # pylint: disable=arguments-differ

    @staticmethod
    def forward(ctx, x) -> torch.Tensor:
        ctx.save_for_backward(x)
        y = torch.zeros_like(x)
        m = x > 0.0
        y[m] = (-1 / x[m]).exp()
        return y

    @staticmethod
    def backward(ctx, dy) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        dx = torch.zeros_like(x)
        m = x > 0.0
        xm = x[m]
        dx[m] = (-1 / xm).exp() / xm.pow(2)
        return dx * dy


def soft_unit_step(x):
    r"""smooth :math:`C^\infty` version of the unit step function

    .. math::

        x \mapsto \theta(x) e^{-1/x}


    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(...)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import torch
        from e3nn.math import soft_unit_step
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        x = torch.linspace(-1.0, 10.0, 1000)
        plt.plot(x, soft_unit_step(x));
    """
    return _SoftUnitStep.apply(x)


def soft_one_hot_linspace(x: torch.Tensor, start, end, number, basis=None, cutoff=None) -> torch.Tensor:
    r"""Projection on a basis of functions

    Returns a set of :math:`\{y_i(x)\}_{i=1}^N`,

    .. math::

        y_i(x) = \frac{1}{Z} f_i(x)

    where :math:`x` is the input and :math:`f_i` is the ith basis function.
    :math:`Z` is a constant defined (if possible) such that,

    .. math::

        \langle \sum_{i=1}^N y_i(x)^2 \rangle_x \approx 1

    See the last plot below.
    Note that ``bessel`` basis cannot be normalized.

    Parameters
    ----------
    x : `torch.Tensor`
        tensor of shape :math:`(...)`

    start : float
        minimum value span by the basis

    end : float
        maximum value span by the basis

    number : int
        number of basis functions :math:`N`

    basis : {'gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel'}
        choice of basis family; note that due to the :math:`1/x` term, ``bessel`` basis does not satisfy the normalization of
        other basis choices

    cutoff : bool
        if ``cutoff=True`` then for all :math:`x` outside of the interval defined by ``(start, end)``,
        :math:`\forall i, \; f_i(x) \approx 0`

    Returns
    -------
    `torch.Tensor`
        tensor of shape :math:`(..., N)`

    Examples
    --------

    .. jupyter-execute::
        :hide-code:

        import torch
        from e3nn.math import soft_one_hot_linspace
        import matplotlib.pyplot as plt

    .. jupyter-execute::

        bases = ['gaussian', 'cosine', 'smooth_finite', 'fourier', 'bessel']
        x = torch.linspace(-1.0, 2.0, 100)

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(-1, 1.5)
        plt.tight_layout()

    .. jupyter-execute::

        fig, axss = plt.subplots(len(bases), 2, figsize=(9, 6), sharex=True, sharey=True)

        for axs, b in zip(axss, bases):
            for ax, c in zip(axs, [True, False]):
                plt.sca(ax)
                plt.plot(x, soft_one_hot_linspace(x, -0.5, 1.5, number=4, basis=b, cutoff=c).pow(2).sum(1))
                plt.plot([-0.5]*2, [-2, 2], 'k-.')
                plt.plot([1.5]*2, [-2, 2], 'k-.')
                plt.title(f"{b}" + (" with cutoff" if c else ""))

        plt.ylim(0, 2)
        plt.tight_layout()
    """
    # pylint: disable=misplaced-comparison-constant

    if cutoff not in [True, False]:
        raise ValueError("cutoff must be specified")

    if not cutoff:
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
    else:
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        step = values[1] - values[0]
        values = values[1:-1]

    diff = (x[..., None] - values) / step

    if basis == "gaussian":
        return diff.pow(2).neg().exp().div(1.12)

    if basis == "cosine":
        return torch.cos(math.pi / 2 * diff) * (diff < 1) * (-1 < diff)

    if basis == "smooth_finite":
        return 1.14136 * torch.exp(torch.tensor(2.0)) * soft_unit_step(diff + 1) * soft_unit_step(1 - diff)

    if basis == "fourier":
        x = (x[..., None] - start) / (end - start)
        if not cutoff:
            i = torch.arange(0, number, dtype=x.dtype, device=x.device)
            return torch.cos(math.pi * i * x) / math.sqrt(0.25 + number / 2)
        else:
            i = torch.arange(1, number + 1, dtype=x.dtype, device=x.device)
            return torch.sin(math.pi * i * x) / math.sqrt(0.25 + number / 2) * (0 < x) * (x < 1)

    if basis == "bessel":
        x = x[..., None] - start
        c = end - start
        bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
        out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

        if not cutoff:
            return out
        else:
            return out * ((x / c) < 1) * (0 < x)

    raise ValueError(f'basis="{basis}" is not a valid entry')


def product(x: np.ndarray, y: np.ndarray):
    return np.asarray(list(itertools.product(x, y)))


def combinations(x):
    return np.asarray(list(itertools.combinations(x, 2)))


def shifted_pairs(x: np.ndarray, shift: int, ax: int = 1):
    return np.stack([x[:-shift], x[shift:]], ax)


def get_segments(n: int = None,
                 length: int = 1,
                 index0: np.ndarray = None,
                 index1: np.ndarray = None):
    """
    Function to retrieve indices of segment pairs for various use cases.
    Returns an (n_segment_pairs, 4) array where each row (quadruplet) contains : (start1, end1, start2, end2)
    """

    if all(i is None for i in (index0, index1)):
        assert n is not None, \
            "Must provide indices (index0:array, (optionally) index1:array) or the number of points (n: int)"
        segments = combinations(shifted_pairs(np.arange(n), length)).reshape(-1, 4)
        return torch.from_numpy(segments[~(segments[:, 1] == segments[:, 2])])

    else:
        assert index0 is not None, ("If providing only one set of indices, must set the index0 argument \n"
                                    "Cannot only supply the index1 argument (doesn't make sense in this context")
        if index1 is not None:
            return torch.from_numpy(product(*[shifted_pairs(i, length) for i in (index0, index1)]).reshape(-1, 4))
        else:
            segments = combinations(shifted_pairs(index0, length)).reshape(-1, 4)
            return torch.from_numpy(segments[~(segments[:, 1] == segments[:, 2])])


##########################################   fastest ways of implementing these linear algebra ops for this purpose  (NOT trivial) ############################################

#
# def nnorm(x: torch.Tensor):
#     """Convenience function for (batched) normalization of vectors stored in arrays with last dimension 3"""
#
#     norm = torch.linalg.norm(x, axis=-1)
#
#     if x.ndim == 4:
#         return x / norm[:, :, :, None]
#     elif x.ndim == 3:
#         return x / norm[:, :, None]
#     elif x.ndim == 2:
#         return x / norm[:, None]
#     else:
#         return x / norm
#
#
# def ncross(x: torch.Tensor, y: torch.Tensor):
#     """Convenience function for (batched) cross products of vectors stored in arrays with last dimension 3"""
#
#     # c = np.array(list(map(cross,x,y)))
#     c = torch.cross(x, y, axis=-1)
#     return c
#
#
# def ndot(x, y):
#     """Convenience function for (batched) dot products of vectors stored in arrays with last dimension 3"""
#
#     # d = np.array(list(map(dot,x,y)))[:,None]
#     d = torch.sum(x * y, axis=-1)
#     return d
#
#
# def ndet(v1, v2, v3):
#     """for the triple product and finding the signed sin of the angle between v2 and v3, v1 should
#     be set equal to a vector mutually orthogonal to v2,v3"""
#     #     det = np.array(list(map(lambda x,y,z:np.linalg.det(np.array([x,y,z])),
#     #                         v1,v2,v3)))[:,None]
#     det = ndot(v1, ncross(v2, v3))
#     return det
#
#
# def writhe_segments(segment=None, xyz=None, smat=None):
#     """compute the writhe (signed crossing) of 2 segments for all frames (index 0) in xyz (xyz can contain just one frame)
#
#     THERE ARE 2 INPUT OPTIONS
#
#     **provide both of the following**
#
#     segment: numpy array of shape (Nsegments, 4) giving the indices of the 4 alpha carbons in xyz creating 2 segments:::
#              array(Nframes, [seg1_start_point,seg1_end_point,seg2_start_point,seg2_end_point]).
#              We have the flexability for segments to simple be one dimensional if only one value of writhe is to be computed.
#
#     xyz: numpy array of shape (Nframes, N_alpha_carbons, 3),coordinate array giving the positions of ALL the alpha carbons
#
#     **OR just the following**
#
#     smat ::: numpy array of shape (Nframes, Nsegments, 4, 3) : sliced coordinate matrix: coordinate array that is pre-sliced
#     with the positions of the Nsegments * 4 alpha carbons constituting the Nsegments * 2 segments to compute the writhe between """
#
#     # ensure correct shape for segment for lazy arguments
#     if smat is None:
#
#         assert all(i is not None for i in (segment, xyz)), "Must input smat or both a segment and xyz coordinates."
#
#         assert segment.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
#
#         segment = segment.unsqueeze(0) if segment.ndim < 2 else segment
#         smat = (xyz.unsqueeze(0) if xyz.ndim < 3 else xyz)[:, segment]
#
#     else:
#         assert smat is not None, "Must input smat or both segment and xyz coordinates."
#
#         assert smat.ndim == 4, "if smat is provided, it must be shape (Nframes, Nsegments, 4, 3) to prevent ambiguity"
#
#     # ensure correct shape for smat regardless of input
#     n_segments = smat.shape[1]
#     sum_dim = 2
#
#     # broadcasting trick
#     # negative sign, None placement and order are intentional, don't change without testing equivalent option
#     displacements = nnorm((-smat[:, :, :2, None, :] + smat[:, :, None, 2:, :]).reshape(-1, n_segments, 4, 3))
#
#     # array broadcasting is (surprisingly) slower than list comprehensions
#     # when using ray for the following operations (without ray, broadcasting should be faster).
#
#     crosses = nnorm(ncross(displacements[:, :, [0, 1, 3, 2]], displacements[:, :, [1, 3, 2, 0]]))
#
#     omega = torch.arcsin(ndot(crosses[:, :, [0, 1, 2, 3]], crosses[:, :, [1, 2, 3, 0]]).clip(-1, 1)).sum(sum_dim)
#
#     signs = torch.sign(ndot(ncross(nnorm(smat[:, :, 3] - smat[:, :, 2]),
#                                    nnorm(smat[:, :, 1] - smat[:, :, 0])),
#                             displacements[:, :, 0]))
#
#     wr = (1 / (2 * torch.pi)) * (omega * signs)
#
#     return wr.squeeze()


@torch.jit.script
def nnorm(x: torch.Tensor):
    """Convenience function for (batched) normalization of vectors stored in arrays with last dimension 3"""
    norm = torch.linalg.norm(x, dim=-1)

    if x.ndim == 4:
        return x / norm[:, :, :, None]
    elif x.ndim == 3:
        return x / norm[:, :, None]
    elif x.ndim == 2:
        return x / norm[:, None]
    else:
        return x / norm


@torch.jit.script
def ncross(x: torch.Tensor, y: torch.Tensor):
    """Convenience function for (batched) cross products of vectors stored in arrays with last dimension 3"""
    c = torch.cross(x, y, dim=-1)
    return c


@torch.jit.script
def ndot(x: torch.Tensor, y: torch.Tensor):
    """Convenience function for (batched) dot products of vectors stored in arrays with last dimension 3"""
    d = torch.sum(x * y, dim=-1)
    return d


@torch.jit.script
def ndet(v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor):
    """for the triple product and finding the signed sin of the angle between v2 and v3, v1 should
    be set equal to a vector mutually orthogonal to v2,v3"""
    det = ndot(v1, ncross(v2, v3))
    return det


@torch.jit.script
def writhe_segments(segment: torch.Tensor = None, xyz: torch.Tensor = None):
    """compute the writhe (signed crossing) of 2 segments for all frames (index 0) in xyz (xyz can contain just one frame)

    THERE ARE 2 INPUT OPTIONS

    **provide both of the following**

    segment: numpy array of shape (Nsegments, 4) giving the indices of the 4 alpha carbons in xyz creating 2 segments:::
             array(Nframes, [seg1_start_point,seg1_end_point,seg2_start_point,seg2_end_point]).
             We have the flexability for segments to simple be one dimensional if only one value of writhe is to be computed.

    xyz: numpy array of shape (Nframes, N_alpha_carbons, 3),coordinate array giving the positions of ALL the alpha carbons

    **OR just the following**

    smat ::: numpy array of shape (Nframes, Nsegments, 4, 3) : sliced coordinate matrix: coordinate array that is pre-sliced
    with the positions of the Nsegments * 4 alpha carbons constituting the Nsegments * 2 segments to compute the writhe between """

    # ensure correct shape for segment for lazy arguments
    assert segment is not None and xyz is not None, "Must input smat or both a segment and xyz coordinates."
    assert segment.shape[-1] == 4, "Segment indexing matrix must have last dim of 4."
    if segment.ndim < 2:
        segment = segment.unsqueeze(0)
    if xyz.ndim < 3:
        xyz = xyz.unsqueeze(0)
    smat = xyz[:, segment]


    # ensure correct shape for smat regardless of input
    n_segments = smat.shape[1]
    sum_dim = 2

    # broadcasting trick
    # negative sign, None placement and order are intentional, don't change without testing equivalent option
    displacements = nnorm((-smat[:, :, :2, None, :] + smat[:, :, None, 2:, :]).reshape(-1, n_segments, 4, 3))

    # array broadcasting is (surprisingly) slower than list comprehensions
    # when using ray for the following operations (without ray, broadcasting should be faster).

    crosses = nnorm(ncross(displacements[:, :, [0, 1, 3, 2]], displacements[:, :, [1, 3, 2, 0]]))

    omega = torch.arcsin(ndot(crosses[:, :, [0, 1, 2, 3]], crosses[:, :, [1, 2, 3, 0]]).clip(-1, 1)).sum(sum_dim)

    signs = torch.sign(ndot(ncross(nnorm(smat[:, :, 3] - smat[:, :, 2]),
                                   nnorm(smat[:, :, 1] - smat[:, :, 0])),
                            displacements[:, :, 0]))

    wr = (1 / (2 * torch.pi)) * (omega * signs)

    return wr.squeeze()


# def to_writhe_pair_list(writhe_features, n_point, length, segments):

class WritheMessage(nn.Module):
    """
    Expidited Writhe message layer.

    """

    def __init__(self,
                 n_atoms: int,
                 n_features: int,
                 batch_size: int,
                 bins: int = 250,
                 node_feature: str = "invariant_node_features",
                 residual: bool = True
                 ):
        super().__init__()
        # prerequisit information
        segments = get_segments(n_atoms)
        edges = torch.cat([segments[:, [0, 2]], segments[:, [1, 3]]]).T.long()
        expand_edges = torch.cat([i * n_atoms + edges for i in range(batch_size)], axis=1)

        self.register_buffer("edges", expand_edges.long())
        self.register_buffer("segments", segments)
        self.register_buffer("n_atoms_", torch.LongTensor([n_atoms]))
        self.register_buffer("batch_size_", torch.LongTensor([batch_size]))
        self.node_feature = node_feature
        self.n_features = n_features
        self.residual = residual

        # writhe embedding

        self.soft_one_hot = partial(soft_one_hot_linspace,
                                    start=-0.25,
                                    end=0.25,
                                    number=bins,
                                    basis="gaussian",
                                    cutoff=False)

        std = 1. / math.sqrt(n_features)

        self.register_parameter("basis",
                                torch.nn.Parameter(torch.Tensor(1, 1, bins, n_features).uniform_(-std, std),
                                                   # normal_(0, std),
                                                   requires_grad=True)
                                )

        # attention mechanism
        self.query = nn.Sequential(nn.Linear(n_features, n_features), nn.LeakyReLU())
        self.key = nn.Sequential(nn.Linear(n_features, n_features), nn.LeakyReLU())
        self.value = nn.Sequential(nn.Linear(n_features, n_features), nn.LeakyReLU())

        self.attention = nn.Sequential(nn.Linear(int(3 * n_features), 1), nn.LeakyReLU())

    @property
    def n_atoms(self):
        return self.n_atoms_.item()

    def embed_writhe(self, wr):
        return (self.soft_one_hot(wr).unsqueeze(-1) * self.basis).sum(-2)

    def compute_writhe(self, x):
        return self.embed_writhe(
            writhe_segments(self.segments, x.x.reshape(-1, self.n_atoms, 3))
        ).repeat(1, 2, 1).reshape(-1, self.n_features)

    def forward(self, x):
        features = getattr(x, self.node_feature).clone()

        src_node, dst_node = (i.flatten() for i in self.edges)

        writhe = self.compute_writhe(x)

        attention_input = torch.cat([getattr(self, i)(j) for i, j in
                                     zip(["query", "key", "value"], [features[dst_node], features[src_node], writhe])
                                     ], dim=-1)

        logits = torch.exp(self.attention(attention_input).flatten())

        attention = (logits / scatter(logits, dst_node)[dst_node]).unsqueeze(-1)

        message = scatter(writhe * attention, dst_node, dim=0)

        x[self.node_feature] = features + message if self.residual else message

        return x

class GraphDataSet(InMemoryDataset):
    def __init__(self, data_list=None, file: str = "graphs.pt"):
        super().__init__()
        if data_list is not None:
            data, slices = self.collate(data_list)
            torch.save((data, slices), file)
        self.load(file)


def dict_map(dic, keys):
    check = list(dic.keys())
    assert all(k in check for k in keys), "Not all keys exist in dict"
    return list(map(dic.__getitem__, keys))


single_letter_codes = ["G", "A", "S", "P", "V", "T", "C", "L", "I", "N",
                       "D", "Q", "K", "E", "M", "H", "F", "R", "Y", "W"]

three_letter_codes = ["GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "LEU", "ILE", "ASN",
                      "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"]

abr_to_code_ = dict(zip(three_letter_codes, single_letter_codes))

code_to_index_ = dict(zip(single_letter_codes, range(len(single_letter_codes))))

bond_codes_ = torch.zeros(20, 20).long()
bond_code_values_ = torch.arange(1, 211)
i_, j_ = torch.triu_indices(20, 20, 0)
bond_codes_[i_, j_] = bond_code_values_
i_, j_ = torch.triu_indices(20, 20, 1)
bond_codes_[j_, i_] = bond_codes_[i_, j_]


def get_codes(traj):
    return list(map(str, list(traj.top.to_fasta())[0]))


def abr_to_code(keys):
    return dict_map(abr_to_code_, keys)


def code_to_index(codes):
    if len(codes[0]) > 1:
        codes = abr_to_code(codes)
    return torch.LongTensor(dict_map(code_to_index_, codes))


def get_edges_bonds(index_sequence: int):
    index_sequence = index_sequence.flatten() if index_sequence.ndim != 1 else index_sequence
    n = len(index_sequence)
    edges = torch.triu_indices(n, n, 1).long()
    edges = torch.cat([edges, torch.flip(edges, (0,))], 1).long()

    # bonding info
    where = abs((edges * torch.LongTensor([1, -1]).reshape(2, 1)).sum(0)) == 1
    i, j = edges[:, where]
    values = bond_codes_[index_sequence[i], index_sequence[j]]
    bonds = torch.zeros(edges.shape[-1]).long()
    bonds[where] = values
    return edges, bonds


def make_dataset(traj):
    traj = traj.atom_slice(traj.top.select("name CA")).center_coordinates()

    index_sequence = code_to_index(get_codes(traj))

    edge_index, bonds = get_edges_bonds(index_sequence)

    xyz = traj.xyz

    scale = np.linalg.norm(xyz.reshape(-1, 3), axis=-1).std()

    xyz /= scale

    print(scale)

    # make data objects
    data_objs = [GeometricData(x=torch.Tensor(x),
                               atoms=index_sequence,
                               edge_index=edge_index,
                               bonds=bonds,
                               )
                 for x in xyz]

    return GraphDataSet(data_list=data_objs)