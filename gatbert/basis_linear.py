# STL
from typing import Tuple
# 3rd Party
import torch

class BasisLinear(torch.nn.Module):
    def __init__(self,
                 in_features: Tuple[int],
                 out_features: int,
                 n_bases: int,
                 n_projections: Tuple[int]):
        """
        Defines a collection of N linear transformations
        that are all parameterized by a shared set of B basis matrices
        """
        super().__init__()

        self.__in_features = tuple(in_features)
        self.__out_features = out_features
        self.__n_bases = n_bases
        self.__n_projections = n_projections

        matched_dims = []
        for (in_dim, proj_dim) in zip(reversed(self.__in_features[:-1]), reversed(self.__n_projections)):
            if in_dim == proj_dim:
                matched_dims.insert(0, proj_dim)
        self.__matched_dims = matched_dims

        # TODO: Investigate whether it's a good thing that we treat this as one big transformation with a bigger fanout
        # (as far as Xavier initialization is concerned)
        self.__bases = torch.nn.Linear(self.__in_features[-1], self.__out_features * n_bases, bias=False)

        # (O, num_bases)
        coefficient_data = torch.empty(*self.__n_projections, self.__n_bases)
        torch.nn.init.xavier_normal_(coefficient_data) # FIXME: Better initializer to use for this?
        self.__coefficients = torch.nn.parameter.Parameter(coefficient_data)

    def forward(self, x: torch.Tensor):
        """

        Args:
            x:
                Tensor of shape (..., in_features)

        
        Returns:
            `torch.Tensor`:
                Tensor of dimensions (..., n_projections, out_features)
        """
        assert tuple(x.shape[-len(self.__in_features):]) == self.__in_features

        basis_output = self.__bases(x)
        # (*leading_x_dims, num_bases, out_features)
        basis_vecs = basis_output.view(*basis_output.shape[:-1], self.__n_bases, self.__out_features)

        basis_shape = (
            *(x.shape[:-len(self.__matched_dims) - 1]),

            # Broadcast over unmatched projection dims
            *(1 for _ in self.__n_projections[:len(self.__n_projections) - len(self.__matched_dims)]),
            *self.__matched_dims,
            
            self.__n_bases,
            self.__out_features
        )
        basis_vecs = basis_vecs.view(*basis_shape)

        coeff_shape = (
            *(1 for _ in x.shape[:-len(self.__matched_dims) - 1]),

            *self.__n_projections,

            self.__n_bases,
            1                # Broadcast over out_features
        )
        coeff_view = self.__coefficients.view(*coeff_shape)
        multiplied = coeff_view * basis_vecs

        
        # (*leading_dims, *n_projections, out_features)
        summed = torch.sum(multiplied, dim=-2)
        return summed
