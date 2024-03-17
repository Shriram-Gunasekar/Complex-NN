import numpy as np
import random

# Define the GrassmannNumber class
class GrassmannNumber:
    def __init__(self, scalar=0, basis={}):
        self.scalar = scalar  # Real scalar component
        self.basis = basis  # Dictionary representing basis elements and their coefficients

    def __repr__(self):
        basis_str = ' + '.join([f'{coeff}*{basis}' for basis, coeff in self.basis.items()])
        return f'{self.scalar} + ({basis_str})'

    def __add__(self, other):
        # Addition of Grassmann numbers
        scalar_sum = self.scalar + other.scalar
        basis_sum = {}
        for basis, coeff in self.basis.items():
            if basis in other.basis:
                basis_sum[basis] = coeff + other.basis[basis]
            else:
                basis_sum[basis] = coeff
        for basis, coeff in other.basis.items():
            if basis not in basis_sum:
                basis_sum[basis] = coeff
        return GrassmannNumber(scalar_sum, basis_sum)

    def __mul__(self, other):
        # Multiplication of Grassmann numbers
        scalar_prod = self.scalar * other.scalar
        basis_prod = {}
        for basis1, coeff1 in self.basis.items():
            for basis2, coeff2 in other.basis.items():
                new_basis = f'{basis1}*{basis2}'  # Concatenate basis elements
                new_coeff = coeff1 * coeff2
                if new_basis in basis_prod:
                    basis_prod[new_basis] += new_coeff
                else:
                    basis_prod[new_basis] = new_coeff
        return GrassmannNumber(scalar_prod, basis_prod)

    def scalar_mul(self, scalar):
        # Scalar multiplication of a Grassmann number
        return GrassmannNumber(self.scalar * scalar, {basis: coeff * scalar for basis, coeff in self.basis.items()})


