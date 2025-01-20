from openfermion.ops import BosonOperator, FermionOperator
from openfermion.utils.indexing import down_index, up_index
from openfermion.hamiltonians.special_operators import number_operator
from numpy import cos, pi
from itertools import product

def K_fermi_hubbard(
    x_dim,
    y_dim,
    tunneling,
    nnn_tunneling,
    coulomb,
    chemical_potential=0.0,
    magnetic_field=0.0,
    periodic=True,
    particle_hole_symmetry=False,
):
    # Initialize operator.
    n_sites = x_dim * y_dim
    n_spin_orbitals = 2 * n_sites
    hubbard_model = FermionOperator()
    
    def decompose(k):
        return k % x_dim, k // x_dim
    
    # Loop through sites and add terms.
    for k in range(n_sites):
        
        # Define kx, ky.
        nx, ny = decompose(k)
        kx = 2 * pi * nx / x_dim
        ky = 2 * pi * ny / y_dim
        
        # Add hopping terms.
        if x_dim == 1 or y_dim == 1:
            epsilon = -2 * tunneling * (cos(kx) + cos(ky) - 1) - 2 * nnn_tunneling * cos(2 * kx) * cos(2 * ky)
        else:
            epsilon = -2 * tunneling * (cos(kx) + cos(ky)) - 4 * nnn_tunneling * cos(kx) * cos(ky)
        hubbard_model += number_operator(n_spin_orbitals, up_index(k), epsilon)
        hubbard_model += number_operator(n_spin_orbitals, down_index(k), epsilon)
        
    for k1, k2, q in product(range(n_sites), repeat=3):
        
        n1x, n1y = decompose(k1)
        n2x, n2y = decompose(k2)
        nqx, nqy = decompose(q)

        k1_plus_q = ((n1x + nqx) % x_dim, (n1y + nqy) % y_dim)
        k2_minus_q = ((n2x - nqx) % x_dim, (n2y - nqy) % y_dim)
        
        k1_plus_q = k1_plus_q[0] + k1_plus_q[1] * x_dim
        k2_minus_q = k2_minus_q[0] + k2_minus_q[1] * x_dim
        
        hubbard_model += FermionOperator(((up_index(k1_plus_q), 1), (down_index(k2_minus_q), 1), (down_index(k2), 0), (up_index(k1), 0)), coefficient=coulomb/n_sites)

    return hubbard_model
        
def fermi_hubbard_nnn(
    x_dimension,
    y_dimension,
    tunneling,
    nnn_tunneling,
    coulomb,
    chemical_potential=0.0,
    magnetic_field=0.0,
    periodic=True,
    particle_hole_symmetry=False,
):
    # Initialize operator.
    n_sites = x_dimension * y_dimension
    n_spin_orbitals = 2 * n_sites
    hubbard_model = FermionOperator()

    # Loop through sites and add terms.
    for site in range(n_sites):
        # Get indices of right and bottom neighbors
        right_neighbor = _right_neighbor(site, x_dimension, y_dimension, periodic)
        bottom_neighbor = _bottom_neighbor(site, x_dimension, y_dimension, periodic)
        top_neighbor = _top_neighbor(site, x_dimension, y_dimension, periodic)

        # Add hopping terms with neighbors to the right and bottom.
        if right_neighbor is not None:
            hubbard_model += _hopping_term(up_index(site), up_index(right_neighbor), -tunneling)
            hubbard_model += _hopping_term(down_index(site), down_index(right_neighbor), -tunneling)
        if bottom_neighbor is not None:
            hubbard_model += _hopping_term(up_index(site), up_index(bottom_neighbor), -tunneling)
            hubbard_model += _hopping_term(down_index(site), down_index(bottom_neighbor), -tunneling)

        # Add local pair Coulomb interaction terms.
        hubbard_model += _coulomb_interaction_term(
            n_spin_orbitals, up_index(site), down_index(site), coulomb, particle_hole_symmetry
        )

        # Add chemical potential and magnetic field terms.
        hubbard_model += number_operator(
            n_spin_orbitals, up_index(site), -chemical_potential - magnetic_field
        )
        hubbard_model += number_operator(
            n_spin_orbitals, down_index(site), -chemical_potential + magnetic_field
        )
        
        # next nearest neighbor hopping
        if y_dimension == 1 and x_dimension > 1:
            right_nnn = _right_neighbor(right_neighbor, x_dimension, y_dimension, periodic)
            
            if right_nnn is not None:
                hubbard_model += _hopping_term(up_index(site), up_index(right_nnn), -nnn_tunneling)
                hubbard_model += _hopping_term(down_index(site), down_index(right_nnn), -nnn_tunneling)
                
        if x_dimension == 1 and y_dimension > 1:
            bottom_nnn = _bottom_neighbor(bottom_neighbor, x_dimension, y_dimension, periodic)
            
            if bottom_nnn is not None:
                hubbard_model += _hopping_term(up_index(site), up_index(bottom_nnn), -nnn_tunneling)
                hubbard_model += _hopping_term(down_index(site), down_index(bottom_nnn), -nnn_tunneling)
            
        if x_dimension > 1 and y_dimension > 1:
            top_right_nnn = _right_neighbor(top_neighbor, x_dimension, y_dimension, periodic)
            bottom_right_nnn = _right_neighbor(bottom_neighbor, x_dimension, y_dimension, periodic)
            
            if top_right_nnn is not None:
                hubbard_model += _hopping_term(up_index(site), up_index(top_right_nnn), -nnn_tunneling)
                hubbard_model += _hopping_term(down_index(site), down_index(top_right_nnn), -nnn_tunneling)
            if bottom_right_nnn is not None:
                hubbard_model += _hopping_term(up_index(site), up_index(bottom_right_nnn), -nnn_tunneling)
                hubbard_model += _hopping_term(down_index(site), down_index(bottom_right_nnn), -nnn_tunneling)

    return hubbard_model

def _hopping_term(i, j, coefficient, bosonic=False):
    op_class = BosonOperator if bosonic else FermionOperator
    hopping_term = op_class(((i, 1), (j, 0)), coefficient)
    hopping_term += op_class(((j, 1), (i, 0)), coefficient.conjugate())
    return hopping_term

def _coulomb_interaction_term(n_sites, i, j, coefficient, particle_hole_symmetry, bosonic=False):
    op_class = BosonOperator if bosonic else FermionOperator
    number_operator_i = number_operator(n_sites, i, parity=2 * bosonic - 1)
    number_operator_j = number_operator(n_sites, j, parity=2 * bosonic - 1)
    if particle_hole_symmetry:
        number_operator_i -= op_class((), 0.5)
        number_operator_j -= op_class((), 0.5)
    return coefficient * number_operator_i * number_operator_j

def _K_coulomb_interaction_term(i1, i2, j1, j2, coefficient, bosonic=False):
    op_class = BosonOperator if bosonic else FermionOperator
    hopping_term = op_class(((i1, 1), (i2, 0)), 1)
    hopping_term *= op_class(((j1, 1), (j2, 0)), 1)
    return coefficient * hopping_term

def _right_neighbor(site, x_dimension, y_dimension, periodic):
    if site is None:
        return None
    if x_dimension == 1:
        return None
    if (site + 1) % x_dimension == 0:
        if periodic:
            return site + 1 - x_dimension
        else:
            return None
    return site + 1


def _bottom_neighbor(site, x_dimension, y_dimension, periodic):
    if site is None:
        return None
    if y_dimension == 1:
        return None
    if site + x_dimension + 1 > x_dimension * y_dimension:
        if periodic:
            return site + x_dimension - x_dimension * y_dimension
        else:
            return None
    return site + x_dimension


def _top_neighbor(site, x_dimension, y_dimension, periodic):
    if site is None:
        return None
    if y_dimension == 1:
        return None
    if site < x_dimension:
        if periodic:
            return site - x_dimension + x_dimension * y_dimension
        else:
            return None
    return site - x_dimension