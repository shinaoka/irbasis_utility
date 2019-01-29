from pytriqs.archive import HDFArchive
from pytriqs.gf import *
from pytriqs.operators import *
from pytriqs.operators.util.op_struct import set_operator_structure, get_mkind
from pytriqs.operators.util.U_matrix import U_matrix
from pytriqs.operators.util.hamiltonians import h_int_slater
from pytriqs.operators.util.observables import N_op, S_op, L_op
from pytriqs.applications.impurity_solvers.pomerol2triqs import PomerolED
from pytriqs.applications.impurity_solvers.pomerol2triqs.triqs_operators import ls_op
from pytriqs.utility import mpi
from itertools import product

# 5-orbital atom with Slater interaction term
# https://github.com/j-otsuki/pomerol2triqs1.4/blob/master/example/slater.py

####################
# Input parameters #
####################

# Angular momentumd
L = 2

# Inverse temperature
beta = 10.0

# Slater parameters
U = 5.0
J = 0.1
# F0 = U
# F2 = J*(14.0/(1.0 + 0.625))
# F4 = F2*0.625

mu = U*0.5  # n=1
# mu = U*1.5  # n=2

# spin-orbit coupling
ls_coupling = 0.0

# Number of Matsubara frequencies for GF calculation
n_iw = 200

# Number of imaginary time slices for GF calculation
n_tau = 1001

# Number of bosonic Matsubara frequencies for G^2 calculations
g2_n_wb = 5
# Number of fermionic Matsubara frequencies for G^2 calculations
g2_n_wf = 10

#####################
# Input for Pomerol #
#####################

spin_names = ("up", "dn")
orb_names = range(-L, L+1)

# GF structure
off_diag = True
gf_struct = set_operator_structure(spin_names, orb_names, off_diag=off_diag)

mkind = get_mkind(off_diag, None)
# Conversion from TRIQS to Pomerol notation for operator indices
index_converter = {mkind(sn, bn) : ("atom", bi, "down" if sn == "dn" else "up")
                   for sn, (bi, bn) in product(spin_names, enumerate(orb_names))}

if mpi.is_master_node():
    print "Block structure of single-particle Green's functions:", gf_struct
    print "index_converter:", index_converter

# Operators
N = N_op(spin_names, orb_names, off_diag=off_diag)
Sz = S_op('z', spin_names, orb_names, off_diag=off_diag)
Lz = L_op('z', spin_names, orb_names, off_diag=off_diag, basis = 'spherical')
Jz = Sz + Lz

# LS coupling term
LS = ls_op(spin_names, orb_names, off_diag=off_diag, basis = 'spherical')

# Hamiltonian
# U_mat = U_matrix(L, radial_integrals = [F0,F2,F4], basis='spherical')
U_mat = U_matrix(L, U_int=U, J_hund=J, basis='spherical')
H = h_int_slater(spin_names, orb_names, U_mat, off_diag=off_diag)

H -= mu*N
H += ls_coupling * LS

# Double check that we are actually using integrals of motion
h_comm = lambda op: H*op - op*H
str_if_commute = lambda op: "= 0" if h_comm(op).is_zero() else "!= 0"

print "Check integrals of motion:"
print "[H, N]", str_if_commute(N)
print "[H, Sz]", str_if_commute(Sz)
print "[H, Lz]", str_if_commute(Lz)
print "[H, Jz]", str_if_commute(Jz)

#####################
# Pomerol ED solver #
#####################

# Make PomerolED solver object
ed = PomerolED(index_converter, verbose = True)

# Diagonalize H
if ls_coupling==0:
    ed.diagonalize(H, [N, Sz, Lz])
else:
    ed.diagonalize(H, [N, Jz])

# Do not split H into blocks (uncomment to generate reference data)
# ed.diagonalize(H, ignore_symmetries=True)

# ed.diagonalize(H)  # block diagonalize using N, Sz (default)
# ed.diagonalize(H, [N,])  # only N

# save data
ed.save_quantum_numbers("quantum_numbers.dat")
ed.save_eigenvalues("eigenvalues.dat")

# set density-matrix cutoff
ed.set_density_matrix_cutoff(1e-10)

# Compute G(i\omega)
G_iw = ed.G_iw(gf_struct, beta, n_iw)

# Compute G(\tau)
G_tau = ed.G_tau(gf_struct, beta, n_tau)

if mpi.is_master_node():
    with HDFArchive('slater_gf.out.h5', 'w') as ar:
        ar['H'] = H
        ar['G_iw'] = G_iw
        ar['G_tau'] = G_tau

###########
# G^{(2)} #
###########

common_g2_params = {'channel' : "PH",
                    'gf_struct' : gf_struct,
                    'beta' : beta,}

###############################
# G^{(2)}(i\omega;i\nu,i\nu') #
###############################

# G2_iw = ed.G2_iw_legacy( index1=('up',0), index2=('dn',0), index3=('dn',1), index4=('up',1), **common_g2_params )

# four-operators indices
four_indices = []
four_indices.append( (('up',0), ('dn',0), ('dn',1), ('up',1)) )
four_indices.append( (('up',0), ('up',0), ('dn',1), ('dn',1)) )

# compute G2 in a low-frequency box
G2_iw = ed.G2_iw_freq_box( four_indices=four_indices, n_f=g2_n_wf, n_b=g2_n_wb, **common_g2_params )

if mpi.is_master_node():
    print "G2_iw    :", type(G2_iw), "of size", len(G2_iw)
    print "G2_iw[0] :", type(G2_iw[0]), "of size", G2_iw[0].shape
    with HDFArchive('slater_gf.out.h5', 'a') as ar:
        ar['G2_iw_freq_box'] = G2_iw

# compute G2 for given freqs, (wb, wf1, wf2)
three_freqs = []
three_freqs.append( (0, 1, 2) )
three_freqs.append( (0, 1, -2) )
G2_iw = ed.G2_iw_freq_fix( four_indices=four_indices, three_freqs=three_freqs, **common_g2_params )

if mpi.is_master_node():
    print "G2_iw    :", type(G2_iw), "of size", len(G2_iw)
    print "G2_iw[0] :", type(G2_iw[0]), "of size", G2_iw[0].shape
    with HDFArchive('slater_gf.out.h5', 'a') as ar:
        ar['G2_iw_freq_fix'] = G2_iw
