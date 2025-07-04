import pyscf

# Set up H2 system
dist = 1.8

mol = gto.Mole()

mol.build(
    atom=[("H", (x, 0.0, 0.0)) for x in dist * np.arange(2)],
    basis="sto-6g",
    symmetry=True,
    unit="Bohr",
)

nelec = mol.nelectron
print("Number of electrons: ", nelec)

myhf = scf.RHF(mol)
ehf = myhf.scf()
norb = myhf.mo_coeff.shape[1]
print("Number of molecular orbitals: ", norb)

# Get one- and two-electron integrals for canonical basis

# 1-electron 'core' hamiltonian terms, transformed into MO basis
h1 = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))

# Get 2-electron electron repulsion integrals, transformed into MO basis
eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,) * 4, compact=False)

# Previous representation exploited permutational symmetry in storage. Change this to a 4D array.
# Integrals now stored as h2[p,q,r,s] = (pq|rs) = <pr|qs>. Note 8-fold permutational symmetry.
h2 = ao2mo.restore(1, eri, norb)