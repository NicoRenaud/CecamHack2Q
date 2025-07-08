import numpy as np
import jax.numpy as jnp

import GPSKet
from GPSKet.hilbert.discrete_fermion import FermionicDiscreteHilbert
from GPSKet.sampler.fermionic_hopping import MetropolisHopping
from GPSKet.operator.hamiltonian.ab_initio import AbInitioHamiltonianOnTheFly
from GPSKet.nn.initializers import normal

import netket as nk

from flax import linen as nn


def solve_GPSKet(h1, h2, nelec, opt_steps=1000, n_samples=1000, M=None):
    """This is an example implementation of a "solver" for ab initio electronic
    structure Hamiltonians using GPSKet, a collection of add-on utilities for NetKet,
    see https://github.com/BoothGroup/GPSKet. This example implementation abstracts
    away quite a lot of the underlying complexity and should only be seen as a starting
    point for further experimentation.

    Args:
        h1 (np.ndarray): The one-electron tensor, shape (norb, norb)
        h2 (np.ndarray): The two-electron tensor, shape (norb, norb, norb, norb)
        nelec (int): The total number of electrons, should be even
        opt_steps (int, optional): The number of optimization steps. Defaults to 1000.
        n_samples (int, optional): The total number of MCMC samples. Defaults to 1000.
        M (int, optional): The support dimension (=network width) of the qGPS prefactor
            in the ansatz. Defaults to None in which case we use the heuristic M = norb.

    Returns:
        List[complex]: The predicted energies at each step of the optimization.
    """
    norb = h1.shape[0]

    if M is None:
        M = norb

    M = h1.shape[0]
    # Set up Hilbert space
    hi = FermionicDiscreteHilbert(norb, n_elec=(nelec // 2, nelec // 2))

    # Set up ab-initio Hamiltonian
    ha = AbInitioHamiltonianOnTheFly(hi, h1, h2)

    # Use Metropolis-Hastings sampler with hopping rule
    sa = MetropolisHopping(hi)

    # Now define a Slater-GPS (a model similar to Slater-Jastrow but with an ML model (the GPS))
    class SlaterGPS(nn.Module):
        SD: nn.module
        GPS: GPSKet.models.qGPS
        apply_fast_update: bool = True

        @nn.compact
        def __call__(self, x, cache_intermediates=False, update_sites=None):
            log_amp_sd = self.SD(
                x, cache_intermediates=cache_intermediates, update_sites=update_sites
            )
            log_amp_GPS = self.GPS(
                x, cache_intermediates=cache_intermediates, update_sites=update_sites
            )
            return log_amp_sd + log_amp_GPS

    # Choosen a very basic mean-field initialization for the Slater determinant to aid convergence
    _, vecs = np.linalg.eigh(h1)
    phi = vecs[:, : nelec // 2]

    def slater_init(_, _, dtype=jnp.complex128):
        out = jnp.array(phi).astype(dtype).reshape((1, norb, nelec // 2))
        return out

    model = SlaterGPS(
        GPSKet.models.qGPS(hi, M, init_fun=normal(sigma=1.0e-2, dtype=jnp.complex128)),
        GPSKet.models.slater.Slater(hi, init_fun=slater_init),
    )

    # Define the variational state
    vs = nk.vqs.MCState(sa, model, n_samples=n_samples)

    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=0.02)
    qgt = nk.optimizer.qgt.QGTJacobianDense(holomorphic=True)
    sr = nk.optimizer.SR(qgt=qgt, diag_shift=0.001)

    # the driver of the optimization loop
    gs = nk.VMC(ha, op, variational_state=vs, preconditioner=sr)

    energies = []
    for i in range(opt_steps):
        gs.advance()
        energies.append(gs.energy.mean)
        print("\r", "Step: {}, Energy: {} ".format(i, gs.energy), end="")

    # after the evaluation one may also evaluate other expectation values (such as the 1-RDM),
    # but we'll leave that the future (check out
    # https://github.com/BoothGroup/GPSKet/blob/master/scripts/GPS_for_ab_initio/H4x4x4.py
    # for an exemplary implementation of the 1-RDM).

    return energies
