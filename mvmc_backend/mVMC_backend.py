import numpy as np

def convert_to_mvmc(h1, h2, pruning_threshold=1.e-8):
  # See [Neuscamman (2013), https://doi.org/10.1063/1.4829835] for the definition of h2_mVMC
  h1_mVMC = -h1
  h2_mVMC = h2 - 0.5 * np.einsum("prrq->pq", h2)

  h1_mVMC[abs(h1_mVMC) < pruning_threshold] = 0.

  I,J = np.nonzero(h1_mVMC)

  out_1 = ""
  out_1 += "Ntransfer {}\n\n\n".format(2* len(I))
  for i in range(len(I)):
    x = I[i]
    y = J[i]
    for spin in [0,1]:
      out_1 += "{} {} {} {} {} 0.\n".format(x, spin, y, spin, h1_mVMC[x,y])

  h2_mVMC[abs(h2_mVMC) < pruning_threshold] = 0.

  I,J,K,L = np.nonzero(h2_mVMC)
  out_2 = ""
  out_2 += "Ntransfer {}\n\n\n".format(2* len(I))
  for i in range(len(I)):
    x = I[i]
    y = J[i]
    z = K[i]
    w = L[i]
    for spin in [0,1]:
      for spin_prime in [0,1]:
        out_2 += "{} {} {} {} {} {} {} {} {} 0.\n".format(x, spin, y, spin, z, spin_prime, w, spin_prime, h2_mVMC[x,y,z,w]/2)
  return out_1, out_2
