import numpy as np

from scipy.linalg import expm

from sklearn.utils import check_random_state


def generate_covariances(n_matrices, n_channels, n_sources, rank=None,
                         distance_A_id=0., direction_A=None,
                         distance_projs=0., sigma=0.,
                         noise_A=0., f_p='log', rng=0):
    rng = check_random_state(rng)
    project = rank is not None
    # Generate A close from id
    if direction_A is None:
        direction_A = rng.randn(n_channels, n_channels)
        direction_A /= np.linalg.norm(direction_A)
    A = expm(distance_A_id * direction_A)
    # Add individual noise
    A_list = [A + noise_A * rng.randn(n_channels, n_channels)
              for _ in range(n_matrices)]
    # Generate powers
    powers = rng.rand(n_matrices, n_sources)
    # Generate source covariances
    Cs = np.zeros((n_matrices, n_channels, n_channels))
    for i in range(n_matrices):
        Cs[i, :n_sources, :n_sources] = np.diag(powers[i])
        N_i = rng.randn(n_channels - n_sources, n_channels - n_sources)
        Cs[i, n_sources:, n_sources:] = N_i.dot(N_i.T)
    # Generate covariances X:
    if noise_A != 0.:
        X = np.array([a.dot(cs).dot(a.T) for a, cs in zip(A_list, Cs)])
    else:
        X = np.array([A.dot(cs).dot(A.T) for cs in Cs])
    if project:  # Generate random projection of rank r
        W_list = []
        M = rng.randn(n_channels, n_channels)
        for _ in range(n_matrices):
            M_ = M + distance_projs * rng.randn(n_channels, n_channels)
            U, D, V = np.linalg.svd(M_)
            W_list.append(U[:, :rank].dot(D[:rank, None] * V[:rank, :]))
        X = np.array([W.dot(x).dot(W.T) for x, W in zip(X, W_list)])
    # Generate y
    alpha = rng.randn(n_sources)
    if f_p == 'log':
        y = np.log(powers).dot(alpha)
    elif f_p == 'sqrt':
        y = np.sqrt(powers).dot(alpha)
    else:
        y = powers.dot(alpha)
    y += sigma * rng.randn(n_matrices)
    return X, y
