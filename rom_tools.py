import numpy as np
import scipy.linalg as la


def flatten(X, np, nt, quantity=5, vector=False):
    if vector is True:
        return X.reshape((np * nt * quantity))
    else:
        return X.reshape((np * nt * quantity, -1))


def undo_flatten(X, np, nt, quantity=5, vector=False):
    if vector is True:
        return X.reshape(quantity, np, nt)
    else:
        return X.reshape(quantity, np, nt, -1)


def qDEIM(U):
    n, k = np.shape(U)
    _, _, P = la.qr(U.T, pivoting=True)
    S = np.zeros((n, k))
    for ii in range(k):
        S[P[ii], ii] = 1
    return S, P


def normalize_min_max(X_train):
    vrmin = np.min(X_train[0, :, :])
    vrmax = np.max(X_train[0, :, :])

    rhomin = np.min(X_train[1, :, :])
    rhomax = np.max(X_train[1, :, :])

    pmin = np.min(X_train[2, :, :])
    pmax = np.max(X_train[2, :, :])

    vpmin = np.min(X_train[3, :, :])
    vpmax = np.max(X_train[3, :, :])

    vtmin = np.min(X_train[4, :, :])
    vtmax = np.max(X_train[4, :, :])

    X_train[0, :, :] = (X_train[0, :, :] - vrmin) / (vrmax - vrmin)
    X_train[1, :, :] = (X_train[1, :, :] - rhomin) / (rhomax - rhomin)
    X_train[2, :, :] = (X_train[2, :, :] - pmin) / (pmax - pmin)
    X_train[3, :, :] = (X_train[3, :, :] - vpmin) / (vpmax - vpmin)
    X_train[4, :, :] = (X_train[4, :, :] - vtmin) / (vtmax - vtmin)
    return X_train
