import copy as cp
import numpy as np
from scipy.linalg import pinv, eigh

from sklearn.base import TransformerMixin
from mne import EvokedArray


def shrink(cov, alpha):
    n = len(cov)
    shrink_cov = (1 - alpha) * cov + alpha * np.trace(cov) * np.eye(n) / n
    return shrink_cov


def fstd(y):
    y = y.astype(np.float32)
    y -= y.mean(axis=0)
    y /= y.std(axis=0)
    return y


def _get_scale(X, scale):
    if scale == 'auto':
        scale = 1 / np.mean([[np.trace(y) for y in x] for x in X])
    return scale


class ProjIdentitySpace(TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


class ProjLWSpace(TransformerMixin):
    def __init__(self, shrink):
        self.shrink = shrink

    def fit(self, X, y):
        return self

    def transform(self, X):
        n_sub, n_fb, p, _ = X.shape
        Xout = np.empty((n_sub, n_fb, p, p))
        for fb in range(n_fb):
            for sub in range(n_sub):
                Xout[sub, fb] = shrink(X[sub, fb], self.shrink)
        return Xout  # (sub , fb, compo, compo)


class ProjRandomSpace(TransformerMixin):
    def __init__(self, n_compo=71):
        self.n_compo = n_compo

    def fit(self, X, y):
        n_sub, n_fb, n_chan, _ = X.shape
        self.filters_ = []
        for fb in range(n_fb):
            U = np.linalg.svd(np.random.rand(n_chan, n_chan))[0][:self.n_compo]
            self.filters_.append(U)  # (fb, compo, chan) row vec
        return self

    def transform(self, X):
        n_sub, n_fb, _, _ = X.shape
        Xout = np.empty((n_sub, n_fb, self.n_compo, self.n_compo))
        for fb in range(n_fb):
            filters = self.filters_[fb]  # (compo, chan)
            for sub in range(n_sub):
                Xout[sub, fb] = filters @ X[sub, fb] @ filters.T
        return Xout  # (sub , fb, compo, compo)


class ProjCommonWassSpace(TransformerMixin):
    def __init__(self, n_compo=71):
        self.n_compo = n_compo

    def fit(self, X, y):
        n_sub, n_fb, _, _ = X.shape
        self.filters_ = []
        for fb in range(n_fb):
            covsfb = X[:, fb]
            C = mean_covs(covsfb, self.n_compo)
            eigvals, eigvecs = eigh(C)
            ix = np.argsort(np.abs(eigvals))[::-1]
            evecs = eigvecs[:, ix]
            evecs = evecs[:, :self.n_compo].T
            self.filters_.append(evecs)  # (fb, compo, chan) row vec
        return self

    def transform(self, X):
        n_sub, n_fb, _, _ = X.shape
        Xout = np.empty((n_sub, n_fb, self.n_compo, self.n_compo))
        for fb in range(n_fb):
            filters = self.filters_[fb]  # (compo, chan)
            for sub in range(n_sub):
                Xout[sub, fb] = filters @ X[sub, fb] @ filters.T
        return Xout  # (sub , fb, compo, compo)


class ProjCommonSpace(TransformerMixin):
    def __init__(self, scale=1, n_compo=71, reg=1e-7):
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, X, y):
        n_sub, n_fb, _, _ = X.shape
        self.scale_ = _get_scale(X, self.scale)
        self.filters_ = []
        self.patterns_ = []
        for fb in range(n_fb):
            covsfb = X[:, fb]
            C = covsfb.mean(axis=0)
            eigvals, eigvecs = eigh(C)
            ix = np.argsort(np.abs(eigvals))[::-1]
            evecs = eigvecs[:, ix]
            evecs = evecs[:, :self.n_compo].T
            self.filters_.append(evecs)  # (fb, compo, chan) row vec
            self.patterns_.append(pinv(evecs).T)  # (fb, compo, chan)
        return self

    def transform(self, X):
        n_sub, n_fb, _, _ = X.shape
        Xout = np.empty((n_sub, n_fb, self.n_compo, self.n_compo))
        Xs = self.scale_ * X
        for fb in range(n_fb):
            filters = self.filters_[fb]  # (compo, chan)
            for sub in range(n_sub):
                Xout[sub, fb] = filters @ Xs[sub, fb] @ filters.T
                Xout[sub, fb] += self.reg * np.eye(self.n_compo)
        return Xout  # (sub , fb, compo, compo)


class ProjSPoCSpace(TransformerMixin):
    def __init__(self, shrink=0, scale=1, n_compo=71, reg=1e-7):
        self.shrink = shrink
        self.scale = scale
        self.n_compo = n_compo
        self.reg = reg

    def fit(self, X, y):
        n_sub, n_fb, _, _ = X.shape
        target = fstd(y)
        self.scale_ = _get_scale(X, self.scale)
        self.filters_ = []
        self.patterns_ = []
        for fb in range(n_fb):
            covsfb = X[:, fb]
            C = covsfb.mean(axis=0)
            Cz = np.mean(covsfb * target[:, None, None], axis=0)
            C = shrink(C, self.shrink)
            eigvals, eigvecs = eigh(Cz, C)
            ix = np.argsort(np.abs(eigvals))[::-1]
            evecs = eigvecs[:, ix]
            evecs = evecs[:, :self.n_compo].T
            evecs /= np.linalg.norm(evecs, axis=1)[:, None]
            self.filters_.append(evecs)  # (fb, compo, chan) row vec
            self.patterns_.append(pinv(evecs).T)  # (fb, compo, chan)
        return self

    def transform(self, X):
        n_sub, n_fb, _, _ = X.shape
        Xout = np.empty((n_sub, n_fb, self.n_compo, self.n_compo))
        Xs = self.scale_ * X
        for fb in range(n_fb):
            filters = self.filters_[fb]  # (compo, chan)
            for sub in range(n_sub):
                Xout[sub, fb] = filters @ Xs[sub, fb] @ filters.T
                Xout[sub, fb] += self.reg * np.eye(self.n_compo)
        return Xout  # (sub, fb, compo, compo)

    def plot_patterns(self, info, components=None, fband=None,
                      ch_type=None, layout=None,
                      vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                      colorbar=True, scalings=None, units='a.u.', res=64,
                      size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                      show=True, show_names=False, title=None, mask=None,
                      mask_params=None, outlines='head', contours=6,
                      image_interp='bilinear', average=None, head_pos=None,
                      axes=None):

        if components is None:
            components = np.arange(self.n_compo)
        patterns = self.patterns_[fband]

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        norm_patterns = patterns / np.linalg.norm(patterns, axis=1)[:, None]
        patterns = EvokedArray(norm_patterns.T, info, tmin=0)
        return patterns.plot_topomap(
            times=components, ch_type=ch_type, layout=layout,
            vmin=vmin, vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors,
            scalings=scalings, units=units, time_unit='s',
            time_format=name_format, size=size, show_names=show_names,
            title=title, mask_params=mask_params, mask=mask, outlines=outlines,
            contours=contours, image_interp=image_interp, show=show,
            average=average, head_pos=head_pos, axes=axes)

    def plot_filters(self, info, components=None, fband=None,
                     ch_type=None, layout=None,
                     vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                     colorbar=True, scalings=None, units='a.u.', res=64,
                     size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                     show=True, show_names=False, title=None, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='bilinear', average=None, head_pos=None,
                     axes=None):

        if components is None:
            components = np.arange(self.n_compo)
        filters = self.filters_[self.fbands.index(fband)]

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        filters = EvokedArray(filters, info, tmin=0)
        return filters.plot_topomap(
            times=components, ch_type=ch_type, layout=layout, vmin=vmin,
            vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors, scalings=scalings, units=units,
            time_unit='s', time_format=name_format, size=size,
            show_names=show_names, title=title, mask_params=mask_params,
            mask=mask, outlines=outlines, contours=contours,
            image_interp=image_interp, show=show, average=average,
            head_pos=head_pos, axes=axes)


def mean_covs(covmats, rank, tol=10e-4, maxiter=50, init=None,
              sample_weight=None):
    Nt, Ne, Ne = covmats.shape
    if sample_weight is None:
        sample_weight = np.ones(Nt)
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    K = sqrtm(C, rank)
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter):
        k = k + 1
        J = np.zeros((Ne, Ne))
        for index, Ci in enumerate(covmats):
            tmp = np.dot(np.dot(K, Ci), K)
            J += sample_weight[index] * sqrtm(tmp)
        Knew = sqrtm(J, rank)
        crit = np.linalg.norm(Knew - K, ord='fro')
        K = Knew
    if k == maxiter:
        print('Max iter reach')
    C = np.dot(K, K)
    return C


def sqrtm(C, rank=None):
    if rank is None:
        rank = C.shape[0]
    d, U = np.linalg.eigh(C)
    U = U[:, -rank:]
    d = d[-rank:]
    return np.dot(U, np.sqrt(np.abs(d))[:, None] * U.T)
