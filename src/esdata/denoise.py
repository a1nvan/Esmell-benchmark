import pywt
import math
from typing import Optional, Union, List, Tuple

import numpy as np
from scipy import interpolate


# ---------- вспомогательные ----------
def _estimate_sigma(detail_coeffs: np.ndarray) -> float:
    return np.median(np.abs(detail_coeffs)) / 0.6745


def _universal_threshold(sigma: float, n: int) -> float:
    return sigma * math.sqrt(2 * math.log(n))


def wavelet_denoise_1d(
    x: np.ndarray,
    wavelet: str = "db4",
    mode: str = "soft",
    use_swt: bool = True,
    level: Optional[int] = None,
) -> np.ndarray:
    """Вейвлет-денойз для 1D сигнала на РАВНОМЕРНОЙ сетке."""
    n = x.size
    if use_swt:
        maxlev = pywt.swt_max_level(n)
        L = maxlev if level is None else min(level, maxlev)
        coeffs = pywt.swt(x, wavelet, level=L)
        new_coeffs = []
        for cA, cD in coeffs:
            sigma = _estimate_sigma(cD)
            thr = _universal_threshold(sigma, n)
            cD = pywt.threshold(cD, thr, mode=mode)
            new_coeffs.append((cA, cD))
        rec = pywt.iswt(new_coeffs, wavelet)
        return rec[:n]
    else:
        w = pywt.Wavelet(wavelet)
        maxlev = pywt.dwt_max_level(n, w.dec_len)
        L = maxlev if level is None else min(level, maxlev)
        coeffs = pywt.wavedec(x, wavelet, level=L)
        cA = coeffs[0]
        out = [cA]
        for cD in coeffs[1:]:
            sigma = _estimate_sigma(cD)
            thr = _universal_threshold(sigma, n)
            out.append(pywt.threshold(cD, thr, mode=mode))
        rec = pywt.waverec(out, wavelet)
        return rec[:n]


def _interp1_shared_t(
    t: np.ndarray, y: np.ndarray, t_new: np.ndarray, kind: str = "pchip"
) -> np.ndarray:
    """Интерполяция всех столбцов y (N,R) с общим временем t (N,) на t_new (M,)."""
    if kind == "pchip":
        f = interpolate.PchipInterpolator(t, y, axis=0)
        return f(t_new)
    else:
        f = interpolate.interp1d(
            t, y, kind=kind, axis=0, bounds_error=False, fill_value="extrapolate"
        )
        return f(t_new)


def _interp1_per_channel(
    T: np.ndarray, y: np.ndarray, t_new: np.ndarray, kind: str = "pchip"
) -> np.ndarray:
    """Интерполяция по каналам при своих временах T (N,R). Возвращает (M,R)."""
    N, R = y.shape
    M = len(t_new)
    Y_new = np.empty((M, R), dtype=float)
    for r in range(R):
        t_r = T[:, r]
        y_r = y[:, r]
        if kind == "pchip":
            f = interpolate.PchipInterpolator(t_r, y_r)
            Y_new[:, r] = f(t_new)
        else:
            f = interpolate.interp1d(
                t_r, y_r, kind=kind, bounds_error=False, fill_value="extrapolate"
            )
            Y_new[:, r] = f(t_new)
    return Y_new


# ---------- основные пайплайны ----------
def denoise_matrix_uniform(
    Y: np.ndarray,
    wavelet: str = "db4",
    mode: str = "soft",
    use_swt: bool = True,
    level: Optional[int] = None,
) -> np.ndarray:
    """
    Случай A: РАВНОМЕРНАЯ сетка (времена не нужны).
    Применяет вейвлет-денойз к каждому каналу независимо.
    """
    Y = np.asarray(Y)
    N, R = Y.shape
    out = np.empty_like(Y, dtype=float)
    for r in range(R):
        xr = Y[:, r]
        # на всякий — заполним NaN монотонным PCHIP до денойза
        if np.isnan(xr).any():
            idx = np.flatnonzero(~np.isnan(xr))
            f = interpolate.PchipInterpolator(idx, xr[idx])
            xr = f(np.arange(N))
        out[:, r] = wavelet_denoise_1d(xr, wavelet, mode, use_swt, level)
    return out


def resample_and_denoise_shared_time(
    Y: np.ndarray,
    t: np.ndarray,
    M: Optional[int] = None,
    kind: str = "pchip",
    wavelet: str = "db4",
    mode: str = "soft",
    use_swt: bool = True,
    level: Optional[int] = None,
    back_to_original_time: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Случай B: общий НЕРЕГУЛЯРНЫЙ вектор времени t (N,).
    1) интерполяция Y(N,R) -> на равномерную сетку t_u(M)
    2) денойз по каналам
    3) опционально обратно на исходные t (N)
    Возвращает: (Y_denoised_on_t, t, Y_denoised_on_uniform)
    """
    Y = np.asarray(Y)
    t = np.asarray(t)
    N, R = Y.shape
    if M is None:
        M = int(2 ** np.ceil(np.log2(N)))  # удобно для вейвлетов
    t_u = np.linspace(t.min(), t.max(), M)
    # интерполяция на общую равномерную сетку
    Y_u = _interp1_shared_t(t, Y, t_u, kind=kind)
    # денойз на равномерной сетке
    Y_u_d = denoise_matrix_uniform(Y_u, wavelet, mode, use_swt, level)
    if back_to_original_time:
        # обратно на исходные времена
        Y_back = _interp1_shared_t(t_u, Y_u_d, t, kind=kind)
        return Y_back, t, Y_u_d
    else:
        return Y_u_d, t_u, Y_u_d


def resample_and_denoise_per_channel(
    Y: np.ndarray,
    T: Union[np.ndarray, List[np.ndarray]],
    M: Optional[int] = None,
    kind: str = "pchip",
    wavelet: str = "db4",
    mode: str = "soft",
    use_swt: bool = True,
    level: Optional[int] = None,
    common_time: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Случай C: у каждого канала свой вектор времени.
    T: (N,R) или список из R массивов длины N_r (поддерживает неравные N_r).
    1) интерполяция всех каналов на общую равномерную сетку t_u(M)
    2) денойз
    3) (опционально) интерполяция на внешний общий вектор времени common_time
    Возвращает: (Y_on_common_time, t_common, Y_on_uniform)
    """
    Y = np.asarray(Y)
    R = Y.shape[1]
    # определим объединённый диапазон времени
    if isinstance(T, list):
        t_min = min([np.asarray(ti).min() for ti in T])
        t_max = max([np.asarray(ti).max() for ti in T])
    else:
        T = np.asarray(T)
        t_min = np.nanmin(T)
        t_max = np.nanmax(T)
    if M is None:
        # возьмём ближайшую степень двойки около среднего N
        if isinstance(T, list):
            N_mean = int(np.mean([len(ti) for ti in T]))
        else:
            N_mean = T.shape[0]
        M = int(2 ** np.ceil(np.log2(max(N_mean, 32))))
    t_u = np.linspace(t_min, t_max, M)

    # интерполяция каждого канала на t_u
    if isinstance(T, list):
        # список времен разной длины
        Y_u = np.empty((M, R), dtype=float)
        for r in range(R):
            tr = np.asarray(T[r])
            yr = Y[: len(tr), r] if Y.shape[0] >= len(tr) else Y[:, r]
            if kind == "pchip":
                f = interpolate.PchipInterpolator(tr, yr)
                Y_u[:, r] = f(t_u)
            else:
                f = interpolate.interp1d(
                    tr, yr, kind=kind, bounds_error=False, fill_value="extrapolate"
                )
                Y_u[:, r] = f(t_u)
    else:
        # T как (N,R)
        Y_u = _interp1_per_channel(T, Y, t_u, kind=kind)

    # денойз на равномерной сетке
    Y_u_d = denoise_matrix_uniform(Y_u, wavelet, mode, use_swt, level)

    # вернуть на указанный общий вектор времени (если задан)
    if common_time is not None:
        Y_common = _interp1_shared_t(t_u, Y_u_d, np.asarray(common_time), kind=kind)
        return Y_common, np.asarray(common_time), Y_u_d
    else:
        return Y_u_d, t_u, Y_u_d
