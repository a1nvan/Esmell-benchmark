from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from esdata.denoise import _interp1_shared_t


def _runs_from_sign(arr: np.ndarray) -> list[tuple[int, int, int]]:
    """(cat, start, end_excl), cat ∈ {-1,0,+1}."""
    if len(arr) == 0:
        return []
    runs = []
    cur = int(arr[0])
    s = 0
    for i in range(1, len(arr)):
        c = int(arr[i])
        if c != cur:
            runs.append((cur, s, i))
            cur = c
            s = i
    runs.append((cur, s, len(arr)))
    return runs


def find_cycle_bounds_by_grad(
    df: pd.DataFrame,
    col: str = "Chip_grad_set",
    *,
    eps: float = 0.0,
    n_cycles_expected: int = 3,
    drop_unknown: bool = False,
) -> tuple[list[tuple[int, int]], pd.DataFrame]:
    """
    Ищет 3 цикла по паттерну серий знака Chip_grad_set:
        +1 -> 0 -> -1 -> 0
    Возвращает границы циклов в индексах df2.iloc[start:end] (end excl),
    где df2 = df (или df без NaN в grad, если drop_unknown=True).
    """
    if col not in df.columns:
        raise KeyError(f"В файле нет колонки '{col}'")

    g = pd.to_numeric(df[col], errors="coerce")
    if drop_unknown:
        df2 = df.loc[g.notna()].copy()
        g2 = pd.to_numeric(df2[col], errors="coerce")
    else:
        df2 = df.copy()
        g2 = g.fillna(0.0)

    vals = g2.to_numpy(dtype=float)

    # cat = np.zeros_like(vals, dtype=int)
    # if eps > 0:
    #     cat[vals > eps] = 1
    #     cat[vals < -eps] = -1
    # else:
    #     cat[vals > 0] = 1
    #     cat[vals < 0] = -1
    # runs = _runs_from_sign(cat)

    runs = _runs_from_sign(vals)
    # print(f"{runs = }")

    def comp_func(seq, target) -> np.bool_:
        return ((np.array(seq) > 0) == (np.array(target) > 0)).all()

    # pattern = (-1, 1, -1, 1, -1)
    pattern = (1, -1, 1, -1)
    n_pattern = len(pattern)

    bounds: list[tuple[int, int]] = []
    i = 0
    while i + (n_pattern - 1) < len(runs) and len(bounds) < n_cycles_expected:
        c, s, e = [], [], []
        for j in range(n_pattern):
            cj, sj, ej = runs[i + j]
            c.append(cj)
            s.append(sj)
            e.append(ej)

        if comp_func(c, pattern):
            # print(f"{c = } found!!!")
            # цикл = от начала нагрева до конца второго ожидания
            bounds.append((s[0], e[-1]))
            i += n_pattern
        else:
            i += 1

    if len(bounds) < n_cycles_expected:
        seq = [r[0] for r in runs]
        raise ValueError(
            f"Найдено циклов {len(bounds)} (ожидалось {n_cycles_expected}). "
            f"Последовательность серий (cat): {seq}"
        )

    # ВАЖНО: возвращаем bounds относительно df2, а df2 может отличаться от df
    # Поэтому также вернём df2 наружу в основном вызове.
    # print(f"{bounds = }")
    return bounds, df2


def interpolate_data(data: np.ndarray, M: int = 500):
    # print(f"{data.shape = }")

    Y = np.asarray(data)
    t = np.linspace(0, 100, data.shape[0])
    N, R = Y.shape
    if M is None:
        M = int(2 ** np.ceil(np.log2(N)))  # удобно для вейвлетов
    t_u = np.linspace(t.min(), t.max(), M)
    # интерполяция на общую равномерную сетку
    Y_u = _interp1_shared_t(t, Y, t_u, kind="pchip")

    return Y_u


def filter_segment(
    segment: np.ndarray, fs: float = 10.0, f_c: float = 1.0, btype: str = "low"
) -> np.ndarray:
    N = 1
    nyq = fs / 2
    Wn = f_c / nyq
    # print(f"{fs = }, {f_c = }, ({nyq = }) ={fs}/2, ({Wn = }) ={f_c}/{nyq}")
    b, a = butter(N, Wn, btype=btype, output="ba")

    filtred = np.zeros_like(segment)
    for i in range(segment.shape[1]):
        r = segment[:, i]
        filtred[:, i] = filtfilt(b, a, r)

    return filtred


def check_monotonic_and_increase(data: np.ndarray) -> Tuple[bool, bool]:
    a = data[1:] - data[:-1]
    s = np.sum(a)

    is_monotonic = False
    is_increase = s > 0

    if is_increase:
        is_monotonic = np.all(a > 0)
    else:
        is_monotonic = np.all(a < 0)

    return is_monotonic, is_increase


def interpolate_data_by_temp(
    r_cols: List[str],
    heating_segment: pd.DataFrame,
    temp_range: Tuple[float, float],
    M: int = 300,
) -> Tuple[np.ndarray, np.ndarray, bool]:

    t1, t2 = temp_range
    if t1 >= t2:
        raise ValueError(f"Wrong temperature range {t1} >= {t2}")

    _heating_segment = heating_segment[
        (heating_segment["Chip_temp_set"] >= t1) & (heating_segment["Chip_temp_set"] <= t2)
    ].copy()

    temp = _heating_segment["T_mean"].to_numpy()
    r_segment = _heating_segment[r_cols].to_numpy()

    temp = filter_segment(temp[..., np.newaxis], fs=40, f_c=1, btype="low").squeeze()

    is_monotonic, is_increase = check_monotonic_and_increase(temp)
    if is_monotonic == False:
        raise Exception("Temperature must be monotonic (check filtration)")

    is_flipped = False
    if is_increase == False:
        temp = np.flip(temp)
        r_segment = np.flip(r_segment, axis=0)
        is_flipped = True

    Y = np.asarray(r_segment)
    t = np.asarray(temp)
    N, R = Y.shape
    if M is None:
        M = int(2 ** np.ceil(np.log2(N)))  # удобно для вейвлетов
    t_u = np.linspace(t.min(), t.max(), M)
    # интерполяция на общую равномерную сетку
    Y_u = _interp1_shared_t(t, Y, t_u, kind="cubic")
    return t_u, Y_u, is_flipped


def interpolate_data_by_temp_col(
    r_cols: List[str],
    _segment1: pd.DataFrame,
    temp_range: Tuple[float, float],
    temp_col: str = "T_mean",
    M: int = 300,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    # temp_col "T_mean", "Chip_T0_C", "Chip_T1_C"

    t1, t2 = temp_range

    _segment = _segment1[(_segment1[temp_col] >= t1) & (_segment1[temp_col] <= t2)].copy()
    temp = _segment[temp_col].to_numpy()
    r_segment = _segment[r_cols].to_numpy()

    tempf = filter_segment(temp[..., np.newaxis], fs=40, f_c=1, btype="low").squeeze()

    # draw.plot_data([temp, tempf, ctemp], ["temp", "tempf", "ctemp"], f"Temp by {temp_col}")

    is_monotonic, is_increase = check_monotonic_and_increase(tempf)
    if is_monotonic == False:
        raise Exception("Temperature must be monotonic (check filtration)")

    is_flipped = False
    if is_increase == False:
        tempf = np.flip(tempf)
        r_segment = np.flip(r_segment, axis=0)
        is_flipped = True

    Y = np.asarray(r_segment)
    t = np.asarray(tempf)
    N, R = Y.shape
    if M is None:
        M = int(2 ** np.ceil(np.log2(N)))  # удобно для вейвлетов
    t_u = np.linspace(t.min(), t.max(), M)
    # интерполяция на общую равномерную сетку
    Y_u = _interp1_shared_t(t, Y, t_u, kind="cubic")
    return t_u, Y_u, is_flipped


def plot_data(temp, title):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(temp, "-o", label="temp")
    ax.legend()
    for fs in range(5, 15, 5):
        tempf = filter_segment(temp[..., np.newaxis], fs=fs, f_c=1, btype="low").squeeze()
        ax.plot(tempf, "-o", label=f"temp_{fs}")
        ax.legend()
        ax.grid(True)

    plt.title(title)
    plt.show()


def interpolate_data_by_canonical_temp_grid(
    r_cols: List[str],
    _segment1: pd.DataFrame,
    temp_range: Optional[Tuple[float, float]] = None,
    temp_col: str = "T_mean",
    M: Optional[int] = 300,
    fill_value: float = 0.0,
    canonical_temp_grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Interpolate one segment on a canonical temperature grid.

    The grid can be generated from ``temp_range`` and ``M`` or supplied
    explicitly through ``canonical_temp_grid``.
    """
    # temp_col "T_mean", "Chip_T0_C", "Chip_T1_C"
    if canonical_temp_grid is None:
        if temp_range is None:
            raise ValueError("temp_range is required when canonical_temp_grid is not set")

        t1, t2 = temp_range
        if t1 >= t2:
            raise ValueError(f"Wrong temperature range {t1} >= {t2}")
        if M is None:
            raise ValueError("M must be set for canonical temperature grid alignment")
        temp_grid = np.linspace(t1, t2, M)
    else:
        temp_grid = np.asarray(canonical_temp_grid, dtype=float)
        if temp_grid.ndim != 1 or len(temp_grid) < 2:
            raise ValueError("canonical_temp_grid must be a 1D array with at least two points")
        if not np.all(np.isfinite(temp_grid)) or np.any(np.diff(temp_grid) <= 0):
            raise ValueError("canonical_temp_grid must be finite and strictly increasing")
        t1, t2 = temp_grid[0], temp_grid[-1]

    # TODO: take full temp range, interpolate first, then filter by range
    _segment = _segment1[(_segment1[temp_col] >= t1) & (_segment1[temp_col] <= t2)].copy()
    _segment = _segment.dropna(subset=[temp_col, *r_cols])
    if _segment.empty:
        raise ValueError("Temperature segment is empty inside canonical range")

    temp = _segment[temp_col].to_numpy()
    r_segment = _segment[r_cols].to_numpy()

    # tempf = filter_segment(temp[..., np.newaxis], fs=100, f_c=1, btype="low").squeeze()
    tempf = filter_segment(temp[..., np.newaxis], fs=20, f_c=1, btype="low").squeeze()
    # tempf = temp
    # plot_data(temp, f"Temp by {temp_col}")
    # draw.plot_data([temp, tempf, ctemp], ["temp", "tempf", "ctemp"], f"Temp by {temp_col}")

    is_monotonic, is_increase = check_monotonic_and_increase(tempf)
    if is_monotonic == False:
        raise Exception("Temperature must be monotonic (check filtration)")

    is_flipped = False
    if is_increase == False:
        tempf = np.flip(tempf)
        r_segment = np.flip(r_segment, axis=0)
        is_flipped = True

    Y = np.asarray(r_segment)
    t = np.asarray(tempf)
    mask = ((temp_grid >= t.min()) & (temp_grid <= t.max())).astype(np.uint8)

    Y_u = np.full((len(temp_grid), Y.shape[1]), fill_value, dtype=float)
    if mask.any():
        Y_u[mask.astype(bool)] = _interp1_shared_t(
            t,
            Y,
            temp_grid[mask.astype(bool)],
            kind="cubic",
        )

    return temp_grid, Y_u, mask, is_flipped


def as_is_data_by_temp_col(
    r_cols: List[str],
    _segment1: pd.DataFrame,
    temp_range: Tuple[float, float],
    temp_col: str = "T_mean",
    M: int = 300,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    # temp_col "T_mean", "Chip_T0_C", "Chip_T1_C"

    t1, t2 = temp_range

    _segment = _segment1[
        (_segment1["Chip_temp_set"] >= t1) & (_segment1["Chip_temp_set"] <= t2)
    ].copy()

    temp = _segment[temp_col].to_numpy()
    # print(f"{temp = }")
    r_segment = _segment[r_cols].to_numpy()

    # tempf = filter_segment(temp[..., np.newaxis], fs=40, f_c=1, btype="low").squeeze()
    tempf = temp

    # draw.plot_data([temp, tempf, ctemp], ["temp", "tempf", "ctemp"], f"Temp by {temp_col}")

    is_monotonic, is_increase = check_monotonic_and_increase(tempf)
    if is_monotonic == False:
        raise Exception("Temperature must be monotonic (check filtration)")

    is_flipped = False
    if is_increase == False:
        tempf = np.flip(tempf)
        r_segment = np.flip(r_segment, axis=0)
        is_flipped = True

    Y = np.asarray(r_segment)
    t = np.asarray(tempf)
    # N, R = Y.shape
    # if M is None:
    #     M = int(2 ** np.ceil(np.log2(N)))  # удобно для вейвлетов
    # t_u = np.linspace(t.min(), t.max(), M)
    # # интерполяция на общую равномерную сетку
    # Y_u = denoise._interp1_shared_t(t, Y, t_u, kind="cubic")
    return t, Y, is_flipped


def get_compensated_segment(
    target_segment: pd.DataFrame,
    base_segment: pd.DataFrame,
    data_cols: List[str],
    title: List[str] = ["1", "2"],
    M: int = 150,
) -> np.ndarray:
    target_t_min = target_segment["Chip_temp_set"].min()
    target_t_max = target_segment["Chip_temp_set"].max()

    base_t_min = base_segment["Chip_temp_set"].min()
    base_t_max = base_segment["Chip_temp_set"].max()

    temp_range = np.max((target_t_min, base_t_min)), np.min((target_t_max, base_t_max))

    _target_temp, target_r_denoised, _ = interpolate_data_by_temp(
        data_cols, target_segment, temp_range, M=M
    )
    _base_temp, base_r_denoised, _ = interpolate_data_by_temp(
        data_cols, base_segment, temp_range, M=M
    )

    compensated = target_r_denoised - base_r_denoised
    compensated = np.concatenate([_target_temp[..., np.newaxis], compensated], axis=1)

    return compensated


def extract_gas_heating_segments(df, min_len=5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Находит интервалы вида:
    - подаётся газ (name == gas_name)
    - идёт нагрев (Chip_grad_set > 0)

    Параметры
    ---------
    df : pd.DataFrame
        Ожидаемые колонки: 'Time', 'name', 'Chip_grad_set', 'T_mean_adc'
    gas_name : str
        Имя газа в колонке name (по умолчанию 'gexogene')
    min_len : int
        Минимальное число точек в сегменте, чтобы считать его валидным
        (отрезаем шумовые вспышки)

    Возвращает
    ----------
    df_with_segments : pd.DataFrame
        Копия df, отсортированная по времени, с колонками:
        - gas_heating (bool)
        - gas_cooling (bool)
        - segment_id_heating (int или NaN)
        - segment_id_cooling (int или NaN)
    heating_segments : pd.DataFrame
        Сводка по найденным сегментам с колонками:
        - segment_id
        - start_time, end_time
        - start_T, end_T
        - mean_grad
        - n_points
    cooling_segments : pd.DataFrame
        Сводка по найденным сегментам с колонками:
        - segment_id
        - start_time, end_time
        - start_T, end_T
        - mean_grad
        - n_points
    """

    # 1. Отсортируем по времени и скопируем, чтобы не менять оригинал снаружи
    df_sorted = df.sort_values("Time").reset_index(drop=True).copy()

    # 2. Булева маска: газ + нагрев
    df_sorted["gas_heating"] = df_sorted["Chip_grad_set"] > 0
    df_sorted["gas_cooling"] = df_sorted["Chip_grad_set"] < 0

    # 3. Промежуточная нумерация "пробегами"
    #    Каждому непрерывному run True/False присваиваем новый id
    run_id = (
        df_sorted["gas_heating"] != df_sorted["gas_heating"].shift(fill_value=False)
    ).cumsum()
    df_sorted["segment_id_heating"] = np.where(df_sorted["gas_heating"], run_id, np.nan)

    run_id = (
        df_sorted["gas_cooling"] != df_sorted["gas_cooling"].shift(fill_value=False)
    ).cumsum()
    df_sorted["segment_id_cooling"] = np.where(df_sorted["gas_cooling"], run_id, np.nan)

    # 4. Аггрегируем только True-сегменты
    def agg_segments(period) -> pd.DataFrame:
        return (
            df_sorted[df_sorted[f"gas_{period}"]]
            .groupby(f"segment_id_{period}")
            .agg(
                start_time=("Time", "first"),
                end_time=("Time", "last"),
                start_T=("T_mean_adc", "first"),
                end_T=("T_mean_adc", "last"),
                mean_grad=("Chip_grad_set", "mean"),
                n_points=("Time", "size"),
            )
            .reset_index()
        )

    def agg_heating() -> pd.DataFrame:
        return agg_segments("heating")

    def agg_cooling() -> pd.DataFrame:
        return agg_segments("cooling")

    heating_segments = agg_heating()
    cooling_segments = agg_cooling()

    # 5. Фильтр по мин. длине, чтобы убрать единичные всплески
    def filter_segments(_segments, period) -> pd.DataFrame:
        segments_name = f"segment_id_{period}"
        if min_len is not None and min_len > 1:
            long_ids = _segments.loc[_segments["n_points"] >= min_len, segments_name]
            valid_ids = set(long_ids.tolist())
            # выставим NaN для коротких
            df_sorted.loc[~df_sorted[segments_name].isin(valid_ids), segments_name] = np.nan
            return _segments[_segments[segments_name].isin(valid_ids)].reset_index(drop=True)

        raise Exception(f"{min_len} is None or {min_len} <= 1")

    heating_segments = filter_segments(heating_segments, "heating")
    cooling_segments = filter_segments(cooling_segments, "cooling")

    return df_sorted, heating_segments, cooling_segments


def get_segment_data(df: pd.DataFrame, segments, segment_num: int = 0) -> pd.DataFrame:
    period = None
    if "segment_id_heating" in segments.columns:
        period = "segment_id_heating"
    elif "segment_id_cooling" in segments.columns:
        period = "segment_id_cooling"

    if segment_num >= len(segments):
        raise ValueError(f"Wrong segment num in period: {segment_num} >= {len(segments)}")

    segment_id = segments.loc[segment_num, period]
    return df[df[period] == segment_id]
