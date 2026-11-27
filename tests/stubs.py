import contextlib
import importlib.util
import sys
import types
from datetime import datetime, timedelta
import numpy as np


class FakeSeries:
    def __init__(self, values):
        self._arr = np.asarray(values, dtype=float)
        self.iloc = _ILoc(self._arr)

    def __len__(self):
        return len(self._arr)

    def __array__(self, dtype=None, copy=None):
        return self._arr if dtype is None else self._arr.astype(dtype)

    def __add__(self, other):
        val = other._arr if isinstance(other, FakeSeries) else other
        return FakeSeries(self._arr + val)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        val = other._arr if isinstance(other, FakeSeries) else other
        return FakeSeries(self._arr - val)

    def __rsub__(self, other):
        val = other._arr if isinstance(other, FakeSeries) else other
        return FakeSeries(val - self._arr)

    def __neg__(self):
        return FakeSeries(-self._arr)

    def __truediv__(self, other):
        val = other._arr if isinstance(other, FakeSeries) else other
        with np.errstate(divide="ignore", invalid="ignore"):
            return FakeSeries(self._arr / val)

    def __mul__(self, other):
        val = other._arr if isinstance(other, FakeSeries) else other
        return FakeSeries(self._arr * val)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __gt__(self, other):
        val = other._arr if isinstance(other, FakeSeries) else other
        return self._arr > val

    def __and__(self, other):
        val = other._arr if isinstance(other, FakeSeries) else other
        return self._arr & val

    def abs(self):
        return FakeSeries(np.abs(self._arr))

    def shift(self, n=1):
        out = np.empty_like(self._arr)
        out[:] = np.nan
        if n < len(self._arr):
            out[n:] = self._arr[:-n]
        return FakeSeries(out)

    def diff(self):
        return self - self.shift(1)

    def fillna(self, value=0.0):
        arr = self._arr.copy()
        arr[np.isnan(arr)] = value
        return FakeSeries(arr)

    def replace(self, old, new):
        arr = self._arr.copy()
        arr[arr == old] = new
        return FakeSeries(arr)

    def tail(self, n):
        return FakeSeries(self._arr[-n:])

    def mean(self):
        return float(np.nanmean(self._arr))

    def max(self):
        return float(np.nanmax(self._arr))

    def min(self):
        return float(np.nanmin(self._arr))

    def rolling(self, window):
        return _Rolling(self._arr, window)

    def ewm(self, span=None, alpha=None, adjust=False):
        a = alpha if alpha is not None else 2.0 / (span + 1.0)
        return _Ewm(self._arr, a)


class _ILoc:
    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, idx):
        return self._arr[idx]


class _Rolling:
    def __init__(self, arr, window):
        self._arr = arr
        self._w = window

    def mean(self):
        out = np.full_like(self._arr, np.nan, dtype=float)
        for i in range(self._w - 1, len(self._arr)):
            out[i] = np.nanmean(self._arr[i - self._w + 1 : i + 1])
        return FakeSeries(out)


class _Ewm:
    def __init__(self, arr, alpha):
        self._arr = arr
        self._alpha = alpha

    def mean(self):
        out = np.full_like(self._arr, np.nan, dtype=float)
        prev = np.nan
        for i, val in enumerate(self._arr):
            if np.isnan(val):
                out[i] = prev
            elif np.isnan(prev):
                prev = val
                out[i] = val
            else:
                prev = self._alpha * val + (1.0 - self._alpha) * prev
                out[i] = prev
        return FakeSeries(out)


class _DFIloc:
    def __init__(self, data, cols):
        self._data = data
        self._cols = cols

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return FakeDataFrame({k: self._data[k][idx] for k in self._cols})
        return {k: self._data[k][idx] for k in self._cols}


class FakeDataFrame:
    def __init__(self, data):
        self._data = {k: list(v) for k, v in data.items()}
        self.columns = list(self._data.keys())
        self.iloc = _DFIloc(self._data, self.columns)

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        if not self._data:
            return 0
        first = next(iter(self._data.values()))
        return len(first)

    def __getitem__(self, key):
        if isinstance(key, list):
            return FakeDataFrame({k: self._data[k] for k in key})
        vals = self._data[key]
        if key == "time":
            return _TimeSeries(vals)
        return FakeSeries(vals)

    @property
    def index(self):
        return list(range(len(self)))

    def __setitem__(self, key, val):
        if isinstance(val, FakeSeries):
            self._data[key] = list(val._arr)
        elif isinstance(val, _TimeSeries):
            self._data[key] = list(val._vals)
        else:
            self._data[key] = list(val)
        if key not in self.columns:
            self.columns.append(key)

    def set_index(self, col, inplace=False):
        target = self if inplace else self.copy()
        target._data.pop(col, None)
        if col in target.columns:
            target.columns.remove(col)
        return None if inplace else target

    def rename(self, columns=None, inplace=False):
        target = self if inplace else self.copy()
        for old, new in (columns or {}).items():
            if old in target._data:
                target._data[new] = target._data.pop(old)
                target.columns = [new if c == old else c for c in target.columns]
        return None if inplace else target

    def copy(self):
        return FakeDataFrame(self._data)

    def tail(self, n):
        return FakeDataFrame({k: v[-n:] for k, v in self._data.items()})

    def dropna(self, subset=None):
        cols = subset or self.columns
        keep = [
            i for i in range(len(self))
            if all(self._data[c][i] is not None and not _is_nan(self._data[c][i]) for c in cols)
        ]
        return FakeDataFrame({k: [v[i] for i in keep] for k, v in self._data.items()})

    def mean(self, axis=0):
        if axis == 1:
            n = len(self)
            return FakeSeries([
                sum(self._data[c][i] for c in self.columns) / len(self.columns) for i in range(n)
            ])
        return FakeSeries([np.mean(self._data[c]) for c in self.columns])

    def iterrows(self):
        for i in range(len(self)):
            yield i, {k: self._data[k][i] for k in self.columns}


class _TimeSeries:
    def __init__(self, vals):
        self._vals = list(vals)
        self.iloc = _ILoc(self._vals)

    def __len__(self):
        return len(self._vals)


def _is_nan(v):
    return isinstance(v, float) and np.isnan(v)


def _fake_concat(objs, axis=1):
    mat = np.column_stack([o._arr for o in objs])
    return _ConcatResult(mat)


class _ConcatResult:
    def __init__(self, mat):
        self._mat = mat

    def max(self, axis=1):
        return FakeSeries(np.nanmax(self._mat, axis=axis))


def _fake_to_datetime(vals):
    if isinstance(vals, _TimeSeries):
        vals = vals._vals
    parsed = [v if isinstance(v, datetime) else datetime.fromisoformat(str(v)) for v in vals]
    return _TimeSeries(parsed)


def _fake_date_range(start, periods, freq="15min"):
    delta = timedelta(hours=1) if freq == "1h" else timedelta(minutes=15)
    return [start + i * delta for i in range(periods)]


def _install_pandas_stub():
    if "pandas" in sys.modules or importlib.util.find_spec("pandas") is not None:
        return
    pd_mod = types.ModuleType("pandas")
    pd_mod.DataFrame = FakeDataFrame
    pd_mod.Series = lambda data, index=None: FakeSeries(data)
    pd_mod.concat = _fake_concat
    pd_mod.to_datetime = _fake_to_datetime
    pd_mod.date_range = _fake_date_range
    pd_mod.DatetimeIndex = tuple
    pd_mod.MultiIndex = tuple
    sys.modules["pandas"] = pd_mod


def _install_torch_stub():
    if "torch" in sys.modules or importlib.util.find_spec("torch") is not None:
        return
    torch_mod = types.ModuleType("torch")
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = type("Module", (), {})
    nn_mod.Linear = lambda *a, **k: None
    nn_mod.Embedding = lambda *a, **k: None
    nn_mod.Dropout = lambda *a, **k: None
    nn_mod.ModuleList = lambda *a, **k: []
    func_mod = types.ModuleType("torch.nn.functional")
    autograd_mod = types.ModuleType("torch.autograd")
    autograd_mod.Function = type("Function", (), {})
    torch_mod.nn = nn_mod
    torch_mod.autograd = autograd_mod
    torch_mod.Tensor = object
    torch_mod.no_grad = contextlib.nullcontext
    torch_mod.inference_mode = contextlib.nullcontext
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch_mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = func_mod
    sys.modules["torch.autograd"] = autograd_mod


def install_missing_stubs():
    _install_pandas_stub()
    _install_torch_stub()
    simple_stubs = {
        "yfinance": {"download": lambda *a, **k: FakeDataFrame({})},
        "requests": {"get": lambda *a, **k: None, "post": lambda *a, **k: None, "RequestException": Exception},
        "schedule": {"every": lambda *a, **k: types.SimpleNamespace(minutes=types.SimpleNamespace(do=lambda fn: None)), "run_pending": lambda: None},
        "matplotlib": {"use": lambda *a, **k: None, "__path__": []},
        "matplotlib.pyplot": {"close": lambda *a, **k: None},
        "mplfinance": {"make_marketcolors": lambda **k: None, "make_mpf_style": lambda **k: None, "plot": lambda *a, **k: None},
        "huggingface_hub": {"PyTorchModelHubMixin": type("PyTorchModelHubMixin", (), {}), "login": lambda **k: None},
        "einops": {"rearrange": lambda x, *a, **k: x, "reduce": lambda x, *a, **k: x},
        "tqdm": {"trange": range},
    }
    for name, attrs in simple_stubs.items():
        top_pkg = name.split(".")[0]
        if name not in sys.modules and (top_pkg in sys.modules or importlib.util.find_spec(top_pkg) is None):
            mod = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(mod, k, v)
            sys.modules[name] = mod
    if "matplotlib" in sys.modules and "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]
