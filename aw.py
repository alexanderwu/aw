"""Alex Wu's utility functions for data processing, visualization, and reporting.

| Metadata      | Description             |
|---------------|-------------------------|
| Author        | Alexander Wu            |
| Email         | alexander.wu7@gmail.com |
"""
import itertools
import warnings
from collections.abc import Callable, Iterable
from functools import cache, reduce, wraps
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from IPython.display import HTML, Markdown, display


def reload(module=None):
    """Reload module (for use in Jupyter Notebook)

    Args:
        module (types.ModuleType, optional): module to reload
    """
    import sys
    import importlib
    importlib.reload(module or sys.modules[__name__])

def _copy(text: str) -> None:
    """Copy text to clipboard.

    Args:
        text (str): text to copy to clipboard
    """
    # Source: https://stackoverflow.com/questions/11063458/python-script-to-copy-text-to-clipboard
    try:
        import pyperclip  # type: ignore
        pyperclip.copy(text)
    except ModuleNotFoundError:
        import sys
        sys.stderr.write("Cannot copy. Try `pip install pyperclip`\n")


def dirr(arg, like: str=None) -> pd.DataFrame:
    """Displays dir(arg) but with more details and formatted as DataFrame.

    Args:
        arg (Any): python object
        like (str, optional): filter string. Defaults to None.
    """
    def get_attr(arg, x: str) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                return getattr(arg, x)
            except AttributeError:
                return '!'
    print(type(arg))
    dirr_list = [x for x in dir(arg) if not x.startswith('_')]
    dirr_df = pd.DataFrame({'attr': dirr_list})
    dirr_df['type'] = [type(get_attr(arg, x)) for x in dirr_list]
    if like is not None:
        dirr_df = dirr_df[dirr_df['attr'].str.contains(like)]
    dirr_df['doc'] = [get_attr(arg, attr).__doc__ if str(tt) == "<class 'method'>" else ''
                      for attr, tt in zip(dirr_df['attr'], dirr_df['type'])]
    dirr_df['doc'] = dirr_df['doc'].astype(str).str.split(r'\.\n').str[0].str.strip()
    dirr_df['doc'] = [get_attr(arg, attr) if str(tt) != "<class 'method'>" else doc
                      for attr, tt, doc in zip(dirr_df['attr'], dirr_df['type'], dirr_df['doc'])]
    return dirr_df


def ls(path: Path | str = '.', resolve=False) -> pd.DataFrame:
    """View contents of `ls` command as DataFrame.

    Args:
        path (Path | str, optional): Path name to `ls`. Defaults to '.'.
        resolve (bool, optional): Resolve to absolute path. Defaults to False.

    Raises:
        ValueError: Invalid path name

    Returns:
        pd.DataFrame: contents of `ls`
    """
    match path:
        case Path():
            pass
        case '~':
            path = Path.home()
        case str():
            path = Path(path)
        case _:
            raise ValueError('invalid path')
    if resolve:
        path = path.resolve()
    df = DF({path: path.iterdir()})
    df.index += 1
    def g(self, row=1):
        return self.loc[row].iloc[0]
    def open(self, row=None):
        import subprocess
        from pathlib import PureWindowsPath
        posix_path = self.loc[row].iloc[0].resolve() if row is not None else PureWindowsPath(path.resolve())
        windows_path = PureWindowsPath(posix_path)
        subprocess.run(['explorer.exe', windows_path])
    df.g = g.__get__(df)
    df.open = open.__get__(df)
    return df


def mkdir(path: Path | str, **kwargs) -> pd.DataFrame:
    """Make directory.

    Args:
        path (Path | str): Directory to create

    Raises:
        ValueError: Invalid path name
    """
    match path:
        case Path():
            pass
        case str():
            path = Path(path)
        case _:
            raise ValueError('invalid path')
    if 'parents' not in kwargs:
        kwargs['parents'] = True
    if 'exist_ok' not in kwargs:
        kwargs['exist_ok'] = True
    path.mkdir(**kwargs)


def S(*args, **kwargs) -> pd.Series:
    """Create PyArrow Pandas Series.

    Returns:
        pd.Series: pd.Series with pyarrow data types
    """
    df = pd.Series(*args, **kwargs).convert_dtypes(dtype_backend='pyarrow')
    return df

def DF(*args, **kwargs) -> pd.Series:
    """Create PyArrow Pandas DataFrame.

    Returns:
        pd.DataFrame: pd.DataFrame with pyarrow data types
    """
    df = pd.DataFrame(*args, **kwargs).convert_dtypes(dtype_backend='pyarrow')
    return df

def wrap_series(fn: Callable) -> Callable:
    """Allows Pandas series operations to apply for other input"""
    def wrapper(series, *args):
        not_series = False
        if not isinstance(series, pd.Series):
            not_series = True
            series = pd.Series(series)
        res = fn(series, *args)
        if not_series:
            res = res.iloc[0]
        return res
    return wrapper

def decorator(func: Callable) -> Callable:
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        # Do something before
        value = func(*args, **kwargs)
        # Do something after
        return value
    return wrapper_decorator

def ignore_warnings(func: Callable) -> Callable:
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = func(*args, **kwargs)
            return value
    return wrapper_decorator

def debug(fn: Callable) -> Callable:
    import time
    def wrapper(*args):
        t1 = time.time()
        result = fn(*args)
        t2 = time.time()
        print(f'{fn.__name__}{args} : {result} ({t2-t1:.1f} s)')
        return result
    return wrapper


def get_sessions(pd_series: pd.Series, diff=pd.Timedelta(30, 'min')) -> pd.Series:
    """Group elements into "sessions".

    Compute groups (sessions) chained together by `diff` units. Assumes pd_series is sorted.

    Args:
        pd_series (pd.Series): Input values
        diff (_type_, optional): Maximum difference between first element and last element.
            Defaults to pd.Timedelta(30, 'min').

    Returns:
        pd.Series: Grouped elements
    """
    assert pd_series.is_monotonic_increasing

    current_session = pd_series.iloc[0]
    sessions = [current_session]

    for item in pd_series.iloc[1:]:
        if sessions[-1] + diff <= item:
            current_session = item
        sessions.append(current_session)
    return pd.Series(sessions)

def date2name(pd_series: pd.Series) -> pd.Series:
    """Convert to date to day of week.

    Args:
        pd_series (pd.Series): Input datetimes

    Returns:
        pd.Series: Day of week names
    """
    DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_name_series = pd.Categorical(pd_series.dt.day_name(), categories=DAY_NAMES)
    return day_name_series

def add_prefix(df: pd.DataFrame, prefix: str, subset=None, regex=None) -> pd.DataFrame:
    """Add prefix to columns

    Args:
        df (pd.DataFrame): Input dataframe
        prefix (str): prefix string to prepend
        subset (_type_, optional): subset of columns to affect. Defaults to None.
        regex (_type_, optional): regex filter of columns to affect. Defaults to None.

    Returns:
        pd.DataFrame: Transformed dataframe with certain columns prepended with prefix
    """
    cols = list(df.columns)
    if regex is not None:
        cols = list(df.columns.str.contains(regex))
    if isinstance(subset, str):
        subset = [subset]
    if hasattr(subset, '__contains__'):
        cols = [col for col in cols if col in subset]
    df_prefix = df.rename(columns={col: f'{prefix}{col}' for col in cols})
    return df_prefix

def add_suffix(df: pd.DataFrame, suffix: str, subset=None, regex=None) -> pd.DataFrame:
    """Add suffix to columns

    Args:
        df (pd.DataFrame): Input dataframe
        suffix (str): suffix string to append
        subset (_type_, optional): subset of columns to affect. Defaults to None.
        regex (_type_, optional): regex filter of columns to affect. Defaults to None.

    Returns:
        pd.DataFrame: Transformed dataframe with certain columns appended with suffix
    """
    cols = list(df.columns)
    if regex is not None:
        cols = list(df.columns.str.contains(regex))
    if isinstance(subset, str):
        subset = [subset]
    if hasattr(subset, '__contains__'):
        cols = [col for col in cols if col in subset]
    df_suffix = df.rename(columns={col: f'{col}{suffix}' for col in cols})
    return df_suffix

def overlaps(x_interval, y_interval) -> list:
    # TODO: Optimize O(XY) to O(X + Y) algo
    '''Compute overlaps. Assumes non_overlapping_monotonic_increasing.

    There are 9 ways Y_interval and overlap with X_interval

        |   XXX   | X_interval   | ( 3,   5 ) |                                       |
        |---------|--------------|------------|---------------------------------------|
        | _YY_    | left         | (_2 , _4 ) | (x_begin < y_begin) & (x_end < y_end) |
        | _YYYY   | left_spill   | (_2 ,  5 ) | (x_begin < y_begin) & (x_end = y_end) |
        | _YYYYY_ | superset     | (_2 ,  6_) | (x_begin < y_begin) & (x_end > y_end) |
        |   Y_    | left_subset  | ( 3 , _4 ) | (x_begin = y_begin) & (x_end < y_end) |
        |   YYY   | equal        | ( 3 ,  5 ) | (x_begin = y_begin) & (x_end = y_end) |
        |   YYYY_ | right_spill  | ( 3 ,  6_) | (x_begin = y_begin) & (x_end > y_end) |
        |    _    | subset       | (_4_, _4_) | (x_begin > y_begin) & (x_end < y_end) |
        |    _Y   | right_subset | (_4 ,  5 ) | (x_begin > y_begin) & (x_end = y_end) |
        |    _YY_ | right        | (_4 ,  6_) | (x_begin > y_begin) & (x_end > y_end) |
        |---------|--------------|------------|---------------------------------------|
        | __      | no_overlap   | (_2 ,  6_) | (x_begin > y_end)                     |
        |      __ | no_overlap   | (_2 ,  6_) | (x_end   < y_begin)                   |
        |---------|--------------|------------|---------------------------------------|
        | 1234567 |              |            |                                       |
    '''
    overlaps_list = []
    for _, (x_begin, x_end) in enumerate(x_interval):
        x_overlap_list = []
        for y_i, (y_begin, y_end) in enumerate(y_interval):
            # Case: no_overlap
            #if x_begin > y_end or x_end < y_begin:
            if x_begin >= y_end or x_end <= y_begin:
                continue

            begin_order = '>' if x_begin < y_begin else '<' if x_begin > y_begin else '='
            end_order = '>' if x_end < y_end else '<' if x_end > y_end else '='
            overlap_str = f'{begin_order}{end_order}'
            overlap_tuple = (y_i, overlap_str)
            x_overlap_list.append(overlap_tuple)

        overlaps_list.append(x_overlap_list)
    return overlaps_list


def df_overlaps(df1: pd.DataFrame, df2: pd.DataFrame, suffixes=('1', '2')) -> pd.DataFrame:
    """Merge based on overlapping 'start', 'end' variables

    Args:
        df1 (pd.DataFrame): Input dataframe 1
        df2 (pd.DataFrame): Input dataframe 2
        suffixes (tuple, optional): identify corresponding columns with this suffix. Defaults to ('1', '2').

    Returns:
        pd.DataFrame: Merged dataframe (based on corresponding 'start' and 'end' of input dataframes)
    """
    assert 'start' in df1.columns and 'end' in df1.columns and 'i' not in df1.columns
    assert 'start' in df2.columns and 'end' in df2.columns and 'i' not in df2.columns
    assert df1['start'].is_monotonic_increasing & df1['end'].is_monotonic_increasing
    assert df2['start'].is_monotonic_increasing & df2['end'].is_monotonic_increasing
    assert all(df1['start'] <= df1['end'])
    assert all(df2['start'] <= df2['end'])
    df1 = df1.reset_index(names='i')
    df2 = df2.reset_index(names='i')
    X_interval = list(df2[['start', 'end']].itertuples(index=False, name=None))
    Y_interval = list(df1[['start', 'end']].itertuples(index=False, name=None))
    overlaps_list = overlaps(X_interval, Y_interval)
    index_list = [[y_i for y_i, _ in x_list] for x_list in overlaps_list]
    overlap_list = [[overlap for _, overlap in x_list] for x_list in overlaps_list]
    i1, _ = f'i{suffixes[0]}', f'i{suffixes[1]}'
    overlap_df = (df2.pipe(add_suffix, suffixes[1])
                  .assign(**{i1: index_list, 'overlap': overlap_list})
                  .explode([i1, 'overlap']))
    overlap_df = overlap_df.merge(df1.pipe(add_suffix, suffixes[0]), on=i1, how='outer')
    return overlap_df

def itertuples(df, **kwargs):
    # Roughly same as `.itertuples(index=False, name=None))`
    kwargs['index'] = False
    kwargs['name'] = None
    df_list = list(df.itertuples(**kwargs))
    return df_list

################################################################################
# Utility functions
################################################################################

def size(num, prefix='', deep=True, verbose=True):
    """Human readable file size (ex: 123.4 KB)"""
    x = num
    if not isinstance(x, (int, float)):
        num = len(num)
    if isinstance(x, (str, set, dict, list)):
        return print(f'{num:,}') if verbose else f'{num:,}'
    if isinstance(x, pd.DataFrame):
        x = x.memory_usage(deep=deep).sum()
    if isinstance(x, pd.Series):
        x = x.memory_usage(deep=deep)

    for unit in ('bytes', 'KB', 'MB', 'GB', 'TB'):
        if abs(x) < 1024:
            return print(f'{prefix}: {num:,}  ({x:3.1f}+ {unit})') if verbose else (f'{num:,}  ({x:3.1f}+ {unit})')
        x /= 1024
    print(f'{prefix}: {num:,}  ({x:.1f}+ PB)') if verbose else (f'{num:,}  ({x:.1f}+ PB)')

@cache
def _read_file(filename: Path | str, base='data', verbose=True, **kwargs) -> pd.DataFrame:
    match filename:
        case Path():
            base = filename.parent
            filename = filename.name
        case str():
            pass
        case _:
            raise ValueError
    if '.' not in filename:
        filename = f'{filename}.feather'
    P_READ = Path(base) / filename
    assert P_READ.exists()
    if filename.endswith('.feather'):
        df = pd.read_feather(P_READ, **kwargs)
    elif filename.endswith('.parquet'):
        df = pd.read_parquet(P_READ, **kwargs)
    elif filename.endswith('.parquet.gzip'):
        df = pd.read_parquet(P_READ, **kwargs)
    elif filename.endswith('.pkl'):
        df = pd.read_pickle(P_READ, **kwargs)
    elif filename.endswith('.csv'):
        df = pd.read_csv(P_READ, **{'dtype_backend': 'pyarrow', **kwargs})
    else:
        raise ValueError
    if verbose:
        df.pipe(size, prefix=filename)
    return df

# https://stackoverflow.com/questions/56544334/disable-functools-lru-cache-from-inside-function
def read_file(filename: Path | str, overwrite=False, base='data', verbose=True, **kwargs) -> pd.DataFrame:
    """Read serialized file, caching the result

    Args:
        filename (Path | str): File to read in
        overwrite (bool, optional): Overwrite cache. Defaults to False.
        base (str, optional): Base filepath. Defaults to 'data'.
        verbose (bool, optional): Print what's going on. Defaults to True.

    Returns:
        pd.DataFrame: Input file as DataFrame
    """
    if overwrite:
        return _read_file.__wrapped__(filename, base=base, verbose=verbose, **kwargs)
    return _read_file(filename, base=base, verbose=verbose, **kwargs)

def rm_file(filename: Path | str, base='data', verbose=True) -> None:
    """Remove file

    Args:
        filename (Path | str): file to remove
        base (str, optional): Base filepath. Defaults to 'data'.
        verbose (bool, optional): Print what's going on. Defaults to True.

    Raises:
        ValueError: Invalid filename
    """
    match filename:
        case Path():
            base = filename.parent
            filename = filename.name
        case str():
            pass
        case _:
            raise ValueError
    if '.' not in filename:
        filename = f'{filename}.feather'
    P_REMOVE = Path(base) / filename
    if P_REMOVE.exists():
        if verbose:
            size(P_REMOVE.stat().st_size, prefix=f'Deleting "{P_REMOVE}"')
        P_REMOVE.unlink()
    else:
        print(f'"{P_REMOVE}" does not exist...')
    if P_REMOVE.parent.exists() and not any(P_REMOVE.parent.iterdir()):
        print(f'Removing empty directory: "{P_REMOVE.parent}"...')
        P_REMOVE.parent.rmdir()

def read_csv(*args, **kwargs):
    kwargs['dtype_backend'] = 'pyarrow'
    df = pd.read_csv(*args, **kwargs)
    return df

def write_file(df: pd.DataFrame, filename: Path | str, overwrite=False, base='data', verbose=True, **kwargs) -> None:
    """Write serialized file

    Args:
        df (pd.DataFrame): DataFrame to save to disk
        filename (Path | str): Filename
        overwrite (bool, optional): Overwrite file. Defaults to False.
        base (str, optional): Base path. Defaults to 'data'.
        verbose (bool, optional): Verbose. Defaults to True.

    Raises:
        ValueError: Invalid filename
        ValueError: Invalid file type
    """
    match filename:
        case Path():
            base = filename.parent
            filename = filename.name
        case str():
            pass
        case _:
            raise ValueError
    if '.' not in filename:
        df = DF(df)
        filename = f'{filename}.feather'
    P_WRITE = Path(base) / filename
    if overwrite or not P_WRITE.exists():
        P_WRITE.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith('.feather'):
            df.to_feather(P_WRITE, **kwargs)
        elif filename.endswith('.parquet'):
            df.to_parquet(P_WRITE, **kwargs)
        elif filename.endswith('.parquet.gzip'):
            df.to_parquet(P_WRITE, **{'compression': 'gzip', **kwargs})
        elif filename.endswith('.pkl'):
            df.to_pkl(P_WRITE, **kwargs)
        elif filename.endswith('.csv'):
            df.to_csv(P_WRITE, **{'index': False, **kwargs})
        else:
            raise ValueError
        df.pipe(size, prefix='(DataFrame rows)')
    if verbose:
        size(P_WRITE.stat().st_size, prefix=P_WRITE)

################################################################################
# (For Jupyter)
################################################################################

# Source: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook/39662359#39662359
def isnotebook() -> bool:
    """Detect if code is running in Jupyter Notebook

    Returns:
        bool: True = code is running in Jupyter Notebook
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def disp(df: pd.DataFrame, caption='', k=2, na_rep='-') -> pd.DataFrame:
    """(For Jupyter) Prints newlines instead of '\\\\n' characters for easier reading.
    Optionally, you can label dataframes with caption and round numbers

    Args:
        df (pd.DataFrame): Input DataFrame
        caption (str, optional): Caption for df. Defaults to ''.
        k (int, optional): Round to k digits. Defaults to 2.
        na_rep (str, optional): Str representation of NA values. Defaults to '-'.

    Returns:
        pd.DataFrame: Styled DataFrame with optional caption
    """
    assert isnotebook()
    # Ensure row names and column names are unique
    df = df.pipe(df_enumerate)
    df = df.style if hasattr(df, 'style') else df
    df_captioned = (df.format(lambda x: str_round(x, k=k),
                              na_rep=na_rep,
                              subset=df.data.select_dtypes(exclude=object).columns)
                      .set_properties(**{'white-space': 'pre-wrap', 'text-align': 'left'})
                      .set_table_attributes("style='display:inline'")
                      .set_caption(caption))
    return df_captioned

# Derived from: https://stackoverflow.com/a/57832026
def displays(*args, captions: list[str] = None, k=2, na_rep='-'):
    """
    (For Jupyter)
    Display tables side by side to save vertical space.
    Prints newlines instead of '\n' characters for easier reading.
    Optionally, you can label dataframes with captions

    Input:
        args: list of pandas.DataFrame
        captions: list of table captions
    """
    assert isnotebook()
    if captions is None:
        captions = []
    if isinstance(captions, str):
        captions = [captions]
    if k is None:
        k = []

    args = (*args, pd.DataFrame())
    args = [arg.to_frame().style.hide_index() if isinstance(arg, pd.Series) else arg for arg in args]
    k_list = [k]*len(args) if isinstance(k, int) else k
    k_list.extend([None] * (len(args) - len(k_list)))
    captions.extend([''] * (len(args) - len(captions)))
    captioned_tables = [df.pipe(disp, caption, k, na_rep)._repr_html_()
                        for caption, df, k in zip(captions, args, k_list)]
    display(HTML('\xa0\xa0\xa0'.join(captioned_tables)))

# Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable: Iterable) -> Iterable:
    """Groups elements pairwise: s -> (s0,s1), (s1,s2), (s2, s3), ...

    Args:
        iterable (Iterable): Input iterable

    Returns:
        Iterable: zipped iterable
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def display100(df: pd.Series | pd.DataFrame, ii=10, N=100, na_rep=None) -> None:
    """Display N elements with ii elements each column

    Args:
        df (pd.Series, pd.DataFrame): Input values
        ii (int, optional): # elements in column. Defaults to 10.
        N (int, optional): # elements to display. Defaults to 100.
        na_rep (_type_, optional): Missing str representation. Defaults to None.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    N = min(N, len(df))
    displays(*[df.iloc[a:b] for a, b in pairwise(range(0, N+ii, ii))], na_rep=na_rep)
disp100 = display100
d100 = display100

def display_code(code: str, language: str = 'python'):
    markdown_code = f'```{language}\n{code}\n```'
    if isnotebook():
        display(Markdown(markdown_code))
    else:
        print(markdown_code)

# https://stackoverflow.com/questions/624926/how-do-i-detect-whether-a-python-variable-is-a-function
def show(item, hide_docstring: bool = False):
    """(For Jupyter) Displays function source code or JSON output"""
    if callable(item):
        import inspect
        code = inspect.getsource(item)
        if hide_docstring:
            function_text = [code.split('"""')[0], *code.split('"""')[2:]]
            code = ''.join([x.rstrip() for x in function_text])
        display_code(code)
    elif isnotebook():
        if isinstance(item, dict):
            import json
            display_code(json.dumps(item, indent=2), 'json')
        elif isinstance(item, str):
            display(Markdown(item))
        else:
            return type(item)
    else:
        return type(item)

def percent(pd_series: pd.Series, caption='', display_false=False) -> pd.DataFrame:
    """Display percentage

    Args:
        pd_series (pd.Series): Input values
        caption (str, optional): Caption. Defaults to ''.
        display_false (bool, optional): Display percentage of False values. Defaults to False.

    Returns:
        pd.DataFrame: Displayed percentage
    """
    # df = pd.value_counts(pd_series).to_frame().T
    df = pd_series.value_counts().to_frame().T
    if True not in df:
        df[True] = 0
    if False not in df:
        df[False] = 0
    df['Total'] = len(pd_series)
    df['%'] = 100*df[True] / df['Total']
    if not display_false:
        df = df.rename(columns={True: 'N'})
        df = df.drop(columns=[False])
    styled_df = (df.style.hide()
            .bar(vmin=0, vmax=100, color='#543b66', subset=['%'])
            .format('{:,.1f}', subset=['%']))
    if caption:
        styled_df = styled_df.set_caption(caption)
    return styled_df
perc = percent

def append_percent(df: pd.DataFrame, col=None, vmax=None, verbose=False, inplace=False) -> pd.DataFrame:
    if not inplace:
        df = df.copy()

    if col is None:
        col = df.shape[1] - 1

    if isinstance(col, int):
        pd_series = df.iloc[:,col]
        col_i = col + 1
        col_name = '%'
    else:
        pd_series = df[col]
        col_i = df.columns.to_list().index(col) + 1
        col_name = f'{df.columns[col_i-1]} %'


    pd_series = pd.to_numeric(pd_series, errors='coerce')
    if vmax is None:
        vmax = pd_series.sum()

    percent_series = 100 * pd_series / vmax
    df.insert(col_i, col_name, percent_series)

    if verbose:
        # print(f'Total: {vmax:,}')
        styled_df = (df.style
            .bar(vmin=0, vmax=100, color='#543b66', subset=[col_name])
            .format('{:,.1f}', subset=[col_name])
        )
        return styled_df
    if not inplace:
        return df
append_perc = append_percent

def vcounts(pd_series, cutoff=20, vmax=None, sort_index=False, verbose=True, **kwargs):
    data = pd_series.value_counts(**kwargs)
    if sort_index:
        data = data.sort_index()
    if vmax is None:
        vmax = data.sum()
    if len(data) > cutoff:
        other = pd.Series([data[cutoff:].sum()], index=['(Other)'])
        if isinstance(pd_series, pd.DataFrame):
            # other.index = pd.MultiIndex.from_tuples([('(Other)',) * data.index.nlevels])
            other_index = (*['-']*(data.index.nlevels-1), '(Other)')
            other.index = pd.MultiIndex.from_tuples([other_index])
        data = pd.concat([data[:cutoff], other])

    data_df = data.reset_index()
    if isinstance(pd_series, pd.Series):
        data_df.rename(columns={'index': pd_series.name}, inplace=True)
    data_df.columns = [*data_df.columns[:-1], 'N']
    data_df.index += 1
    value_counts_df = data_df.pipe(append_percent, vmax=vmax, verbose=verbose)
    return value_counts_df

def describe(pd_series, caption='', count=False):
    df = pd_series.describe().to_frame().T
    df['IQR'] = df['75%'] - df['25%']
    num_missing = pd_series.isna().sum()
    if num_missing > 0:
        df['N/A'] = num_missing
    if not count:
        df = df.drop(columns=['count'])
    if caption:
        styled_df = df.style.hide().set_caption(caption).format(
            lambda x: int(x) if x == int(x) else round(x, 4))
        display(styled_df)
    else:
        return df

def uniq(x_list):
    unique_list = list(set([x for x in x_list if pd.notnull(x)]))
    return unique_list

def duplicates(df, subset=None, keep=False):
    duplicates_df = df[df.duplicated(subset=subset, keep=keep)]
    return duplicates_df

def df_index(df, verbose=False, k=False):
    if hasattr(df, 'data'):
        df = df.data
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    index_name = tuple(['-']*df.columns.nlevels) if df.columns.nlevels > 1 else '-'
    index_df = df.iloc[:,0].map(lambda _: '').to_frame(name=index_name)
    if verbose:
        displays(index_df, df.style.hide(), k=k)
    else:
        return index_df

def df_enumerate(df, rows=None, columns=None, inplace=False):
    enumerate_rows = (not df.index.is_unique or rows) and rows is not False
    enumerate_columns = (not df.columns.is_unique or columns) and columns is not False
    if not inplace and (enumerate_rows or enumerate_columns):
        df = df.copy()
    if enumerate_rows:
        df.index = pd.MultiIndex.from_tuples([(i, *x) if isinstance(x, tuple) else (i,x)
                                              for i, x in enumerate(df.index, 1)])
        # df = df.set_index(pd.MultiIndex.from_tuples([(i, *x) if isinstance(x, tuple) else (i,x)
        #                                              for i, x in enumerate(df.columns, 1)]))
    if enumerate_columns:
        df.columns = pd.MultiIndex.from_tuples([(i, *x) if isinstance(x, tuple) else (i,x)
                                                for i, x in enumerate(df.columns, 1)])
    if not inplace:
        return df

def combo_sizes(set_list: list[set], set_names=None, vmax=None, sort=True, drop_zeros=True) -> pd.DataFrame:
    """Summarary table of set combinations sizes. Rows represent size of overlapping sets

    Args:
        set_list (list[set]): inputs sets
        set_names (_type_, optional): Names corresponding to input sets. Defaults to None.
        vmax (_type_, optional): Denominator for percentage. Defaults to # unique elements among all sets in set_list.
        sort (bool, optional): Sort overlap percentages. Defaults to True.

    Returns:
        pd.DataFrame: size of overlapping sets
    """
    if vmax is None:
        vmax = len(reduce(lambda x, y: x | y, set_list))
    if set_names is None:
        set_names = list(range(len(set_list)))
    combo_list = [()]
    sizes_list = [vmax]
    for k in range(1, len(set_list)+1):
        for indices_combo in itertools.combinations(enumerate(set_list), k):
            indices, combo = zip(*indices_combo)
            size = len(reduce(lambda x, y: x & y, combo))
            sizes_list.append(size)
            combo_list.append(indices)
    combo_df = pd.DataFrame([['Yes' if i in i_list else '-' for i in range(len(set_names))]
                             for i_list in combo_list], columns=set_names)
    combo_df['Size'] = sizes_list
    combo_df['%'] = 100*combo_df['Size'] / vmax
    if sort:
        combo_df = combo_df.sort_values('Size', ascending=False)
    if drop_zeros:
        combo_df = combo_df.query('Size > 0')
    combo_df = combo_df.reset_index(drop=True)
    combo_df.index += 1

    def highlight(s):
        return ['background-color: green' if v else '' for v in s == 'Yes']
    combo_df_styled = (combo_df
            .style.apply(highlight)
            .bar(color='#543b66', vmin=0, vmax=100, subset=['%'])
            .format(precision=1))
    return combo_df_styled

def combo_sizes2(set_list: list[set], set_names=None, vmax=None, sort=True, drop_zeros=True) -> pd.DataFrame:
    """Summarary table of set combinations sizes (strict).

    Rows represent size of overlapping sets only(which don't containing others).

    Args:
        set_list (list[set]): inputs sets
        set_names (_type_, optional): Names corresponding to input sets. Defaults to None.
        vmax (_type_, optional): Denominator for percentage. Defaults to # unique elements among all sets in set_list.
        sort (bool, optional): Sort overlap percentages. Defaults to True.

    Returns:
        pd.DataFrame: size of overlapping sets (strict)
    """
    if vmax is None:
        vmax = len(reduce(lambda x, y: x | y, set_list))
    if set_names is None:
        set_names = list(range(len(set_list)))
    combo_list = [()]
    sizes_list = [vmax]

    for k in range(1, len(set_list)+1):
        for indices, combo, other_combo in zip(
            itertools.combinations(range(len(set_list)), k),
            itertools.combinations(set_list, k),
            list(itertools.combinations(set_list, len(set_list) - k))[::-1]
        ):
            row_vals = reduce(lambda x, y: x & y, combo)
            if other_combo:
                row_vals = row_vals - reduce(lambda x, y: x | y, other_combo)
            size = len(row_vals)
            sizes_list.append(size)
            combo_list.append(indices)

    combo_df = pd.DataFrame([
            # First row is union of all values
            ['-']*len(set_names),
            # All other rows proceed normally as combinations of 'Yes', 'No'
            *[['Yes' if i in i_list else 'No' for i in range(len(set_names))]
                for i_list in combo_list[1:]]
        ], columns=set_names)
    combo_df['Size'] = sizes_list
    combo_df['%'] = 100*combo_df['Size'] / vmax
    if sort:
        combo_df = combo_df.sort_values('Size', ascending=False)
    if drop_zeros:
        combo_df = combo_df.query('Size > 0')
    combo_df = combo_df.reset_index(drop=True)
    combo_df.index += 1

    def highlight(s):
        return ['background-color: green' if v == 'Yes' else 'background-color: darkred'
                if v == 'No' else '' for v in s]
    combo_df_styled = (combo_df
            .style.apply(highlight)
            .bar(color='#543b66', vmin=0, vmax=100, subset=['%'])
            .format(precision=1))
    return combo_df_styled

# def highlight(v, color='DarkSlateGray'):
#     """Example usage: df.style.applymap(aw.highlight(1))"""
#     f = lambda x: 'background-color: DarkSlateGray' if x == v else ''
#     return f

def highlight(df: pd.DataFrame, v, color='DarkSlateGray', subset=None):
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        v (Any): _description_
        color (str, optional): _description_. Defaults to 'DarkSlateGray'.
        subset (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_

        Example usage:

        df.pipe(highlight, 1, color='k')
    """
    if hasattr(df, 'style'):
        df = df.style
    color = {'w': 'white', 'k': 'black'}.get(color, color)
    return df.applymap(lambda x: f'background-color: {color}' if x == v else '', subset=subset)

def str_contains(pd_series, *regex_str_list, **kwargs):
    '''
    Filters Pandas Series strings using regex patterns from `regex_str_list`

    Parameters
    ----------
    pat : str
        Character sequence or regular expression.
    case : bool, default True
        If True, case sensitive.
    flags : int, default 0 (no flags)
        Flags to pass through to the re module, e.g. re.IGNORECASE.
    na : scalar, optional
        Fill value for missing values. The default depends on dtype of the
        array. For object-dtype, ``numpy.nan`` is used. For ``StringDtype``,
        ``pandas.NA`` is used.
    regex : bool, default True
        If True, assumes the pat is a regular expression.

        If False, treats the pat as a literal string.
    '''
    if 'case' not in kwargs:
        kwargs['case'] = False

    match pd_series:
        case pd.Series():
            pass
        case str():
            pd_series = pd.Series([pd_series])
        case _:
            raise ValueError

    mask_list = [pd_series.str.contains(x, **kwargs) for x in regex_str_list]
    pd_series_masked = pd_series[reduce(lambda x,y: x|y, mask_list)]
    return pd_series_masked

################################################################################
# Statistical
################################################################################

# https://stackoverflow.com/questions/26102867/python-weighted-median-algorithm-with-pandas
def median(df, val, weight=None):
    if weight is None:
        return df[val].median()
    df_sorted = df.sort_values(val)
    cumsum = df_sorted[weight].cumsum()
    cutoff = df_sorted[weight].sum() / 2.
    return df_sorted[cumsum >= cutoff][val].iloc[0]

def chi2_table(table_df, prob=0.99, verbose=True):
    if not isinstance(table_df, pd.DataFrame):
        table_df = pd.DataFrame(table_df)
    stat, p_val, dof, expected = st.chi2_contingency(table_df)
    if verbose:
        expected_df = pd.DataFrame(expected, index=table_df.index, columns=table_df.columns)
        displays(
            table_df,
            expected_df,
            table_df - expected_df,
            captions=['Table', 'Expected', 'Difference']
        )
        # interpret test-statistic
        critical = st.chi2.ppf(prob, dof)
        print(f'dof={dof}, probability={prob:.3f}, critical={critical:.3f}, stat={stat:.3f}')
        if abs(stat) >= critical:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
    return p_val

def chi2_pair(treatment, control):
    assert isinstance(treatment, (list, tuple)) and len(treatment) == 2
    assert isinstance(control, (list, tuple)) and len(control) == 2
    treatment_n, treatment_size = treatment
    control_n, control_size = control
    treatment_vs_control_df = pd.DataFrame({
        'Table 1': [treatment_n, treatment_size - treatment_n],
        'Table 2': [control_n, control_size - control_n],
    })
    p_val = chi2_table(treatment_vs_control_df, verbose=False)
    return p_val

@ignore_warnings
def p_value(pd_series1: pd.Series, pd_series2: pd.Series, verbose=False, mww=False):
    if isinstance(pd_series1, pd.DataFrame):
        assert all(pd_series1.columns == pd_series2.columns)
        pval_list = [p_value(pd_series1[col], pd_series2[col], mww=mww) for col in pd_series1.columns]
        pval_df = pd.DataFrame(pval_list, index=pd_series1.columns, columns=['p-val'])
        return pval_df
    try:
        if isinstance(pd_series1, (list, tuple)) and len(pd_series1) == 2:
            p_val = chi2_pair(pd_series1, pd_series2)
            return p_val
        if pd_series1.dtype.name == 'bool':
            pd_series1 = pd_series1.astype('category')
            pd_series2 = pd_series2.astype('category')
        if pd_series1.dtype.name == 'category':
            table_df = pd.concat([
                pd_series1.value_counts(sort=False, dropna=False).rename('Table 1'),
                pd_series2.value_counts(sort=False, dropna=False).rename('Table 2'),
            ], axis=1).fillna(0)
            # Remove rows with all zeros
            table_df = table_df.loc[(table_df != 0).any(axis=1)]
            p_val = chi2_table(table_df, verbose=verbose)
            return p_val
        if pd_series1.dtype.name == 'datetime64[ns]':
            pd_series1 = pd_series1.astype(int)
            pd_series2 = pd_series2.astype(int)
        if mww:
            statistic, pvalue = st.mannwhitneyu(pd_series1.dropna(), pd_series2.dropna())
        else:
            statistic, pvalue = st.ttest_ind(pd_series1.dropna(), pd_series2.dropna(), equal_var=False)
        if verbose:
            print(f'stat={statistic:.3f}')
            # print(f'dof={dof}, probability={prob:.3f}, critical={critical:.3f}, stat={statistic:.3f}')
    except ValueError:
        return np.nan
    return pvalue

# Inspiration: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.trim_mean.html
def trim(pd_series: pd.Series, proportiontocut=0.05, conservative=True) -> pd.Series:
    """Trimmed bottom and top percentiles.

    Args:
        pd_series (pd.Series): Input values
        proportiontocut (float, optional): Proportion of elements to discard. Defaults to 0.05.
        conservative (bool, optional): Keep bordered element. Defaults to True.

    Raises:
        ValueError: Cannot trim more than 50% from bottom and top

    Returns:
        pd.Series: Trimmed values
    """
    from math import ceil
    if pd_series.size == 0:
        return pd_series

    nobs = pd_series.shape[0]
    lowercut = int(proportiontocut * nobs) if conservative else ceil(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:
        raise ValueError("Proportion too big.")

    atmp = np.partition(pd_series, (lowercut, uppercut - 1))
    return pd.Series(atmp[lowercut:uppercut])

# Source: https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.95):
    a = np.array(data, dtype=float)
    a = a[~np.isnan(a)]
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2, n-1)
    return m, m-h, m+h

# Derived from: https://cran.r-project.org/web/packages/stddiff/stddiff.pdf
def stddiff_categorical(treatment, control):
    T = pd.value_counts(treatment, sort=False, normalize=True)[1:].values
    C = pd.value_counts(control, sort=False, normalize=True)[1:].values
    assert len(T) == len(C)
    K_1 = len(T)
    if K_1 == 1:
        S = np.array([[(( T[0]*(1-T[0]) + C[0]*(1-C[0]) ) / 2)]])
    else:
        S = np.zeros([K_1, K_1])
        for kk, jj in itertools.product(range(K_1), range(K_1)):
            if kk == jj:
                S[kk, jj] = ( T[kk]*(1-T[kk]) + C[kk]*(1-C[kk]) ) / 2
            else:
                S[kk, jj] = -( T[kk]*T[jj] + C[kk]*C[jj] ) / 2

    try:
        smd = np.sqrt( (T-C).T @ np.linalg.inv(S) @ (T-C) )
    except np.linalg.LinAlgError:
        smd = np.nan
    return smd

def stddiff_numerical(treatment, control):
    if treatment.dtype.name == 'datetime64[ns]' and control.dtype.name == 'datetime64[ns]':
        treatment = treatment.astype(int)
        control = control.astype(int)
    numer = treatment.mean() - control.mean()
    denom = np.sqrt((treatment.var() + control.var()) / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        smd = numer / denom
    return smd

def stddiff_pair(treatment, control):
    assert isinstance(treatment, (list, tuple)) and len(treatment) == 2
    assert isinstance(control, (list, tuple)) and len(control) == 2
    treatment_n, treatment_size = treatment
    control_n, control_size = control
    treatment_array = np.array([[1]*treatment_n + [0]*(treatment_size - treatment_n)])
    control_array = np.array([[1]*control_n + [0]*(control_size - control_n)])
    smd = stddiff_numerical(treatment_array, control_array)
    return smd

@ignore_warnings
def stddiff(treatment, control, ci=False, k=3):
    if isinstance(treatment, pd.DataFrame):
        assert all(treatment.columns == control.columns)
        smd_list = [stddiff(treatment[col], control[col], ci=ci) for col in treatment.columns]
        smd_df_colnames = ['SMD', 'lower', 'upper'] if ci else ['SMD']
        smd_df = pd.DataFrame(smd_list, index=treatment.columns, columns=smd_df_colnames)
        return smd_df
    if isinstance(treatment, (list, tuple)) and len(treatment) == 2:
        smd = stddiff_pair(treatment, control)
    elif treatment.dtype.name == 'category':
        smd = stddiff_categorical(treatment, control)
    else:
        smd = stddiff_numerical(treatment, control)
    if k:
        smd = round(smd, k)

    if ci:
        n1, n2 = len(treatment), len(control)
        sigma = np.sqrt( (n1+n2)/(n1*n2) + (smd**2)/(2*(n1+n2)) )
        smd_l = smd - 1.96*sigma
        smd_u = smd + 1.96*sigma
        if k:
            smd_l, smd_u = round(smd_l, k), round(smd_u, k)
        return smd, smd_l, smd_u
    else:
        return smd

@ignore_warnings
def compare_dfs(df1: pd.DataFrame, df2: pd.DataFrame, cols=None, each=True, fillna=False, style=True):
    rows = {}
    if cols is None:
        assert isinstance(df1, pd.Series) and isinstance(df2, pd.Series)
        if df1.name == df2.name:
            cols = df1.name
        else:
            cols = f'{df1.name} vs {df2.name}'
        df1 = df1.to_frame(name=cols)
        df2 = df2.to_frame(name=cols)
    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        if df1[col].dtype.name == 'bool':
            rows[(col, '')] = ['\n', '']
            rows[(col, 'Yes')] = [stddiff(df1[col], df2[col]), str(p_value(df1[col], df2[col]))]
            rows[(col, 'No')] = ['\n', '']
        elif df1[col].dtype.name == 'category':
            # rows[(col, '')] = ['\n', '']
            rows[(col, '')] = [stddiff(df1[col], df2[col]), str(p_value(df1[col], df2[col]))]
            if each:
                rows.update({(col, k): [stddiff(df1[col] == k, df2[col] == k),
                                        str(p_value(df1[col] == k, df2[col] == k))]
                            for k in df1[col].cat.categories})
                rows[(col, '(nan)')] = [stddiff(df1[col].isna(), df2[col].isna()),
                                        str(p_value(df1[col].isna(), df2[col].isna()))]
            else:
                rows.update({(col, k): ['\n', ''] for k in df1[col].cat.categories})
                rows[(col, '-')] = [stddiff(df1[col], df2[col]), str(p_value(df1[col], df2[col]))]
        else:
            _smd = stddiff((sum(~np.isnan(df1[col])), df1[col].shape[0]),
                           (sum(~np.isnan(df2[col])), df2[col].shape[0]))
            _pval = p_value((sum(~np.isnan(df1[col])), df1[col].shape[0]),
                            (sum(~np.isnan(df2[col])), df2[col].shape[0]))
            rows[(col, ' (N)')] = [_smd, str(_pval)]
            if fillna:
                df1[col] = df1[col].fillna(0)
                df2[col] = df2[col].fillna(0)
            rows[(col, ' (mean, SD, 95% CI)')] = [stddiff(df1[col], df2[col]), str(p_value(df1[col], df2[col]))]
            rows[(col, ' (median, [Q1, Q3], [min, max])')] = [' ', str(p_value(df1[col], df2[col], mww=True))]
    res = pd.DataFrame(
        rows,
        index=['SMD', 'p-val']
    ).T
    if style:
        def highlight_smd(s):
            s = pd.to_numeric(s, errors='coerce')
            return ['background-color: darkslateblue' if v else ''
                    for v in (s <= -0.1) | (s > 0.1)]
        def highlight_pval(s):
            return ['background-color: brown' if v else ''
                   for v in s.astype(float) < 0.05]
        return res.style.apply(highlight_smd, subset=['SMD']).apply(highlight_pval, subset=['p-val'])
    return res

################################################################################
# Reporting results
################################################################################

def str_round(x, k=None) -> str:
    if k is False:
        return str(x)
    try:
        x = float(x)
        if x == int(x):
            return f'{int(x):,}'
        elif k is None:
            return f'{x:,}'
        else:
            return f'{round(x, k):,}'
    except OverflowError:
        return str(x)
    except ValueError:
        return str(x)
    except TypeError:
        return str(x)

def bracket_str(q1, q3, k=2, is_date=False) -> str:
    if is_date:
        # if not isinstance(k, str):
        #     k = '1s'
        # q1q3_str = f'[{q1.round(k)}, {q3.round(k)}]'
        q1q3_str = f'[{q1.date()}, {q3.date()}]'
    else:
        q1q3_str = f'[{str_round(q1, k)}, {str_round(q3, k)}]'
    return q1q3_str

def q1q3(pd_series: pd.Series, k=2, is_date=False) -> str:
    try:
        q1, q3 = pd_series.describe()[['25%', '75%']].values
    except KeyError:
        q1, q3 = pd_series.astype(float).describe()[['25%', '75%']].values
    q1q3_str = bracket_str(q1, q3, k, is_date)
    return q1q3_str

def report_categorical(pd_series: pd.Series, dropna=True, style=True) -> dict:
    """Report descriptive stats of categorical variable.

    Args:
        pd_series (pd.Series): Input values
        dropna (bool, optional): Drop missing elements. Defaults to True.
        style (bool, optional): Style output. Defaults to True.

    Returns:
        dict: Descriptive stats
    """
    if pd_series.dtype.name == 'bool':
        pd_series = pd.Categorical(pd_series, categories=[True, False]).rename_categories({True: 'Yes', False: 'No'})
    vcounts = pd.value_counts(pd_series, sort=False, dropna=dropna)
    vcount_dict = dict(zip(vcounts.index.to_list(), [[x, '', ''] for x in vcounts]))
    if not dropna:
        vcount_dict[np.nan] = vcount_dict.get(np.nan, [0, '', ''])
        vcount_dict['(N/A)'] = vcount_dict.pop(np.nan)

    if style:
        vsum = vcounts.sum()
        vcount_dict = {k: [v[0], f'({100*v[0]/vsum:.1f}%)', f'{100*v[0]/vsum:.1f}%']
                    for k, v in vcount_dict.items()}
    return vcount_dict

def report_numerical(pd_series: pd.Series, name='', k=2, proportiontocut=0, fillna=False, N=True) -> dict:
    """Report descriptive statis of numerical variable.

    Args:
        pd_series (pd.Series): Input values
        name (str, optional): Name of variable. Defaults to ''.
        k (int, optional): round to k digits. Defaults to 2.
        proportiontocut (int, optional): Trimmed propertion (from 0 to 1). Defaults to 0.
        fillna (bool, optional): Filled value. Defaults to False.
        N (bool, optional): Proportion elements complete (i.e. not missing). Defaults to True.

    Returns:
        dict: _description_
    """
    if proportiontocut > 0:
        pd_series = trim(pd_series, proportiontocut)
    report_numerical_dict = {}
    if N:
        _row0 = [pd_series.notna().sum(),
                 f'({100*pd_series.notna().mean():.1f}%)',
                 f'{100*pd_series.notna().mean():.1f}%']
        report_numerical_dict[f'{name} (N)'] = _row0
    if fillna:
        pd_series = pd_series.fillna(0)
    if isinstance(pd_series, pd.Series) and pd_series.dtype == 'datetime64[ns]':
        _row1 = [pd_series.mean().date(),
                 f'{pd_series.std().days} days',
                 '-']
        _row2 = [pd_series.median().date(),
                 q1q3(pd_series, is_date=True),
                 bracket_str(pd_series.min(), pd_series.max(), is_date=True)]
        report_numerical_dict[f'{name} (mean, SD)'] = _row1
        report_numerical_dict[f'{name} (median, [Q1, Q3], [min, max])'] = _row2
    else:
        mean, ci_left, ci_right = mean_confidence_interval(pd_series)
        _row1 = [str_round(mean, k),
                 str_round(pd_series.std(), k),
                 bracket_str(ci_left, ci_right, k=k)]
        _row2 = [str_round(pd_series.median(), k),
                 q1q3(pd_series, k=k),
                 bracket_str(pd_series.min(), pd_series.max(), k=k)]
        report_numerical_dict[f'{name} (mean, SD, 95% CI)'] = _row1
        report_numerical_dict[f'{name} (median, [Q1, Q3], [min, max])'] = _row2
    return report_numerical_dict

def report_rows(df: pd.DataFrame, cols: str | list[str]=None,
                dropna=False, k=2, proportiontocut=0, fillna=False, style=True) -> dict:
    """Report descriptive statistics as a dict.

    Args:
        df (pd.DataFrame): Input DataFrame
        cols (str | list[str], optional): column names. Defaults to None.
        dropna (bool, optional): Whether to drop missing rows. Defaults to False.
        k (int, optional): Round to k digits. Defaults to 2.
        proportiontocut (int, optional): Trimmed propertion (from 0 to 1). Defaults to 0.
        fillna (bool, optional): Filled value. Defaults to False.
        style (bool, optional): Style dataframe. Defaults to True.

    Returns:
        dict: description statistics
    """
    rows = {}
    if isinstance(df, list):
        rows_list = [report_rows(d) for d in df]
        return {k: v for x in rows_list for k, v in x.items()}
    if isinstance(df, pd.Series):
        df = df.to_frame(name='')
    if cols is None:
        cols = list(df.columns)
    for col in cols:
        if col.startswith('-'):
            rows[('-', col[1:])] = ['\n', '', '']
        elif df[col].dtype.name == 'object':
            rows[(col, '(N, uniq)')] = [df[col].notna().sum(), df[col].nunique(), '']
        elif df[col].dtype.name in ('bool', 'category'):
            rows[(col, f'{col} N (%)')] = ['\n', '', '']
            rows.update({(col, k): v for k, v in report_categorical(df[col], dropna=dropna, style=style).items()})
        else:
            # rows[(col, '-')] = ['\n', '', '']
            _items = report_numerical(df[col], k=k, proportiontocut=proportiontocut, fillna=fillna).items()
            rows.update({(col, k): v for k, v in _items})
    return rows

def report_rows_df(df: pd.DataFrame, cols: str | list[str]=None,
                   dropna=False, k=2, proportiontocut=0, fillna=False, style=True) -> pd.DataFrame:
    """Report descriptive statistics.

    Args:
        df (pd.DataFrame): Input DataFrame
        cols (str | list[str], optional): column names. Defaults to None.
        dropna (bool, optional): Whether to drop missing rows. Defaults to False.
        k (int, optional): Round to k digits. Defaults to 2.
        proportiontocut (int, optional): Trimmed propertion (from 0 to 1). Defaults to 0.
        fillna (bool, optional): Filled value. Defaults to False.
        style (bool, optional): Style dataframe. Defaults to True.

    Returns:
        pd.DataFrame: Descriptive statistics
    """
    INDEX_COLS = ['N', '%/SD/IQR', '95% CI/Range']
    res = pd.DataFrame(
        report_rows(df, cols, dropna=dropna, k=k, proportiontocut=proportiontocut, fillna=fillna, style=style),
        index=INDEX_COLS
    ).T
    if style:
        def bar_percent(x, color='#543b66'):
            if str(x).endswith('%'):
                x = float(x[:-1])
                return (f'background: linear-gradient(90deg, {color} {x}%, transparent {x}%); '
                        'width: 10em; color: rgba(0,0,0,0);')
        return res.style.applymap(bar_percent, color='steelblue', subset=['95% CI/Range'])
    return res

# Source: https://docs.python.org/3/library/itertools.html#itertools-recipes
def grouper(iterable: Iterable, n: int, fillvalue=None) -> Iterable:
    """Collect data into fixed-length chunks or blocks.

    Args:
        iterable (Iterable): Iterable to group into chunks
        n (int): size of chunks
        fillvalue (Any, optional): Missing elements to fill in. Defaults to None.

    Returns:
        Iterable: fixed-length chunks or blocks
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

@ignore_warnings
def report(func: Callable | list | pd.DataFrame, *df_args,
           compare=False, concat=False, k_list=2, fillna=False, **kwargs) -> pd.DataFrame:
    """Report descriptive statistics.

    Args:
        func (Callable): Function to map
        compare (bool, optional): Compare pairwise dataframes. Defaults to False.
        concat (bool, optional): Concatenate descriptive columns. Defaults to False.
        k_list (int, optional): Round to k digits. Defaults to 2.
        fillna (bool, optional): NA str representation. Defaults to False.

    Returns:
        pd.DataFrame: Styled DataFrame(s)
    """
    if isinstance(func, list):
        assert len(df_args) > 0
        cols = func
        def func(df):
            return report_rows_df(df, cols=cols, fillna=fillna, **kwargs)
    if not callable(func):
        df_args = [func, *df_args]
        cols = None
        def func(df):
            return report_rows_df(df, cols=cols, fillna=fillna, **kwargs)

    if compare and len(df_args) % 2 == 1:
        df_args = (*df_args[:-1], df_args[0], df_args[-1])
    if compare:
        if cols is None:
            cols = list(df_args[0].columns)
        compared_dfs = [(func(df1), func(df2), compare_dfs(df2, df1, cols, fillna=fillna))
                        for df1, df2 in grouper(df_args, 2)]
        result_dfs = list(itertools.chain(*compared_dfs))
    else:
        result_dfs = [func(df) for df in df_args]

    if k_list is None:
        k_list = [None] * len(result_dfs)
    if isinstance(k_list, int):
        k_list = [k_list] * len(result_dfs)
    k_list = [None, *k_list]

    if concat:
        result_dfs = [df.data if hasattr(df, 'data') else df for df in result_dfs]
        # result_dfs = [df.applymap(lambda x: str_round(x, k=k)) for df, k in zip(result_dfs, k_list[1:])]
        ## Ensure only one new-line per empty row
        result_dfs = [result_dfs[0]] + [df.replace('\n', '') for df in result_dfs[1:]]
        result_dfs = pd.concat(result_dfs, axis=1).pipe(df_enumerate)
        result_dfs.pipe(df_index, verbose=True, k=k_list)
    else:
        index_df = result_dfs[0].pipe(df_index)
        displays(index_df, *[df.style.hide() if hasattr(df, 'style') else df.hide()
                             for df in result_dfs], k=k_list)
