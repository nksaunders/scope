__all__ = ['ScopeError', 'ScopeWarning']

class ScopeError(Exception):
    """Exception raised for scope-specific errors."""
    pass

class ScopeWarning(Warning):
    """Log warnings for scope."""
    pass

def _interpolate_nans(y):
    """Helper to handle indices and logical indices of NaNs.

    Parameters
    ----------
    `y` :
        1d numpy array with possible NaNs

    Returns
    -------
    `nans` :
        logical indices of NaNs
    `index` :
        a function, with signature indices= index(logical_indices),
        to convert logical indices of NaNs to 'equivalent' indices

    Example
    -------
    >>> # linear interpolation of NaNs
    >>> nans, x= nan_helper(y)
    >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    nans = np.isnan(y)
    x = lambda z: z.nonzero()[0]

    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y
