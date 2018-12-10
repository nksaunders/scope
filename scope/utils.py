__all__ = ['ScopeError', 'ScopeWarning']

class ScopeError(Exception):
    """Exception raised for scope-specific errors."""
    pass

class ScopeWarning(Warning):
    """Log warnings for scope."""
    pass
