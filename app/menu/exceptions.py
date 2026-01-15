# app/menu/exceptions.py

class MenuLoadError(Exception):
    """
    Raised when the menu or entity index fails to load or validate.
    This is a startup / configuration error, not a user error.
    """
    pass
