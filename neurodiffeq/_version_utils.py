import warnings
import functools


def warn_deprecate_class(new_class):
    """get a new class name that issues a warning when instantiated
    :param new_class: new class name
    :type new_class: type
    :return: a function that, when called, acts as if it is a class constructor
    :rtype: callable
    """

    @functools.wraps(new_class)
    def old_class_getter(*args, **kwargs):
        pass

    return old_class_getter


def deprecated_alias(**aliases):
    """A decorator to deprecate old argument names in favor of new ones.
    See more here https://stackoverflow.com/a/49802489.

    :param aliases: A sequence of keyword argument of the form: old_name="name_name"
    :param aliases: Dict[str,str]
    :return: A decorated function that can receive either `old_name` or `new_name` as input
    :rtype: function
    """

    def deco(f):
        pass

    return deco


def _rename_kwargs(func_name, kwargs, aliases):
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise KeyError(f'{func_name} received both `{alias}` (deprecated) and `{new}` (recommended)')
            warnings.warn(f'The argument `{alias}` is deprecated for {func_name}; use `{new}` instead.', FutureWarning)
            kwargs[new] = kwargs.pop(alias)
