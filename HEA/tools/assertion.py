"""
**Module with functions to assert a statement**:

* assert that a python object is a list or tuple
* some assertion function (used internally)
"""

from inspect import signature


def is_list_tuple(L):
    """ Check that ``L`` is a ``list`` or a ``tuple``

    Parameters
    ----------
    L : python object
        potential ``list`` or ``tuple``

    Returns
    -------
    bool:
        ``True`` if ``L`` is a ``list`` or a ``tuple``, else ``False``
    """
    return (isinstance(L, tuple)) or (isinstance(L, list))


def _assert_list_tuple(L):
    """ Assert that ``L`` is a ``list`` or a ``tuple``

    Parameters
    ----------
    L : python object
        potential ``list`` or ``tuple``
    """
    assert is_list_tuple(L), f'{L} is neither a list nor a tuple'


def _assert_number_params_function(function, n):
    """ Assert that the number of parameters of a function is equal to some number

    Parameters
    ----------
    function: func or None
        function
    n: int
        number of expected arguments
    """

    if function is None:
        assert n == 1, f"The number of parameters of the identity is 1 but the expected number of parameters is {n}"

    sig = signature(function)
    number_params_function = len(sig.parameters)

    assert number_params_function == n, f"The number of parameters of the function {function} is {number_params_function}, which is different from the expected number {n}"


def _assert_same_length(*args):
    """ Assert that all the iterables provided have the same length

    Parameters
    ----------
    *args: list of iterables
    """

    if not args:
        return
    else:
        common_length = len(args[0])

    for arg in args[1:]:
        assert len(
            arg) == common_length, f"The length of {arg} is different from the other lengths of the list \n {args}"


def _assert_not_none(e):
    """ Assert that a python object is not ``None``

    Parameters
    ----------
    e: python object
    """

    assert e is not None, f"{e} is None"


def _assert_is_in_iterable(e, L):
    """ Assert that an element is in an iterable

    Parameters
    ----------
    e: python object
    L: iterable
    """

    assert e in L, f"{e} is not in {L}"
