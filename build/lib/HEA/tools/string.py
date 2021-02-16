"""
Anthony Correia
08/02/21
Formatting of string
"""

def _remove_latex(text):
    """ Turn latex text into normal text
    Parameters
    ----------
    text: str
    
    Return
    ------
    str
    """
    if text is not None:
        return text.replace('\_','_').replace('$','')
    else:
        return None

def _remove_space(text):
    """ Replace the space of a string into underscores
    Parameters
    ----------
    text: str
    
    Returns
    -------
    str
    """
    if text is not None:
        return text.replace(' ','_')
    else:
        return None

def list_into_string(L, sep='_'):
    """Transform a list into a string of elements seperated by sep
    Parameters
    ----------
    L    : List
    sep  : Str
    
    Returns
    -------
    str 
        String with all the elements separated by the argument `sep`    
    """
    if L is None:
        return None
    elif not isinstance(L, str):
        return sep.join(L)
    else:
        return L
    
def _latex_format(text):
    """Replace `_` by `\_` in a string to avoid latex errors with matplotlib"""
    return text.replace('_','\_') # to avoid latex errors when I want a label like 'B0_M'


def string_between_brackets(text, bracket=None): # previously: redefine_format_text
    """ Return the correct text for the label of plots    
    
    Parameters
    ----------
    text    : str
    bracket : 
        string of bracket around text
        - '('  : parenthetis
        - '['  : bracket
        - None : no brackets
    
    Returns
    -------
    str
        string with
                * a space before it
                * between the brackets specified by 'bracket'
    """
    
    if (text is not None) and (text != ''):
        if bracket is None:
            brackets = ['', '']
        elif bracket == '(':
            brackets = ['(', ')']
        elif bracket == '[':
            brackets = ['[', ']']
        
        text = " %s%s%s"%(brackets[0], text, brackets[1])
    else:
        text = ""
    
    return text


def add_text(text1, text2, sep = ' ', default=None):
    """ concatenate 2 texts with sep between them, unless one of them is None
    
    Parameters
    ----------
    text1       : str
    text2       : str
        text to add
    sep         : str
        separator between text and text_to_add
    default     : str or None
        default value to return if both text1 and text2 are None
    
    Returns
    -------
    str
        `{text1}_{text2}` if they are both not None
                        else, return text1, or text2, 
                        or the default value is they are both None
    """
    
    if text1 is None and text2 is None:
        return default
    elif text1 is None:
        return text2
    elif text2 is None:
        return text1
    else:
        return text1 + sep + text2

