

def get_current_values(pars):
    """ Takes a dictionary of parameter (key: [panel Input, function, value]) pairs
    and returns dictionary of key: value pairs, retrieving the value from the Input
    object or calling the function if necessary"""
    new_pars = {}
    for k, v in pars.items():
        if hasattr(v, 'value'):
            # if it is an input, get the value
            new_pars[k] = v.value
        elif callable(v):
            # if it is a function, call it to get the value
            new_pars[k] = v()
        else:
            # otherwise assign directly
            new_pars[k] = v
    return new_pars