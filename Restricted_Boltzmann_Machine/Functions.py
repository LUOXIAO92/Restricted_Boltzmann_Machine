import cupy as np
from itertools import product

def softmax(x : np.ndarray, axis : int = 0, dtype : type = np.float32):
    _x = x - max_values(x, axis = -1)
    expx = np.exp(_x)
    sf = expx / np.sum(expx, axis = axis, keepdims = True, dtype = dtype)
    return sf

def max_values(a : np.ndarray, axis : int):
    _axis = a.ndim + axis if axis < 0 else axis

    maxval = a.max(axis = _axis, keepdims = True)
    pad_width = tuple([(0, a.shape[dim] - 1) if dim == _axis else (0, 0) for dim in range(a.ndim)])
    maxval = np.pad(maxval, pad_width, mode = "constant", constant_values = 0)

    idx0 = tuple([0 if dim == _axis else slice(0, a.shape[dim]) for dim in range(a.ndim)])
    for i in range(a.shape[_axis]):
        idxi = tuple([i if dim == _axis else slice(0, a.shape[dim]) for dim in range(a.ndim)])
        maxval[idxi] = maxval[idx0]
    
    return maxval

def argmax_mask(a : np.ndarray, axis : int):
    """
    Get mask of indices of max values along specific axis.

    Exapmle
    -------
    >>> a = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    >>> a
    >>> array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],
       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]])
    >>> argmax(a, axis = 2)
    >>> array([[[False, False, False,  True],
        [False, False, False,  True],
        [False, False, False,  True]],
       [[False, False, False,  True],
        [False, False, False,  True],
        [False, False, False,  True]]])
    >>> max(a, axis = 1)
    """

    maxval = max_values(a, axis)

    mask = (np.abs((a - maxval) / maxval) < 1e-13)

    return mask

def get_most_likely_state(states : np.ndarray, probabilities : np.ndarray, rs : np.random.RandomState):
    """
    For a given states and correspond probabilities with a size of states.shape[-1] == probabilities.shape[-1] == num_of_states, and select the most likely states.
    `states` and `probabilities` must have the same shape.

    If the inputs are states and probabilities with batch size and/or input data size, the axis of states must be at the end.

    Retrun
    ------
    state : Most likely state

    Example
    -------
    >>> states = [0, 1, 2, 3]
    >>> probabilities = [0.01, 0.01, 0.97, 0.01]
    >>> get_most_likely_state(states, probabilities)
    >>> array([2])

    States and probabilities with batch_size = 2, input_size = 3
    >>> states = [[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], 
                  [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]
    >>> probabilities = [[[0.01, 0.01, 0.97, 0.01], [0.01, 0.96, 0.92, 0.01], [0.01, 0.06, 0.33, 0.60]], 
                         [[0.80, 0.18, 0.01, 0.01], [0.12, 0.13, 0.14, 0.61], [0.01, 0.20, 0.33, 0.46]]]
    >>> get_most_likely_state(states, probabilities)
    >>> [[2, 1, 3], [0, 3, 3]]
    """

    #Calulate mask
    mask  = argmax_mask(probabilities, -1)

    #Force the mask of state to be one-hot
    shape = states.shape
    count_true = mask[mask].size
    if count_true != shape[0] * shape[1]:
        duplicate_list = []
        for i in range(shape[2]):
            for j in range(i + 1, shape[2]):
                location_coli_eq_colj = np.where(mask[:,:,i] == mask[:,:,j])
                if location_coli_eq_colj[0].size != 0:
                    duplicate_list.append([i,j])
        
        for i, j in duplicate_list:
            for k, l in zip(*np.where(mask[:,:,i] == mask[:,:,j])):
                randIdx = rs.randint(0, shape[2])
                state_mask = np.full(shape = shape[2], fill_value = False, dtype = bool)
                state_mask[randIdx] = True
                mask[k, l] = state_mask


    state = states[mask]
    merge_dim = []
    for dim in range(states.ndim):
        if dim < (states.ndim - 1):
            merge_dim.append(states.shape[dim])

    state = state.reshape(tuple(merge_dim))
    
    return state

