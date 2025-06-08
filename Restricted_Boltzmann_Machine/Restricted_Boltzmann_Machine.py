import copy
import numpy as np
import opt_einsum as oe

from . import Functions as F

class RBM:
    def __init__(
            self, 
            visible_size    : int, 
            hidden_size     : int, 
            inverse_temp    : float = 1,
            k               : int   = 1,
            seed            : int   = None,
            dtype           : type  = np.float32,
            weight_factor   : float = 1.0
            ):
        
        self.vs = visible_size
        self.hs = hidden_size 
        self.beta = inverse_temp
        self.k = k

        self.dtype = dtype
        self.__float_max = np.finfo(dtype).max

        # v^T @ W @ h + v^T @ θv + θh @ h
        #self.__rs = np.random.RandomState(seed)
        np.random.seed(seed)
        self.Weight = np.random.uniform(
            low   = - 0.1 * np.sqrt(6. / (self.vs + self.hs)),
            high  =   0.1 * np.sqrt(6. / (self.vs + self.hs)),
            size  = (self.vs, self.hs), 
            ).astype(self.dtype) * weight_factor
        self.bias_v = np.zeros(self.vs, dtype = self.dtype)
        self.bias_h = np.zeros(self.hs, dtype = self.dtype)

    def pseudo_likelihood(self, v):
        ind = (np.arange(v.shape[0]), np.random.randint(0, v.shape[1], v.shape[0]))
        v_noise = copy.deepcopy(v)
        v_noise[ind] = 1 - v_noise[ind]
        f_model = self.free_energy(v)
        f_noise = self.free_energy(v_noise)

        #pseudo_likelihood = -np.log(1 + np.exp(-(f_model - f_noise)))
        return -np.logaddexp(0, -(f_model - f_noise))

    def free_energy(self, v : np.ndarray):
        """
        Return
        ------
        Free energy for each batch.
        """
        external_field_term = oe.contract("bi,i->b", v, self.bias_v)

        batch_size, _ = v.shape
        _bias_h_padded = np.pad(
            array       = self.bias_h.reshape((1, self.hs)), 
            pad_width   = ((0, batch_size - 1), (0, 0)) 
            )
        
        _log = oe.contract("bi,ij->bj", v, self.Weight) - _bias_h_padded
        _log_max = np.max(_log)

        if (_log_max < np.log(self.__float_max) * 0.8):
            _effective_field = np.log(1 / np.exp(_log_max) + np.exp(_log - _log_max)) + _log_max
        else:
            _effective_field = _log
        effective_field_term = np.sum(_effective_field, axis = 1, dtype = self.dtype)

        f = - external_field_term - effective_field_term

        return np.average(f)
    

    def __visible_to_hidden(self, v : np.ndarray):
        p_hv = F.sigmoid(
            self.beta * (oe.contract("bi,ij->bj", v, self.Weight) + self.bias_h)
        )
        h = F.binormal_distribution(p_hv)
        return h, p_hv
    
    def __hidden_to_visible(self, h : np.ndarray):
        p_vh = F.sigmoid(
            self.beta * (oe.contract("ij,bj->bi", self.Weight, h) + self.bias_v)
        )
        v = F.binormal_distribution(p_vh)
        return v, p_vh
    
    def forward(self, v : np.ndarray):
        """
        Parameters
        ----------
        v : Input vectors with a size of (batch_size, input_size)
        k : Iteration of Contrastive-Divergence. Original RBM is k=1.

        Retrun
        ------
        `v_model`, `h_model`, `h_data`, `p_hv_model`, `p_vh_model`, `p_hv_data`
        """

        h_data,  p_hv_data = self.__visible_to_hidden(v)
        h_model = copy.deepcopy(h_data)
        for _ in range(self.k):
            v_model, p_vh_model = self.__hidden_to_visible(h_model)
            h_model, p_hv_model = self.__visible_to_hidden(v_model)
        
        return v_model, h_model, h_data, p_hv_model, p_vh_model, p_hv_data
    
    def backward(
            self, 
            v_data          : np.ndarray, 
            h_data          : np.ndarray, 
            v_model         : np.ndarray | None = None, 
            h_model         : np.ndarray | None = None,
            p_hv_data       : np.ndarray | None = None,
            p_hv_model      : np.ndarray | None = None,
            learning_rate   : float             = 0.001,
            algorithm       : str               = "CD"
        ):

        batch_size, _ = v_data.shape
        #Do the same calculation as folling: 
        #dw_{bij} = <v_{bi}⊗h_{bj}>_data - <v'_{bi}⊗h'_{bj}>_model, then calculate dw_{ij} = 1 / batchsize * sum_b dw_{bij} 
        if algorithm == "CD":
            dw = oe.contract("bi,bj->ij", v_data, p_hv_data) - oe.contract("bi,bj->ij", v_model, p_hv_model)
            dbias_h = p_hv_data - p_hv_model
        elif algorithm == "PCD":
            dw = oe.contract("bi,bj->ij", v_data, h_data) - oe.contract("bi,bj->ij", v_model, h_model)
            dbias_h = h_data - h_model
        dbias_v = v_data - v_model
        
        self.Weight += learning_rate * dw / batch_size
        self.bias_v += learning_rate * np.average(dbias_v, axis = 0)
        self.bias_h += learning_rate * np.average(dbias_h, axis = 0)

