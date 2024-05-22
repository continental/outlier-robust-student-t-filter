# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Abstract classes :class:`AbstractDynamicModel` and :class:`AbstractFilter`. One the one hand, they
separate the functionality of forecasting model behaviour and adapting state estimates to observed 
measurements. On the other hand, they predefine general functionality so that only the key features
of each filtering method have to be implemented.
"""

from __future__ import annotations
import time
from typing import Dict
import numpy
from .. import utils


print("-   Loading File 'abstract.py'")

class AbstractDistribution():
    """
    Defines the expected properties of multivariate distributions in this framework. These are 
    originating from the subclasses of ``scipy.stats.multi_rv_frozen``. However, scipy lacks the 
    abstraction level for more complicated distributions like mixture- or joint-distributions, 
    hence, this class.

    In particular, for the efficiency of logging hundreds of past estimated state distributions, we
    choose to implement the distributions so that a stack of different distributions can be 
    represented by stacks of their parameters.

    """

    n: int                            #: The dimension of the multivariate distribution
    k: int                            #: The number of stacked distributions
    _params: Dict[str, numpy.ndarray] #: The internal parameters of the distribution
    _init_seed: int                   #: The seed which initially constructed the RNG
    _rng_gen: numpy.random.Generator  #: The random number generator of the distribution
    
    @staticmethod
    def create_params(**kwargs) -> Dict[str, numpy.ndarray]:
        """
        Takes the same arguments as :meth:`~src.filters.abstract.AbstractDistribution.__init__` 
        (except ``seed``, ``peek`` and ``params``), 
        but instead of creating a Distribution object, it returns a dict of :class:`numpy.ndarray`
        parameters. These parameters in array form can then be stacked, and such a stack can be
        used to create a stack of distributions by calling 
        :meth:`~src.filters.abstract.AbstractDistribution.__init__` with the list of these stacks 
        in the ``params`` key. Note that the parameters will be stacked along the first axis.

        :returns: a dictionary of array-ised parameters.

        """
        raise NotImplementedError("'create_params' not implemented in abstract class AbstractDistribution")

    def __init__(self, seed: int | numpy.random.Generator | None = None, peek: bool = False, 
                       params: Dict[str, numpy.ndarray] = None, **kwargs) -> None:
        """
        Creating the distribution

        :param seed: the seed for the RNG, can be an int, or a separate RNG which is used to generate
                the seed, defaults to None for a random seed
        :param peek: If a separate RNG is provided peek determines if the generation of the seed 
                influences the state of the reference RNG, defaults to False. Note that if 
                ``peek = False`` and 
        :param params: A stack of parameters for multiple distributions given as dictionary, 
                other arguments will be ignored if this is not ``None``.
        """
        # if a generator object is given as seed the default_rng method just returns this object
        # However, since we don't want the RNG of the distribution to interfere with the creating 
        # RNG, we need to create a seed reproducibly from the parent generator without interference
        if peek and isinstance(seed, numpy.random.Generator):
            tmp_gen = numpy.random.default_rng()
            tmp_gen.__setstate__(seed.__getstate__())
            seed = tmp_gen.integers(2147483647)
        else:
            seed = numpy.random.default_rng(seed).integers(2147483647)
        self._init_seed = seed
        self._rng_gen = numpy.random.default_rng( seed )
        if params is None:
            params = self.__class__.create_params(**kwargs)
        self._params = params
        self.k = next(iter(params.values())).shape[0]

    def stack(self, other_distr: AbstractDistribution):
        """
        Stacks the parameter list of the other distribution on top of the own parameter list.
        This can be used to, for example, stack the distribution of a random process at two diffrent
        time steps. The two distributions have to have the same dimension :attr:`n` and the same 
        parameter names :attr:`_params`.

        :param other_distr: The other distribution of the same class

        """
        assert self._params.keys() == other_distr._params.keys(), "Needs to have same parameters"
        assert self.n == other_distr.n, "Needs to have same dimensions"
        other_params = other_distr._params
        new_params = {name: numpy.concatenate((arr, other_params[name])) for name, arr in self._params.items()}
        self._params = new_params
        self.k = self.k + other_distr.k

    def marginal(self, comp):
        r"""
        Returns the marginal distribution of the given components.

        For example if we have a multivariate normal of 4 components 

        .. math::
            \mathcal{N}\left(
                \left( \begin{array}{c} \!\mu_1\\ \!\mu_2\\ \!\mu_3\\ \!\mu_4\end{array} \right), 
                \left( \begin{array}{cccc} 
                    \!\Sigma_{11}& \Sigma_{12}& \Sigma_{13}& \Sigma_{14}\\
                    \!\Sigma_{12}& \Sigma_{22}& \Sigma_{23}& \Sigma_{24}\\
                    \!\Sigma_{13}& \Sigma_{23}& \Sigma_{33}& \Sigma_{34}\\
                    \!\Sigma_{14}& \Sigma_{24}& \Sigma_{34}& \Sigma_{44}\\
                \end{array} \right)
            \right) 
    
        then the marginal distribution of components :math:`[2,4]` is 

        .. math::
            \mathcal{N}\left( 
                \left( \begin{array}{c} \!\mu_2\\ \!\mu_4\end{array} \right), 
                \left( \begin{array}{cccc} 
                    \!\Sigma_{22}& \Sigma_{24}\\
                    \!\Sigma_{24}& \Sigma_{44}\\
                \end{array} \right)
                \right) 
    
        :param comp: The components of the marginal distribution. Needs to be sorted in ascending order.

        """
        raise NotImplementedError("'marginal' not implemented in abstract class AbstractDistribution")

    def mean(self):
        """
        Returns the mean of the distribution.

        :returns: the (:math:`n`,) dimensional mean vector
        :rtype: numpy.ndarray 

        """
        raise NotImplementedError()

    def cov(self):
        """
        Returns the covariance matrix of the distribution.
        :returns: the (:math:`n`,:math:`n`) dimensional covariance matrix
        :rtype: numpy.ndarray 

        """
        raise NotImplementedError()

    def pdf(self, x):
        """
        Gives the probability density function evaluated at :math:`x`.

        :param x: The evaluation point of the density. Either of shape :math:`(n,)` or :math:`(n,k)` if the the 
                density should be evaluated at multiple :math:`(k)` points.
        :type x: numpy.ndarray

        """
        raise NotImplementedError()
    
    def logpdf(self, x):
        """
        Returns the logarithm of the :attr:`pdf`.

        :param x: The evaluation point of the density. Either of shape :math:`(n,)` or :math:`(n,k)` if the the 
                density should be evaluated at multiple :math:`(k)` points.
        :type x: numpy.ndarray

        """
        raise NotImplementedError()
    
    def rvs(self, m=1):
        """
        Samples :math:`m` random variables according to the distribution

        :param m: Number of generated samples
        :type m: int
        
        """
        raise NotImplementedError()



class AbstractDynamicModel:
    r"""
    Abstraction for the dynamic model underlying the filters. The dynamic model should be a 
    Hidden Markov Model (HMM) with additive noise following in some form the system 
    dynamics

    .. math:: 
        x_t &= F(x_{t-1}) + v_t, &\quad v_t \sim \mathcal{N}(0, Q)\\
        y_t &= H(x_t) + e_t,     &\quad e_t \sim \mathcal{N}(0, R).

    The class provides the linearised model around given state hypotheses, see method 
    :meth:`~src.filters.abstract.AbstractDynamicModel.forecast`. Note that we explicitly assume 
    Gaussian distributions, since this is the 
    standard assumption. If the HMM is instead based on non-Gaussian distributions, approximations
    via Gaussian distributions should still be provided so that every method can work with the model.
    More specific model assumptions should be filter-specific.

    Note that this is a purely abstract class and cannot be used directly.

    The intended extensions :class:`~src.filters.models.LinearDynamicModel` (and 
    :class:`~src.filters.models.UnscentedNonLinearDynamicModel` not yet)
    are already implemented, but extensions like, e.g., the extended Kalman Filter 
    or particle filters are also possible.
    """
    
    state_dim: int = None  #: the dimension of the states :math:`n`
    obs_dim:   int = None  #: the dimension of the observations :math:`m`
    _init_seed     = None  #: The initial seed with which :attr:`_rng_gen` was initiated
    #: RNG generator which should be used for generating all internal random processes.
    _rng_gen: numpy.random.Generator

    def __init__(self, F, Q, H, R, seed: int = None, state_dim: int = None, obs_dim: int = None) -> None:
        r"""
        Initialises the model

        :param F: The transition model. Given a state estimation :math:`x_t` at time step :math:`t`,
                :math:`F(x_t) = x_{t+1}` should return the state estimation at the next time step.
        :param Q: The model of the covariance matrix of the process noise :math:`v_t`.
        :param H: The deterministic observation model describing how a observation :math:`y_t` is 
                generated from a state :math:`x_t`.
        :param R: The model of the covariance matrix of the observation noise :math:`e_t`.
        :param seed: A seed to instantiate the random number generator of the model, if None a 
                random seed is used.
        :param state_dim: The dimension of the state vectors :math:`n`, i.e. ``Q`` should produce 
                :math:`n \times n` covariance matrices. If None extracted from other arguments.
        :type state_dim: int
        :param obs_dim: The dimension of the observation vectors :math:`m`, i.e., ``R`` should 
                produce :math:`m \times m` covariance matrices. If None extracted from other arguments.
        :type obs_dim: int
        """
        self._F = F
        self._Q = Q
        self._H = H
        self._R = R
        self._init_seed = numpy.random.default_rng(seed).integers(2147483647)
        self._rng_gen   = numpy.random.default_rng(self._init_seed)
        self.state_dim  = state_dim
        self.obs_dim    = obs_dim
    
    def forecast(self, mu: numpy.ndarray = None, P: numpy.ndarray = None, next_time: float = None, 
                 dt:float=None) -> tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                                         numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray,
                                         numpy.ndarray, numpy.ndarray, numpy.ndarray):
        r"""
        Predicts model behaviour at given state estimates.
        
        Given state estimates as Gaussian distributions, give estimates to the states at the 
        next timestep, their covariance matrices, the 
        expected observations and their covariance matrices, as well as the covariance matrices of 
        the additive process- and observation-noises. 
        In the following :math:`n = {}`:attr:`state_dim`, :math:`m = {}`:attr:`obs_dim` and :math:`k` is
        the number of points the model should be evaluated at

        :param mu: Means of current state estimates, of shape :math:`(n,), (n,1), (k,n)` or or :math:`(k,n,1)`.
                In the later case, we also need a stack of :math:`k` covariance matrices.
        :type mu: numpy.ndarray
        :param P: Covariance matrices of the current state, either a matrix of shape 
                :math:`(n,n)` or a stack of matrices of shape :math:`(k,n,n)`.
        :type P: numpy.ndarray
        :param time: The current time, depending on whether the model requires this parameter.
        :type time: float
        :param dt: The time difference between the current :obj:`time` and next time step :obj:`time + dt`,
                depending on whether the model requires this parameter.
        :type dt: float
        :raises NotImplementedError: The function is not implemented in the abstract class 
                AbstractDynamicModel.
        :returns: E.g., for a linear model, a tuple 
        
                .. math::
                    (F \mu, F P F^\top, Q, HF \mu, H F P F^\top H^\top, H Q H^\top, R, 
                        F P F^\top H^\top, Q H^\top, PF, PFH)
                
                of next estimated transitioned state means :math:`F \mu`, estimated transitioned
                state covariances :math:`F P F^\top`, process noise covariance matrix :math:`Q`,
                expected observations :math:`HF \mu`, expected observation noise covariance
                :math:`H F P F^\top H^\top` induced by the state uncertainty, expected observation
                noise covariance :math:`H Q H^\top` induced by the process noise, pure observation
                noise covariance matrix :math:`R`, cross correlation between next state estimate and
                observation :math:`F P F^\top H^\top`, as well as cross-correlation between process
                noise and observation :math:`Q H^\top`, between current and next state :math:`PF`, 
                and between current state and observation :math:`PFH`.
        :rytpe: list of numpy.ndarray of shapes: :math:`(\,(k,n,1), (k,n,n), (k,n,n), (k,m,1), 
                (k,m,m),` :math:`(k,m,m), (k,m,m), (k,n,m), (k,n,m), (k,n,n), (k,n,m) \,)`
        """
        raise NotImplementedError("function 'forecast' is not implemented in the abstract class"+
                                  "AbstractDynamicModel")
    
    def forecast_smoother(self, mu: numpy.ndarray = None, P: numpy.ndarray = None, next_time: float = None, 
                 dt:float=None) -> tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):
        r"""
        Predicts model behaviour at given state estimates, while not considering measurements.
        
        Given state estimates as Gaussian distributions with mean ``mu`` and covariance ``P``, 
        give estimates to the states at the next timestep, their covariance matrices and cross-correlation.
        In the following :math:`n = {}`:attr:`state_dim`, :math:`m = {}`:attr:`obs_dim` and :math:`k` is
        the number of points the model should be evaluated at

        :param mu: A Mean of the current state estimate, of shape :math:`(n,), (n,1)`
        :type mu: numpy.ndarray
        :param P: A covariance matrix of the current state, a matrix of shape :math:`(n,n)`.
        :type P: numpy.ndarray
        :param time: The current time, depending on whether the model requires this parameter.
        :type time: float
        :param dt: The time difference between the current :obj:`time` and next time step :obj:`time + dt`,
                depending on whether the model requires this parameter.
        :type dt: float
        :raises NotImplementedError: The function is not implemented in the abstract class 
                AbstractDynamicModel.
        :returns: E.g. for a linear model, a tuple 
        
                .. math::
                    (F \mu, F P F^\top, Q, P F^\top)
                
                of next estimated transitioned state means :math:`F \mu`, estimated transitioned 
                state covariance :math:`F P F^\top`, process noise covariance :math:`Q` and estimated 
                transitioned cross correlation between previous and current state :math:`P F^\top`.
        :rytpe: list of numpy.ndarray of shapes: :math:`(\,(n,1), (n,n), (n,n), (n,n) \,)`
        """
        raise NotImplementedError("function 'forecast' is not implemented in the abstract class"+
                                  "AbstractDynamicModel")
    
    def sample(self, x: numpy.ndarray, t0 : float, dt: float | numpy.ndarray, give_noise: bool = False
               ) -> tuple(numpy.ndarray, numpy.ndarray) | tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):
        r"""
        Samples possible next states.

        :param x: current state estimates, either a single array of shape (:attr:`state_dim`,) 
                or (:attr:`state_dim`, :math:`k`) for :math:`k` consecutive positions.
        :type x: numpy.ndarray
        :param t0: The initial time.
        :type t0: float
        :param dt: The time differences between the sample timesteps.
        :type dt: float or a numpy.ndarray
        :param give_noise: If ``True`` also returns the sampled noise :math:`\tilde v_{t+1}` and :math:`\tilde e_{t+1}`.
                Defaults to ``False``.
        :type give_noise: bool
        :raises NotImplementedError: The function is not implemented in the abstract class 
                AbstractDynamicModel.
        :returns:  If ``give_noise = False`` sampled possible next states and observations :math:`(x_{t+1}, \tilde y_{t+1})`,
                more precisely :math:`(F(\nu_{t}+\tilde v_{t+1}), H(F(\nu_{t}+\tilde v_{t+1}) +\tilde e_t)`.
                or a tuple :math:`([x_{t+1}, ..., x_{t+k}], [\tilde y_{t+1}, ..., \tilde y_{t+k}])`,
                if an array of time differences or starting positions is given.

                If ``give_noise = True`` returns :math:`(x_{t+1}, \tilde y_{t+1}, \tilde v_{t+1}, \tilde e_{t+1})`,
                or the corresponding tuple of arrays.
        """
        raise NotImplementedError("function 'sample' is not implemented in the abstract class"+
                                  "AbstractDynamicModel")
    
    def reset(self) -> None:
        """
        Resets the model to the state at initial construction. Usually only involves resetting the
        RNG. However, if a subclass implements side effects or additional states, these should also be 
        resetted.
        """
        init_gen = numpy.random.default_rng(self._init_seed)
        self._rng_gen.__setstate__(init_gen.__getstate__())

    def __copy__(self) -> AbstractDynamicModel:
        """
        Returns an independent but identical copy of the model.
        """
        new_model = self.__class__.__new__(self.__class__)
        new_model.__dict__.update(self.__dict__)
        new_model._rng_gen.__setstate__(self._rng_gen.__getstate__())
        return new_model



class AbstractFilter:
    """
    Abstract Class for all following implementations of Bayesian Filters. 
    
    Bayesian Filters here describe algorithms that reconstruct the timeseries :math:`x_t` of a 
    dynamic state only using noisy observations of these states :math:`y_t` and an internal model, 
    that describes how :math:`x_t` evolves over time and how :math:`y_t` is produced from :math:`x_t`.

    The internal model will be handled by an :class:`~src.filters.abstract.AbstractDynamicModel` instance and the actual
    filtering mechanism of the Filters should be implemented by overriding :meth:`~src.filters.abstract.AbstractFilter._filter`.
    """

    _state_mean  = None #: The estimated mean of the true state
    _state_covar = None #: The estimated covariance matrix of the true state
    _state_time  = None #: The current time step of the filter
    _model       = None #: The dynamic model, an instance of :class:`~src.filters.abstract.AbstractDynamicModel`

    _state_dim = None   #: The dimension of the filtered state 
    _obs_dim   = None   #: The dimension of the observations
    
    #: The history of the model, where for each past timestep, a number of parameters are documented.
    #: By default, the history tracks the past timesteps ``times``, the estimated state 
    #: distributions as ``estimates``, the processing time needed in the model 
    #: :meth:`~src.filters.abstract.AbstractDynamicModel.forecast` as ``comp_time_model`` and in the
    #: pure :meth:`~src.filters.abstract.AbstractFilter._filter` as ``comp_time_filter`` and 
    #: for the rest as ``comp_time_rest``.
    #: Also note that the initial state also has to be documented
    #:
    #: If further parameters of the :meth:`~src.filters.abstract.AbstractFilter._filter` method 
    #: should be documented the corresponding keys have to be created during creation, initiated 
    #: with zero arrays of fitting size as values (with the first axis iterating over timesteps).
    #: Eg. if a scalar parameter ``eta`` should be tracked too, the ``__init__`` method of the
    #: subclass should contain the line ``self.hist["eta"] = np.zeros((1,1))``. And if three
    #: boolean values should be tracked in each timestep, the line should be
    #: ``self.hist["checks"] = np.zeros((1,3), dtype=bool)``.
    _hist: dict  = {}

    _init_seed   = None   #: The initial seed with which _rng_gen was initiated
    #: RNG generator, which should be used for generating all internal random processes.
    _rng_gen: numpy.random.Generator
    #: The class of the state estimate distribution, has to be defined by subclasses
    _state_distr_class : AbstractDistribution

    #: For optimisation of the method, will multiply with Q
    _process_scalar = 1.
    #: For optimisation of the method, will multiply with R
    _obs_scalar   = 1.

    def __init__(self,model: AbstractDynamicModel, mean: numpy.ndarray,
                      covar: numpy.ndarray, current_time: float = None, seed: int = None,
                      exp_hist_len: int = 100, process_scalar: float = 1., obs_scalar: float = 1.) -> None:
        """
        Constructor of the Filter.
        
        :param model: The model describing the evolution of states and observations.
        :type model: an instance of a subclass of :class:`~src.filters.abstract.AbstractDynamicModel`
        :param mean: The initially estimated mean of a shape such that the last dimensions are (n,)
                or (n,1).
        :type mean: numpy.ndarray
        :param covar: The initially estimated covariance matrix of a shape such that the last 
                dimensions are (n,n), every other dimension has to coincide with mean.
        :type covar: numpy.ndarray
        :param current_time: The time of the current initial state, 0 if None
        :type current_time: float
        :param exp_hist_len: The expected number of history entries, if to short is dynamically 
                extended
        :type exp_hist_len: positive int
        :param process_scalar: For further finetuning of the filter, will multiply with Q
        :type process_scalar: float
        :param obs_scalar:  For further finetuning of the filter, will multiply with R
        :type obs_scalar: float
        """
        self._model     = model.__copy__()
        self._state_dim = model.state_dim
        self._obs_dim   = model.obs_dim
        mu = numpy.atleast_1d(numpy.copy(mean))
        if mu.shape[-1] != 1:
            # make it so that the last dimension is 1 -> The last two dimensions should be (n,1)
            # except when n == 1, this will be handled in the next case
            mu = numpy.expand_dims(mu, axis=-1)
        if mu.ndim < 3:
            mu = mu[None,:,:] if mu.ndim == 2 else mu[None,:,None]
        n = self._state_dim
        assert mu.shape[-2] == n, f"'mean's last two dimensions should be ({n},) or ({n},1), instead it has shape {mean.shape}."
        assert covar.shape[-2:] == (n,n), f"'covar's last two dimensions should be ({n},{n}), instead it has shape {covar.shape}."
        assert numpy.prod(mean.shape[:-2]) == numpy.prod(covar.shape[:-2]), ("'mean's first dimensions"+
                f" and 'covar's should be broadcastable, instead get {mean.shape[:-2]} and {covar.shape[:-2]}")
        P_shape = tuple(list(mu.shape[:-2])+[n,n])
        P = numpy.copy(covar).reshape(P_shape)

        self._state_mean     = mu
        self._state_covar    = P
        self._state_time     = current_time if current_time is not None else 0
        self._process_scalar = process_scalar
        self._obs_scalar     = obs_scalar
        self._init_seed      = numpy.random.default_rng(seed).integers(2147483647)
        self._rng_gen        = numpy.random.default_rng(self._init_seed)
        try:
            init_state = self._state_distr()
        except Exception as e:
            print("\n Note that all parameters for the creation of the initial state distribution"+
                  " have to be set before the Filter is initialised by super().__init__()")
            raise e
        self._hist = {"times":  numpy.empty((exp_hist_len+1,1)),
                      # get example for how many and how large the arrays have to be
                      "estimates": { name: numpy.empty( (exp_hist_len+1, *arr.shape) ) for name,arr in init_state.items() },
                      "comp_time_model":  numpy.empty((exp_hist_len+1,1)),
                      "comp_time_filter": numpy.empty((exp_hist_len+1,1)),
                      "comp_time_rest": numpy.empty((exp_hist_len+1,1)), }
        for name,arr in init_state.items():
            self._hist["estimates"][name][0] = arr
        self._hist["times"][0] = self._state_time
        self._hist["comp_time_model"][0]  = 0
        self._hist["comp_time_filter"][0] = 0
        self._hist["comp_time_rest"][0]   = 0

        # index of first unfilled history rows
        self._hist_idx:int = 1
        # first impossible history index 
        self._hist_len = exp_hist_len+1

    def get_history(self):
        """
        The history of the model, where a number of parameters are documented for each past timestep.
        By default, the history tracks the past timesteps ``times``, the estimated state 
        distributions as ``estimates``, the processing time needed in the model 
        :meth:`~src.filters.abstract.AbstractDynamicModel.forecast` as ``comp_time_model`` and in 
        the pure :meth:`~src.filters.abstract.AbstractFilter._filter` as ``comp_time_filter``.
        
        If further parameters of the :meth:`~src.filters.abstract.AbstractFilter._filter` method 
        should be documented, the corresponding keys have to be created during creation, initiated 
        with zero arrays of fitting size as values (with the first axis iterating over timesteps).
        E.g., if a scalar parameter ``eta`` should be tracked too, the ``__init__`` method of the
        subclass should contain the line ``self.hist["eta"] = np.zeros((exp_hist_len,1))``. 
        And if three boolean values should be tracked in each timestep, the line should be
        ``self.hist["checks"] = np.zeros((exp_hist_len,3), dtype=bool)``.
        """
        return {name: arr[:self._hist_idx] for name,arr in self._hist.items() if name != "estimates"}
    
    @staticmethod
    def label() -> str:
        """
        Gives a simple handle for the filtering algorithm. For example, the standard Kalman Filter is
        often simply abbreviated as KF, or the robust Kalman Filter by Chang as RKF...

        :returns: the simple label
        :rtype: str
        """
        raise NotImplementedError()
    
    
    def label_long(self) -> str:
        raise NotImplementedError()

    def desc(self) -> str:
        """
        Gives a long description of the method, preferably with non-default parameters.

        :returns: the detailed description
        """
        raise NotImplementedError()

    def desc_short(self, maxc:int=20, hardcap:bool=False) -> str:
        """
        Gives a short description to be used in legends of plots or in tables

        :param maxc: the maximum number of characters in the output
        :param hardcap: Whether the maximum number of characters cannot be overstepped. If False, 
                important annotations can still be extended over the soft cap.
        :returns: a brief description
        """
        raise NotImplementedError()
    
    def _state_distr(self) -> Dict[str, numpy.ndarray]:
        """
        Returns the parameterisation of the current estimated distribution as by a method extending
        :meth:`~src.filters.abstract.AbstractDistribution.create_params`.
        """
        raise NotImplementedError()
    
    def get_state_distr(self, start_time: float | int | None = None, end_time: float | int | None = None, 
                        indexed=False) -> tuple(AbstractDistribution, numpy.ndarray):
        r"""
        Gives the estimated distribution of the state by this filter at time ``start_time`` 
        (inclusive) until time ``end_time`` (exclusive) and the corresponding time steps. 
        If ``start_time`` or ``end_time`` is ``None`` they will be interpreted as the current time and 
        ``start_time+1`` respectively.
        If ``indexed`` is ``True`` then ``start_time`` and ``end_time`` are expected to be integers
        indexing the past timesteps. For example, if you are interested in the first five state 
        estimations, you are not supposed to first find their timestamps, but instead can call 
        ``filter.get_state_distr(0,5,indexed=True)`` (Note that the ``end_time`` is still exclusive)

        So, ``state_distr()`` will return the current distribution; ``state_distr(t)`` will return
        the distribution at the first time step greater or equal to ``t``; ``state_distr(t0, t1)``
        will return a stack of all distributions in the interval :math:`[t0, t1)`. In comparison,
        ``state_distr(numpy.NINF, t1)`` will return a stack of all distributions before time ``t1`` and
        ``state_distr(numpy.NINF, numpy.PINF)`` will return a stack of all estimated state distributions at all
        timesteps (for which observations were presented).

        :return: the estimated state distributions and their timesteps.
        """
        if start_time is None:
            first_idx = self._hist_idx-1
        elif numpy.isneginf(start_time):
            first_idx = 0
        else:
            if indexed:
                first_idx = min( max(int(start_time),0), self._hist_idx-1)
            else:
                first_idx = numpy.searchsorted(self._hist["times"][:self._hist_idx, 0], start_time, side="left")
        if end_time is None:
            last_idx = first_idx +1
        elif numpy.isposinf(end_time):
            last_idx = self._hist_idx
        else:
            if indexed:
                last_idx = min( max(int(end_time),  0), self._hist_idx)
            else:
                last_idx = numpy.searchsorted(self._hist["times"][:self._hist_idx, 0], end_time, side="right")
        return (self._state_distr_class(
                    params = {name: arr[first_idx:last_idx] for name, arr in self._hist["estimates"].items()}
                ),
                self._hist["times"][first_idx:last_idx,0])
    
    def get_all_state_distr(self) -> tuple(AbstractDistribution, numpy.ndarray):
        """
        Returns all estimated state distributions and their timesteps. Equivalent to 
        :meth:`~src.filters.abstract.AbstractFilter.get_state_distr` evaluated on
        ``(0, numpy.PINF, indexed=True)``.
        """
        return self.get_state_distr(0, numpy.PINF, indexed=True)

    def _filter(self, obs, Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH) -> dict:
        """
        Pure filter update step, individual for each method. All preprocessing, logging, etc is done
        in :meth:`~src.filters.abstract.AbstractFilter.filter`, which should not be overridden by 
        subclasses. Note that all inputs labeled covariance matrices are indeed covariance matrices 
        returned by :meth:`~src.filters.abstract.AbstractDynamicModel.forecast`, 
        even if the model itself might work with different matrices.

        This method should have the side effect that the attributes 
        :attr:`~src.filters.abstract.AbstractFilter._state_mean` and 
        :attr:`~src.filters.abstract.AbstractFilter._state_covar`, as well as every other internal 
        attribute introduced by subclasses that represent the estimated state distribution, are 
        updated such that :meth:`~src.filters.abstract.AbstractFilter.state_distr` returns the 
        correct current estimated state distribution.

        :param obs: The observation of a single timestep, shape :math:`(1,m,1)`
        :param Fm: The a priori estimated new state mean, shape :math:`(k,n,1)`, where :math:`k` is
                the number of state means present.
        :param FPF: The a priori estimated new state covariance matrix before process noise, shape
                :math:`(k,n,n)`
        :param Q: The estimated process noise covariance matrix, shape :math:`(k,n,n)`
        :param HFm: The estimated observation mean, shape :math:`(k,m,1)`
        :param HFPFH: The estimated observation covariance without process or observation noise,
                shape :math:`(k,m,m)`
        :param HQH: The estimated observation noise covariance only induced by process noise, shape 
                :math:`(k,m,m)`
        :param R: The observation noise covariance matrix, shape :math:`(k,m,m)`
        :param FPFH: The cross correlation between current state and observation, shape :math:`(k,n,m)`
        :param QH: The cross correlation between process noise and observation, shape :math:`(k,n,m)`
        :param PF: The cross correlation between current and next state, shape :math:`(k,n,n)`
        :param PFH: The cross correlation between the current state and next observation, shape :math:`(k,n,m)`
        :return: Updates the internal state of the filter such that 
                :meth:`~src.filters.abstract.AbstractFilter.state_distr` returns the estimated 
                distribution of the current state. What is returned is an empty dictionary
                by default. Subclasses which track additional parameters added to 
                :attr:`~src.filters.abstract.AbstractFilter.hist` 
                should return these parameters here with the same keywords.
        """
        raise NotImplementedError("The method '_filter' has to be specified in the subclass.")

    def _filter_postprocess(self, processed_steps : int = 1):
        """
        Some methods might need some postprocessing opportunities after the filter method was called.

        :param processed_steps: The number of observations/timesteps were observed by the last 
                :meth:`~src.filters.abstract.AbstractFilter._filter` call.
        :type processed_steps: int
        """
        pass
    
    def filter(self, obs:numpy.ndarray, times:numpy.ndarray|float|None=None, debug=False) -> AbstractFilter:
        """
        Applies the Bayesian Filter to the given observations at the given timesteps. The results 
        are tracked in the internal :attr:`~src.filters.abstract.AbstractFilter._hist` dictionary 
        and can be extracted using the 
        :meth:`~src.filters.abstract.AbstractFilter.get_history` and 
        :meth:`~src.filters.abstract.AbstractFilter.get_state_distr` methods.
        
        Note that for each observation, the filter first updates its internal state estimation to 
        the timestep of the observation and using the internal dynamic model (prediction step) and 
        then updates this estimate based on the observation (update step).

        :param obs: The observations of a single or multiple timestep, shape :math:`(m,)` or 
                :math:`(T,m)` or :math:`(T,m,1)`
        :param times: the corresponding timesteps, shape :math:`(T,)` or a float when only a single
                timestep is observed.
        
        :returns: a reference to this filter
        """
        m = self._obs_dim
        obs = obs.reshape((-1,m,1))
        T = obs.shape[0]

        times = numpy.atleast_1d(times) if times is not None else self._state_time + numpy.arange(T)
        assert times.ndim == 1 and times.shape[0] == T, (
               f"'times' should have shape ({T},) but has {times.shape}." )
        while self._hist_idx+T > self._hist_len:
            #print(f" Extending history array: (which is currently at pos {self._hist_idx+1} of {self._hist_len} but needs {self._hist_idx+T})")
            for name, arr in self._hist.items():
                if name != "estimates":
                    self._hist[name] = numpy.concatenate([arr, numpy.empty(arr.shape)], axis=0)
                    #print("  "+name+" has now shape: ", self._hist[name].shape)
                else:
                    est_dict = self._hist[name]
                    for name2, arr2 in arr.items():
                        est_dict[name2] = numpy.concatenate([arr2, numpy.empty(arr2.shape)], axis=0)
                        #print("  "+name+"."+name2+" has now shape: ", est_dict[name2].shape)
            self._hist_len *= 2
            #print("  new history length: ", self._hist_len)

        estimates_pointer = self._hist["estimates"]
        for i in range(T):
            comp_time_rest   = time.process_time()
            comp_time_model  = time.process_time()
            Fm, FPF, Q, HFm, HFPFH, HQH, R, FPFH, QH, PF, PFH = self._model.forecast(
                    mu = self._state_mean, P = self._state_covar, 
                    next_time = times[i], dt = times[i] - self._state_time  )
            comp_time_model  = time.process_time() - comp_time_model

            comp_time_filter = time.process_time()
            filter_res       = self._filter(obs=obs[i:i+1,:,:], Fm=Fm, FPF=FPF, Q=self._process_scalar*Q,
                                            HFm=HFm, HFPFH=HFPFH, HQH=self._process_scalar*HQH,
                                            R=self._obs_scalar*R, FPFH=FPFH, 
                                            QH=self._process_scalar*QH, PF=PF, PFH=PFH  )
            comp_time_filter = time.process_time() - comp_time_filter

            self._state_time = times[i]

            hidx = self._hist_idx
            self._hist["times"][hidx] = times[i]
            self._hist["comp_time_model" ][hidx] = comp_time_model
            self._hist["comp_time_filter"][hidx] = comp_time_filter
            for name, arr in filter_res.items():
                #print(f"setting hist[{name}] which was {self._hist[name][hidx]} to {arr}")
                self._hist[name][hidx] = arr
            try:
                new_params = self._state_distr()
            except numpy.linalg.LinAlgError as e:
                if False: # we figured out that this is the problem usually
                    print(f"Filter {self.__class__.__name__} has a problem at step {hidx} with state estimation",
                        utils.nd_to_str(numpy.squeeze(self._state_mean)),
                        f" and covariance matrix:\n",
                        utils.nd_to_str(numpy.squeeze(self._state_covar)))
                raise e
            for name, arr in new_params.items():
                estimates_pointer[name][hidx] = arr
            if debug:
                print("new params:\n", utils.nd_to_str(numpy.squeeze(new_params["mu"]), shift=3))
                print("estimates:\n", utils.nd_to_str(numpy.squeeze(self._hist["estimates"]["mu"][hidx]), shift=3))
            self._hist_idx += 1
            comp_time_rest = time.process_time() - comp_time_rest - comp_time_model - comp_time_filter
            self._hist["comp_time_rest"][hidx] = comp_time_rest
        self._filter_postprocess(processed_steps=T)

        return self

class AbstractSmoother:
    """
    Abstract Class for all following implementations of Bayesian Smoothers. 
    
    Bayesian Smoothers here describe algorithms that propagate future information backwards in time.
    That is, they transform the estimates of a filter of the unknown state given all prior observations
    :math:`x_{t} | y_1, ..., y_t` for :math:`t=1,...,T` into estimates given all observations 
    :math:`x_{t} | y_1, ..., y_T` for :math:`t=1,...,T`.

    A Smoother here can be instantiated by a Filter and will process the past information of the Filter,
    see :meth:`~src.filters.abstract.AbstractFilter.get_history` and 
    :meth:`~src.filters.abstract.AbstractFilter.get_state_distr`, whenever 
    :meth:`~src.filters.abstract.AbstractSmoother.smoothen` is called.

    Note that this implementation allows for a mismatch of filters and smoothers (as long as 
    :meth:`_smoothen` does not explicitly depend on quantities computed during the forward pass of
    a certain :meth:`~src.filters.abstract.AbstractFilter._filter` call). For example, we can use a
    Kalman Smoother to smoothen estimates produced by a :class:`~src.filters.basic.StudentTStateFilter`,
    even though they represent their state estimations by different classes of distributions. 
    """

    _filter : AbstractFilter  #: The filter the smoother will smoothen the estimates for
    _init_seed : int          #: The initial seed with which _rng_gen was initiated
    _rng_gen: numpy.random.Generator #: RNG generator which should be used for generating all internal random processes.

    _state_mean = None        #: The mean  of the last smoothed timestep
    _state_covar = None       #: The (not necessarily) covariance matrix of the last smoothed timestep
    _state_distr_class = None #: The distribution for the smoothened state estimate
    _process_scalar = 1.      #: additional scaling for the process noise covariance matrix Q
    #: A dictionary documenting important dictionaries during smoothing, like the timesteps as ``"times"``
    #: the computation time for linearising the dynamical models as ``"comp_time_model_smoother"`` and
    #: the computation time of each :meth:`_smoothen` call as ``"comp_time_smoother"``, any further
    #: variables have to be defined in :meth:`_init_smoothing` and returned in :meth:`_smoothen`
    _hist = None                        

    def __init__(self, filter : AbstractFilter) -> None:
        """
        Initialises the smoother

        :param filter: The filter the smoother shall smoothen the estimates for. Also the source of
                the underlying dynamical model and initial randomness if required.
        """
        self._filter = filter
        self._process_scalar = filter._process_scalar
        self._init_seed = numpy.random.default_rng(filter._init_seed)
        self._rng_gen = numpy.random.default_rng(self._init_seed)

    def _init_smoothing(self, last) -> None:
        """
        Prepares the smoother to smooth the current filter by initiating all variables of the 
        smoother such that the distribution of the last estimate :math:`x_{T}|y_1,...,y_T` is 
        sufficiently initialised for :meth:`_smoothen`.
        Also initialises the documentation of all important variables in the local dictionary :attr:`_hist`
        """
        curr_distr, curr_time = self._filter.get_state_distr()
        self._state_mean  = numpy.copy(curr_distr.mean())
        self._state_covar = numpy.copy(curr_distr.cov())

        self._hist = {"comp_time_model_smoother": numpy.zeros((last,1)), 
                      "comp_time_smoother": numpy.zeros((last,1))}

    
    def _state_distr(self) -> Dict[str, numpy.ndarray]:
        """
        Returns the parameterisation of the distribution of the last smoothened state estimate given 
        by the internal variables, such as :attr:`_state_mean`
        and :attr:`_state_covar`
        """
        raise NotImplementedError

    def smoothen(self, last=None) -> tuple(AbstractDistribution, numpy.ndarray, Dict[str, numpy.ndarray]):
        """
        Smoothens the last ``last`` timesteps of the underlying filter. If ``last`` is ``None``
        smoothens every timestep.

        Returns a distribution object representing the smoothened estimates stacked over time, the
        individual timesteps and the :attr:`_hist` dictionary documenting important quantities during 
        smoothing
        """
        hist_idx  = self._filter._hist_idx
        last      = hist_idx if last is None else last
        last      = hist_idx if last > hist_idx else last
        assert last > 0, f"Processing last {last} timesteps not possible. Make sure that there is data to process"
        self._init_smoothing(last)
        start_idx = hist_idx - last
        init_state = self._state_distr()
        post_distr_params = {  name: numpy.empty( (last, *arr.shape) ) for name,arr in init_state.items() }
        for name in post_distr_params.keys():
            post_distr_params[name][-1] = init_state[name]

        filter_params = { name: arr[start_idx:hist_idx] 
                          for name, arr in self._filter._hist.items() 
                          if name not in ["estimates", "times", "comp_time_model", "comp_time_filter", "comp_time_rest"] }
        times = self._filter._hist["times"][start_idx:hist_idx]
        pre_estimates, times2 = self._filter.get_state_distr(float(times[0]), numpy.PINF)
        means, covs = (pre_estimates.mean(), pre_estimates.cov())
        n = self._filter._state_dim

        for i in range(last-2, -1, -1):
            mu, P = (means[i], covs[i])
            mu = mu.reshape((-1,n,1))
            P  = P.reshape((-1,n,n))
            k = mu.shape[0]
            Fm  = numpy.empty((k,n,1))
            FPF = numpy.empty((k,n,n))
            Q   = numpy.empty((k,n,n))
            PF  = numpy.empty((k,n,n))


            t = time.process_time()
            for j in range(k):
                Fm[j], FPF[j], Q[j], PF[j] = self._filter._model.forecast_smoother(
                        mu=mu[j], P=P[j], next_time=float(times[i+1]), dt=float(times[i+1]-times[i]) )
            comp_time_model_smoother = time.process_time()-t
            new_hist = self._smoothen(mu, P, Fm, FPF, Q*self._process_scalar, PF, 
                                      pre_args={name: arr[i] for name, arr in filter_params.items()})
            comp_time_smoother = time.process_time()-t-comp_time_model_smoother
            self._hist["comp_time_model_smoother"][i] = comp_time_model_smoother
            self._hist["comp_time_smoother"][i] = comp_time_smoother
            for name, attr in new_hist:
                self._hist[name][i] = attr
            new_distr = self._state_distr()
            for name in post_distr_params.keys():
                post_distr_params[name][i] = new_distr[name]

            #print(f" step {i} original   mean:  {mu.shape}\n", utils.nd_to_str(numpy.squeeze(mu), shift=3,))
            #print(f" step {i} smoothened mean:  {self._state_mean.shape}\n", utils.nd_to_str(numpy.squeeze(self._state_mean), shift=3,))
            #print(f" step {i} original   covar: {P.shape}\n", utils.nd_to_str(numpy.squeeze(P), shift=3,))
            #print(f" step {i} smoothened covar: {self._state_covar.shape}\n", utils.nd_to_str(numpy.squeeze(self._state_covar), shift=3,))

        return (self._state_distr_class(params=post_distr_params), times, self._hist)

    

    def _smoothen(self, mu, P, Fm, FPF, Q, PF, pre_args) -> Dict[str, numpy.ndarray]:
        """
        The individual smoothing function for a single timestep. Thus, the smoothened state 
        estimate of the next future timestep :math:`x_{t+1}|y_1,...,y_T` is represented by internal
        variables, in particular ``_state_mean`` and ``_state_covar``. 
        
        This method has the explicit side effect, that after each call of 
        :meth:`~src.filters.abstract.AbstractSmoother._smoothen` these internal variables of the 
        smoother will be updated to represent the smoothened estimate of :math:`x_{t}|y_1,...,y_T`.
        Their old values will be tracked in between calls of :meth:`~src.filters.abstract.AbstractSmoother._smoothen`.

        The parameters ``mu``, ``P`` are extracted from the filter and the parameters ``Fm``, ``FPF``, ``Q`` and ``PF`` 
        are meant to be computed by the models :meth:`~src.filters.abstract.AbstractDynamicModel.forecast_smoother` 
        method.
        
        Returns a dictionary with all important quantities that should be tracked in :attr:`_hist`,
        e.g. the number of iterations needed.

        :param mu: means of the filter :math:`x_{t}|y_1,...,y_t`, shape (k,n,1)
        :param P: covariances (!) of the filter :math:`x_{t}|y_1,...,y_t`, shape (k,n,n)
        :param Fm: expected means of :math:`x_{t+1}|y_1,...,y_t`, shape (k,n,n)
        :param FPF: expected covariances of :math:`x_{t+1}|y_1,...,y_t` without process noise, shape (k,n,n)
        :param Q: process noise covariance matrices :math:`Q_t`, shape (k,n,n)
        :param PF: cross-correlations between :math:`x_{t}|y_1,...,y_t` and :math:`x_{t+1}|y_1,...,y_t`, shape (k,n,n)
        :param pre_args: The additional arguments returned by the forward 
                :meth:`~src.filters.abstract.Abstractfilter._filter` pass.
        """
        raise NotImplementedError()
    