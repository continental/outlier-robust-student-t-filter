# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Defines the basic simulation scheme for this framework
"""

from __future__ import annotations
import pickle, json, hashlib
import multiprocessing, concurrent.futures
import os, time
from ast import literal_eval
from pathlib import Path
from array import array
import numpy as np
from tqdm import tqdm
import hyperopt
import matplotlib.pyplot as plt
import subprocess
try:
    import fitz
except ImportError:
    print("Couldn't find fitz (PyMuPDF). Only problematic for displaying rendered latex images")
    pass
try:
    from PIL import Image
except ImportError:
    print("Couldn't find PIL (pillow). Only problematic for displaying rendered latex images")
from IPython.display import display
from .. import filters, utils

print("-   Loading File 'simulate.py'")

def make_seed(seed : int | str, as_state=False):
    """
    Transforms an arbitrary length string into a seed for random number generation
    via deterministic and reproducible hashing. Alternatively, integers or other RNGs can
    also be used to create the new seed.
    """
    if isinstance(seed, str):
        seed = seed.encode()
        seed = int.from_bytes(seed + hashlib.sha512(seed).digest(), 'big')
    if as_state:
        return np.random.default_rng(seed)
    return np.random.default_rng(seed).integers(2147483647)

def nice_sci_nbr(fl, precision=2):
    """
    Converts a float into the latex code of its scientific representation.
    E.g., 23414.2 would be transformed into the latex expression for :math:`2.34 \\cdot 10^{4}`.
    If :math:`10^{\\texttt{precision}} > \\texttt{fl} > 10^{-\\texttt{precision}}` then no scientific notation is used

    :param fl: the float in question
    :param precision: how many digits will be printed after the dot, defaults to 2.
    :return: a string representing latex code for the expression
    """
    sign = "" if np.sign(fl) > 0 else "-"
    fl = np.abs(fl)
    try:
        log = np.log10(fl)
    except RuntimeWarning as e:
        print(f"Problem when forming logarithm of {fl}")
        raise e
    if precision > log and log > -precision:
        return sign+f"{fl:.{max(int(precision-log+2),0)}f}"
    s = f"{fl:.{precision}e}"
    b, e = s.split("e")
    e = e[::2] if (e[1]=='0') else e
    e = e[1:] if (e[0]=='+') else e
    return sign+b+"\\cdot 10^{"+e+"}"


def simulate_Singer(seed : int, init_state = np.zeros((6,)),
                    alpha : float = 0.5, beta : float = 0.25, sigma2 : float = 1, Nbr_sens : int = 1, 
                    R_inhom : float = 1, R_phase : float = 0, sigmaR : float = 0.25, R_distr : str = "Student-T", #("Normal"),
                    R_nu : float = 1, T : int = 500, dt : float = 0.20) -> tuple(
                        filters.abstract.AbstractDynamicModel, tuple(np.ndarray, np.ndarray), 
                        tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray) ):
    """
    Creates a two-dimensional :meth:`~src.filters.models.ExtendedSingerModel` given the parameters 
    with additional options for custom observation noise covariances, then samples from this model
    to provide reproducible date for testing different filtering methods.

    :param seed: The seed controlling the randomness in all underlying processes.
    :param alpha: The coefficient controlling the decay of the acceleration.
            :math:`\\frac{1}{\\alpha}` can be interpreted as the maneuver time constant, describing 
            how long each maneuver takes.
            Defaults to 0.5.
    :param beta: The coefficient controlling the decay of the velocity.
            This controls the friction in the system with :math:`\\frac{1}{\\beta}` describing for 
            how long each acceleration in the past still influences the current velocity.
            Defaults to 0.25.
    :param sigma2: The square of the coefficient :math:`\\sigma` controlling the magnitude of the random
            accelerations.
            Affects the magnitude of change between timesteps and acts as a multiplicative factor on
            the process noise covariance matrix :math:`Q`.
            Defaults to 1.
    :param Nbr_sens: The number of sensors, each measuring the position of the agent independently 
            of each other.
            Important for Student's t-distributed noise, since methods usually assume a multivariate
            Student's t-distribution, which would not be the case.
            Defaults to 1.
    :param R_inhom: deforms the observation noise covariance matrix :math:`R` away from the 
            :math:`2\\times 2` identity matrix.
            Instead, use the :math:`2\\times 2` diagonal matrix with 1 and :math:`r` on it's diagonal
            (:math:`r` being the chosen parameter here).
            In the case of multiple sensors. Each sensor will have R as an individual observation noise
            covariance matrix.
            Defaults to 1.
    :param R_phase: While the filters assume to have observation noise covariance matrix 
            :math:`diag(1,r)`, the observations are actually generated with a rotated version of 
            :math:`R`. Rotated by :math:`\\phi \\cdot 360°` degrees to be precise.
            Defaults to 0.
    :param sigmaR: additional scaling of the covariance matrix :math:`R`.
            Note that we further scale :math:`R` by the average distance traveled by the agent per
            second, such that the spread of observation outliers is comparable for different dynamic
            model parameters.
            Defaults to 0.25.
    :param R_distr: The distribution of the observation noise. Either a multivariate "Normal" or 
            "Student-T" distribution.
            Defaults to "Student-T".
    :param R_nu: If the observation noise is Student's t-distributed, controls the degrees of freedom
            :math:`\\nu` of the distribution. This parameter controls how frequent and how distant outliers 
            are from the ground truth. A low parameter induces more and farther outliers.
            Defaults to 1.
    :param T: How many timesteps are sampled.
            Defaults to 500.
    :param dt: The difference between two timesteps
            Defaults to 0.20.

    :returns: ``(model, (x0, P0), (x,y,v,e,t))`` a tuple of the created :meth:`~src.filters.models.ExtendedSingerModel`,
            initial mean and covariance matrix estimates for filtering, the ground-truth agent states
            :math:`x_t`, observations :math:`y_t`, process noises :math:`v_t` and observation noises
            :math:`e_t` and simulation times :math:`t`
    """
    R = np.array([[1,0],[0,R_inhom]])
    R = np.kron(np.eye(Nbr_sens), R)
    H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],])
    H = np.concatenate([H]*Nbr_sens, axis=0)

    # initial state of the simulation 
    P0 = filters.models.extendedSinger_noise(alpha=alpha, beta=beta, sigma2=sigma2, d=2)(dt*5)
    x0 = filters.distributions.NormalDistribution(seed=seed, mu=init_state, P=P0).rvs().flatten()

    rotated_R = utils.covar_from_ellipse(theta=R_phase*2*np.pi, height=R_inhom)
    rotated_R = np.kron(np.eye(Nbr_sens), rotated_R)
    R_RNG = np.random.default_rng(seed)
    if R_distr == "Normal":
        def R_distr(mat):
            return filters.distributions.NormalDistribution( mu=np.zeros((2,)), P=mat, 
                                                             seed=R_RNG, peek=False )
    else:
        def R_distr(mat):
            return filters.distributions.Joint_Indep_Distribution.join_distr( 
                *[ 
                    filters.distributions.StudentTDistribution(
                        mu=np.zeros((2,)), P=utils.KLDmin_Norm_T(2,R_nu)*mat[2*i:2*(i+1),2*i:2*(i+1)], 
                        nu=R_nu, P_covar=False
                    ) 
                    for i in range(Nbr_sens)
                ], 
                seed=R_RNG, peek=False 
            )
    model = filters.models.ExtendedSingerModel(alpha=alpha, beta=beta, sigma2=sigma2, d=2, 
                                               R=sigmaR*rotated_R, H=H, seed=seed, R_distr=R_distr)
    x, y, v, e = model.sample(x=init_state, t0=0, dt=np.full((T,), dt), give_noise=True)
    # Rscale is the arclength of the trajectory normalised by the duration of the simulation
    Rscale = np.sqrt((x[1:,0]-x[:-1,0])**2+(x[1:,1]-x[:-1,1])**2).sum()/T/dt
    # rescaling the noise to make comparisons for different beta values meaningful
    # i.e., the lower beta, the faster the agent goes and the longer the trajectories
    # therefore we renormalise (only) the noise by the length of the trajectory
    y += (Rscale-1)*e
    e = Rscale * e
    t = np.arange(1,T+1)*dt
    # Cov(r R) = r**2 Cov(R), hence the different scaling between e and R
    model._R = (sigmaR* Rscale**2)* R
    return model, (x0, P0), (x,y,v,e,t)


#: A list of attributes that are logged during testing simulations of the filters.
#: Strings are keys for the dictionary returned by :meth:`~src.filters.AbstractFilter.get_history`.
FILTER_HIST_LIST = ["comp_time_filter", "comp_time_model", "comp_time_rest",]
#: A list of attributes that are logged during testing simulations of the smoothers.
#: Strings are keys for the dictionary returned as part of :meth:`~src.filters.AbstractSmoother.smoothen`.
SMOOTHER_HIST_LIST = ["comp_time_smoother", "comp_time_model_smoother",]
def in_hist_list(key):
    """
    Test if ``key`` is a superstring of a valid key. E.g., the data contains the key 
    ``comp_time_smoother_RTS`` which is the Rauch-Tung-Striebel result for the key 
    ``comp_time_smoother`` and thus valid.
    """
    return any(substr in key for substr in FILTER_HIST_LIST+SMOOTHER_HIST_LIST)

#: Tuples consist of the string they will be documented by and functions that compute the 
#: values from the true state positions and the estimated state distributions.
METHOD_EST_LIST  = [("euc_error",       lambda x, est: np.linalg.norm((x-est.mean())[1:,:,0], axis=1)),
                    ("likelihoods_pos", lambda x, est: est.marginal([0,1]).logpdf(x[:,[0,1],:])[1:]),
                    ("likelihoods",     lambda x, est: est.logpdf(x)[1:],),]
HISTPREC = 1000000000  #: precision of values in FILTER_HIST_LIST
ESTPREC  = 1000        #: precision of values in METHOD_EST_LIST (i.e. error values will have at most 4 positions after the decimal point)

def dict2str(d : dict):
    """
    Transforms a dictionary into a string. Used, e.g., to note the hyperparameter set for each run of a filter.
    """
    if len(d) == 0:
        return "_"
    return "_".join([f"{str(key)}-{str(val) if type(val)!=float else f'{val:.4f}' }" for key,val in d.items()])
def params2path(model_kwargs, filterclass=None, filter_kwargs=None):
    """
    Used to define where final and intermediary results should be saved.
    """
    if filterclass is None:
        return "../data/quantitative_results/"+dict2str(model_kwargs)
    if filter_kwargs is None:
        return "../data/quantitative_results/"+dict2str(model_kwargs)+"/"+filterclass.__name__
    return "../data/quantitative_results/"+dict2str(model_kwargs)+"/"+filterclass.__name__+"/"+dict2str(filter_kwargs)

def no_GT_knowledge(x,y,v,e,t):
    """
    A function providing **no** ground truth knowledge. As arguments, it receives ground truth states ``x``, 
    observations ``y``, process noise ``v``, observation noise ``e`` and times ``t``.
    The only filter currently receiving ground truth knowledge (by a different function with the same signature
    :meth:`~src.eval.simulate.xe_GT_knowledge`) is :class:`~src.filters.proposed.StudentTFilter_GT`.
    """
    return {}
def xe_GT_knowledge(x,y,v,e,t):
    """
    A function providing the ground truth knowledge on true states ``x`` and true observation noises ``e``.
    For more information, check out :meth:`~src.eval.simulate.no_GT_knowledge`.
    """
    return {"GTx":x, "GTe":e}

class TooManyFailedAttempts(Exception):
    """
    This is raised when a filtering method has failed too many times during simulations
    """
    def __init__(self, message="Too many failed attempts."):
        self.message = message
        super().__init__(self.message)

def test_filter_for_file(init_seed: int | str, N: int, filterclass: filters.abstract.AbstractFilter, 
                         filter_kwargs={}, model_kwargs={}, GT_knowledge=no_GT_knowledge, queue=None,
                         MAX_ERROR=20):
    """
    Simulates N trajectories using ``simulate_Singer(seed=next_seed, **model_kwargs)``, ie. with the 
    provided seed and model parameters and tests the performance of a filter of class ``filterclass``
    and parameters ``filter_kwargs`` on these trajectories (+ smoothed version by Gaussian/Student-t Smoothers).
    The computation times, euclidean errors in position estimates, likelihoods in position, and 
    likelihoods in (position, vellocity, acceleration). 
    Returned is the path of the target directories and an array of compressed results if no queue is given,
    otherwise, these are appended to the queue.
    Only allows the filter to fail to terminate ``MAX_ERROR`` times, otherwise will retun maximal values for
    all tracked quantities. Also happens if the number of errors becomes larger than the number of tries
    """
    MAX_ERROR = min(MAX_ERROR, N)
    data = {s+post : [] for s,lam in METHOD_EST_LIST for post in ["","_RTS", "_STS"]}; data["seeds"] = []
    for s in FILTER_HIST_LIST:
        data[s] = []
    for s in SMOOTHER_HIST_LIST:
        data[s+"_RTS"] = []
        data[s+"_STS"] = []

    dir_path = params2path(model_kwargs, filterclass=filterclass, filter_kwargs=filter_kwargs)
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    if Path(dir_path+"/seeds.pickle").exists():
        # looks if the seeds have already been processed before and skips them
        with open(dir_path+"/seeds.pickle", 'rb') as f:
            found_seed = False; already_computed = 0; last_seed = init_seed
            while True: # reading seeds line by line to find the init_seed
                if not f.peek(4): # if at EOF break
                    break
                arr = array('i'); byte_data = pickle.load(f); arr.frombytes(byte_data)
                if found_seed:
                    already_computed += len(arr)
                    last_seed = arr[-1]
                else:
                    try:
                        already_computed = len(arr)-arr.index(init_seed)
                        found_seed = True
                    except ValueError:
                        pass
        if found_seed:
            N -= already_computed
            if N <= 0:
                    if queue is None:
                        return (dir_path, None)
                    return
            init_seed = make_seed(last_seed)
    next_seed = init_seed

    MININT = np.iinfo(np.int32).min; MAXINT = np.iinfo(np.int32).max
    i = 0; err_cnt = 0
    while i < N and err_cnt < MAX_ERROR :
        error_bool = True
        while error_bool and err_cnt < MAX_ERROR :
            try:
                model, (x0, P0), (x,y,v,e,t) = simulate_Singer(seed=next_seed, **model_kwargs)
                filter = filterclass(model=model, mean=x0, covar=P0, current_time=0, exp_hist_len=t.shape[0],
                                    **filter_kwargs, **GT_knowledge(x,y,v,e,t))
                filter.filter(obs=y, times=t)
                est, t2 = filter.get_all_state_distr()
                RTS_est, _ts, smoother_hist_RTS = filters.basic.KalmanSmoother(filter).smoothen()
                STS_est, _ts, smoother_hist_STS = filters.proposed.StudentTSmoother(filter).smoothen()
                error_bool = False
            except Exception as excep:
                # by returning the writing head to the start of the line after writing a '-' we overwrite 
                #  recurring error message, but indicate the number of additional error messages by the 
                #  increasing number of '-'s at the end of the message (if only few parallel processes)
                #  (since the '-' are not overwritten since the message is too short)
                print(f"-\rGot exception '{excep}' in filterclass {filterclass.__name__}, resampling", end="")
                err_cnt += 1
                if err_cnt >= MAX_ERROR:
                    raise TooManyFailedAttempts()
                next_seed = make_seed(next_seed) #np.random.default_rng(next_seed).integers(2147483647)
        data["seeds"].append([next_seed])
        for s in FILTER_HIST_LIST:
            arr = np.clip(filter.get_history()[s][1:]*HISTPREC, MININT, MAXINT)
            data[s].append((arr).astype(np.int32).flatten())
        for s in SMOOTHER_HIST_LIST:
            for hist, post in [(smoother_hist_RTS, "_RTS"), (smoother_hist_STS, "_STS")]:
                arr = np.clip(hist[s][1:]*HISTPREC, MININT, MAXINT)
                data[s+post].append((arr).astype(np.int32).flatten())
        for curr_est, post in [(est,""),(RTS_est,"_RTS"), (STS_est, "_STS")]:
            for s,lam in METHOD_EST_LIST:
                arr = np.clip(lam(x,curr_est)*ESTPREC, MININT, MAXINT)
                data[s+post].append( arr.astype(np.int32).flatten() )
        next_seed = make_seed(next_seed) #np.random.default_rng(next_seed).integers(2147483647)
        i += 1

    #big_data = {key: np.concatenate(val) for key, val in data.items()}
    #big_data = {key: (min(val), max(val)) for key, val in big_data.items()  if key!="seeds" and key[-3:]!="RTS" and ((abs(min(val)+1)>>30 > 0) or (abs(max(val))>>30 > 0))}
    #if len(big_data) > 0:
    #    print(f"found suspiciously big data in {filterclass.__name__}: {big_data}")
    data = {key: array('i', np.clip(np.concatenate(val), -(2<<31), (2<<31)-1)).tobytes() for key, val in data.items()}
    if queue is None:
        return (dir_path, data)
    else:
        queue.put((dir_path, data))
        return dir_path

#: List of filter methods that are tested on simulated trajectories.
#: the tuples consist of
#:
#: - a reference to the filter class
#: - a directory of parameters for initialising the filter
#: - a directory of hyperopt search spaces for parameters for initialising the filter. 
#:   Over these search spaces the hyperparameter optimisation is performed
#: - a function that returns additional ground truth knowledge over the simulation
#:   for the filtering. :meth:`~src.eval.simulate.no_GT_knowledge` provides no knowledge.
Filters_To_Optimise = [
    ( filters.proposed.StudentTFilter_GT, {"nu":1}, 
        {}, 
      xe_GT_knowledge ),
    #   We also give initially found 'optimal' hyperparameter configurations to
    #   justify our choice of search sapce definitions.
    # manually: {#nu': 3}, by hyperopt: {'nu': 3.2644}
    ( filters.proposed.StudentTFilter, {}, 
        { "nu":hyperopt.hp.quniform("nu",1,7.5,0.0001) }, 
      no_GT_knowledge ),
    # manually: {'SSM': 0, 'nu_SSM': 3}, by hyperopt: {'SSM': 0, 'nu_SSM': 3.6022}
    ( filters.robust.Huang_SSM, {"separate":True, "process_non_gaussian": True, "sigma_SSM": 1, "gating":1},
        { "SSM":hyperopt.hp.choice("SSM", ["log", "sqrt"]), 
          "nu_SSM":hyperopt.hp.quniform("nu_SSM",0.5,7.5,0.0001) }, 
      no_GT_knowledge ),
    # manually: {'alpha': 0.01}, by hyperopt: {'alpha': 0.0883}
    ( filters.robust.chang_RKF, {}, 
        { "alpha":hyperopt.hp.quniform("alpha",0,0.2,0.0001) }, 
      no_GT_knowledge ),
    # manually: {'nu': 8}, by hyperopt: {'nu': 12.3095}
    ( filters.robust.Agamennoni_VBF, {"gating":1}, 
        { "nu":hyperopt.hp.quniform("nu",1,15,0.0001) }, 
      no_GT_knowledge ),
    # by hyperopt: {'gating': 0.9999}
    ( filters.basic.KalmanFilter, {}, 
        { "gating": hyperopt.hp.quniform("gating",0.9,1+1e-6,0.0001)}, 
      no_GT_knowledge ),
    # manually: {'alpha': 0.01}, by hyperopt: {'alpha': 0.1764}
    ( filters.robust.chang_ARKF, {}, 
        { "alpha":hyperopt.hp.quniform("alpha",0,0.2,0.0001) }, 
      no_GT_knowledge ),
    # manually: {'rho': 0}, by hyperopt: {'rho': 0.0001}
    ( filters.robust.Saerkkae_VBF, {"gating":1}, 
        { "rho":hyperopt.hp.quniform("rho",0,0.15,0.0001) },
      no_GT_knowledge ),
    # manually: {'state_nu': 5, 'process_gamma': 11, 'obs_delta': 2.5}
    # by hyperopt: {'state_nu': 4.4041, 'process_gamma': 12.3875, 'obs_delta': 2.1737}
    ( filters.robust.roth_STF, {},
        { "state_nu": hyperopt.hp.qnormal("state_nu",mu=5.5,sigma=0.4,q=0.0001),
          "process_gamma": hyperopt.hp.qnormal("process_gamma",mu=12,sigma=0.4,q=0.0001),
          "obs_delta": hyperopt.hp.quniform("obs_delta",2.0001,3,q=0.0001), },
      no_GT_knowledge ),
]

def parallelise_test_filters(init_seed, N, model_kwargs, filter_class_kwargs_searchspace_GT_list, k=1, 
                             max_evals=100, create_threads_per_method=True, verbose=False):
    """
    Performs hyperparameter optimisation for a list of filters via parallelises calls of 
    :meth:`~src.eval.simulate.test_filter_for_file`.

    :param seed: A seed controlling the random exploration of the hyperparameter search space, as well as
            the generation of random sample trajectories, such that this function is reproducible.
    :param N: The number of trajectories processed by a single call to :meth:`~src.eval.simulate.test_filter_for_file`.
            The total number of trajectories processed for each set of hyperparameters is ``N`` :math:`\\cdot` ``k``.
    :param model_kwargs: the arguments of the :meth:`~src.filters.models.ExtendedSingerModel` used to generate
            the trajectories. See :meth:`~src.eval.simulate.test_filter_for_file` for more info.
    :param filter_class_kwargs_searchspace_GT_list: A list of tuples per filter method that are tested on 
            simulated trajectories. We will use :attr:`~src.eval.simulate.Filters_To_Optimise` in general,
            but see its documentation for more detail on the structure of the list.
    :param k: How many sets of ``N`` trajectories should be processed in parallel. I.e., ``k`` instances are
            running simultaneously and the total number of trajectories processed for each set of 
            hyperparameters is ``N`` :math:`\\cdot` ``k``.
    :param max_evals: The maximal number of hyperparameter sets that will be processed per method.
    :param create_threads_per_method: If True, all methods/filters will be processed simultaneously.
    :param verbose: Printing progress bar of ``hyperopt.fmin``. It is recommended to set to ``False`` in case of multiple
            kernels running in parallel, especially if ``create_threads_per_method=True``.
    """
    # Since we run k instances on N simulated trajectories each, find initial_seeds
    # for each of the k instances
    init_seeds = [init_seed]; test_seed = init_seed
    for i in range(1,k): 
        test_seed = (48271 * test_seed) % ((1 << 31)-1) # C++11's minstd_rand
        init_seeds.append(test_seed)


    manager = multiprocessing.Manager() # allows for communication between processes
    writing_queue = manager.Queue()     # collects all the data from the test_filter_for_file calls 
                                        # to be written without interference
    def writing_queue_listener(q):
        # process checking the writing queue and writing down everything it finds
        while True:
            if not q.empty():
                path, data = q.get()
                if path == "kill":
                    print("Writing listener killed. Further simulations are not effective!")
                    break
                Path(path).mkdir(parents=True, exist_ok=True)
                for name, bytes in data.items():
                    with open(path+"/"+name+".pickle", 'ab') as f:  
                        pickle.dump(bytes, f)

    def hyperparameter_search(filter_class_kwargs_searchspace_GT, writing_queue, max_evals=max_evals):
        # Internally parallelised hyperparameter search
        fliter_class, kwargs, search_space, GT = filter_class_kwargs_searchspace_GT
        def counter_gen():
            x = 0
            while True:
                x += 1
                yield x
        counter = counter_gen()
        def objective_function(opt_args):
            # function the hyperparameter search tries to minimise, here the
            # mean over the estimation errors in the positions over multiple
            # trajectories
            new_filter_kwargs = dict(kwargs,**opt_args)
            with concurrent.futures.ProcessPoolExecutor() as process_executor:
                processes = [process_executor.submit(
                                test_filter_for_file, init_seed=seed, N=N, 
                                filterclass=fliter_class, model_kwargs=model_kwargs, 
                                GT_knowledge=GT, queue=writing_queue,
                                filter_kwargs=new_filter_kwargs)
                            for seed in init_seeds]
                for future in concurrent.futures.as_completed(processes):
                    try: # wait for all processes to finish
                        future.result()  # This will raise an exception if the thread had an exception
                    except TooManyFailedAttempts as e:
                        print(e)
                        for remaining_future in processes:
                            if remaining_future != future and not remaining_future.done():
                                remaining_future.cancel()
                        print(f" !!! Running {fliter_class.__name__} ({next(counter)}) with parameters { {key:f'{val:.4f}' if type(val)==float else f'{val}' for key,val in opt_args.items()} } results in too many failed attempts !!! !!!")
                        return np.PINF
                    except Exception as e:
                        print(f"Process raised an exception: {e}")
                        raise e           
            while not writing_queue.empty():
                time.sleep(0.01)
            # everything is computed. Now computes the mean error over all trajectories as 
            # objective function we want to minimise via the hyperparameter search
            means = []
            for file in ["/euc_error.pickle", "/likelihoods.pickle", "/likelihoods_pos.pickle"]:
                mean = 0; nbr = 0
                with open(params2path(model_kwargs=model_kwargs, filterclass=fliter_class, filter_kwargs=new_filter_kwargs)+file, 'rb') as f:
                    while True:
                        if not f.peek(4): # if at EOF break
                            break
                        arr = array('i'); byte_data = pickle.load(f); arr.frombytes(byte_data)
                        new_nbr = nbr + len(arr)
                        mean = mean*nbr/new_nbr + sum(arr)/ESTPREC/new_nbr
                        nbr = new_nbr
                    means.append(mean)
            # These values are chosen such that the STF_GT method scores around 1 and the normal method around 2
            # such that we map these values to the same scale
            objective_vals = [(means[0]-1.5246)/0.2466 , (means[1]+4.7304)/-0.6634, (means[2]+0.5647)/-0.6902]
            objective_worst = np.argmax(objective_vals)
            print(f" !!! Running {fliter_class.__name__} ({next(counter)}) with parameters { {key:f'{val:.4f}' if type(val)==float else f'{val}' for key,val in opt_args.items()} } results in a error mean of {means[0]:.4f} ({objective_vals[0]:.4f}), likelihood mean of {means[1]:.4f} ({objective_vals[1]:.4f}), likelihood_pos mean of {means[2]:.4f} ({objective_vals[2]:.4f}). With the worst being {['error','likelihood','likelihood_pos'][objective_worst]} !!!")
            return objective_vals[objective_worst]

        if len(search_space) > 0:
            best_params = hyperopt.fmin(fn=objective_function, space=search_space, 
                                        algo=hyperopt.tpe.suggest, max_evals=max_evals,
                                        rstate=make_seed(fliter_class.__name__, as_state=True),
                                        verbose=verbose) #no progress bar
        else:
            best_params = {}
            print("empty search space")
            objective_function(best_params)
        print(fliter_class.__name__, " computed the best parameters:\n", best_params, "\n")
        with open(params2path(model_kwargs=model_kwargs, filterclass=fliter_class)+"/best.txt", 'a') as f:
            f.write("\n#################\nMost recent best parameters:\n"+str(best_params)+"\n#################\n")
    
    # parallelise the optimisation of all filter methods
    if create_threads_per_method:
        with concurrent.futures.ThreadPoolExecutor() as thread_executor:
            # make sure that the process of writing data from the queue to memory always runs
            queue_listener = multiprocessing.Process(target=writing_queue_listener, args=(writing_queue,))
            queue_listener.start() 
            # start a thread for each method
            threads = [thread_executor.submit(hyperparameter_search, filter_class_kwargs_searchspace_GT=filter_class_kwargs_searchspace_GT,
                                            writing_queue=writing_queue, max_evals=max_evals, ) 
                    for filter_class_kwargs_searchspace_GT in filter_class_kwargs_searchspace_GT_list]
            # Wait for all threads to finish
            for future in concurrent.futures.as_completed(threads):
                try:
                    future.result()  # This will raise an exception if the thread had an exception
                except Exception as e:
                    print(f"Thread raised an exception: {e}")
                    raise e
            writing_queue.put(("kill", None))
            print("killing listener")
            # Terminate the listener
            queue_listener.join()
    else:
        queue_listener = multiprocessing.Process(target=writing_queue_listener, args=(writing_queue,))
        queue_listener.start()
        for filter_class_kwargs_searchspace_GT in filter_class_kwargs_searchspace_GT_list:
            hyperparameter_search(filter_class_kwargs_searchspace_GT=filter_class_kwargs_searchspace_GT,
                                  writing_queue=writing_queue, max_evals=max_evals)
        writing_queue.put(("kill", None))
        print("killing listener")
        # Terminate the listener
        queue_listener.join()

def key2filterkwargs(key):
    """
    Tries to reconstruct the hyperparameters used to initialise a filter. 
    This is targeting the result categories (such as ``StudentTFilter_GT;nu-1``)
    but also works with results from :meth:`~src.eval.simulate.dict2str` as arguments,
    where ``\\`` has been replaced with ``;``.
    """
    name, params = key.split(';')[:2]
    if params != "_":
        param_split = params.split('-')
        params = [(param_split[i].split('_')[0], '_'.join(param_split[i].split('_')[1:])) for i in range(1, len(param_split)-1)]
        params = [param_split[0]] + [substr for substrtuple in params for substr in substrtuple if len(substr) > 0] + [param_split[-1]]
        params = {params[2*i]: params[2*i+1] for i in range(len(params)//2)}
        def dynamic_cast(input):
            if input == "False":
                return False
            elif input == "True":
                return True
            try:
                return int(input)
            except:
                try:
                    return float(input)
                except:
                    return input
        params = {k: dynamic_cast(v) for k, v in params.items()}
    else:
        params = {}
    filterclass = filters.METHODS[name]
    return filterclass, params

def collect_best_results(model_kwargs, filter_class_kwargs_searchspace_GT_list):
    """
    Collects the results of the best-performing hyperparameters and saves them in a
    JSON file called ``best_results.json`` in the super directories of the experiments.
    The results contain a compressed description of the data distributions in the experiments
    in the form of percentiles. See :meth:`~src.eval.simulate.collect_results` to see how 
    the results are extracted from the raw data.
    """
    results = {}
    for fliter_class, kwargs, search_space, GT in filter_class_kwargs_searchspace_GT_list:
        class_path = params2path(model_kwargs=model_kwargs, filterclass=fliter_class)
        with open(class_path+"/best.txt") as f:
            print(f"for {fliter_class}: ", end="")
            for line in f.readlines():
                if line[0] == '{':
                    best = literal_eval(line)
            if len(best) > 0:
                best = hyperopt.space_eval(search_space, best)
            print(best)
        best_kwargs = dict(kwargs, **best)
        key = fliter_class.__name__+";"+dict2str(best_kwargs)
        results[key] = collect_results(model_kwargs=model_kwargs, filterclass=fliter_class, filter_kwargs=best_kwargs)
    with open(params2path(model_kwargs=model_kwargs)+'/best_results.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    return results

def extract_mean_and_percentiles(data, perc=[0,1,5,25,50,75,95,99,100]):
    """
    Extracting mean and percentiles from the data. ``data`` has to be a list/array of floats.
    """
    return dict({f"perc_{perc[i]}":vals for i,vals in enumerate(np.percentile(data,perc)) }, mean=np.sum(data) / len(data))

def collect_results(model_kwargs, filterclass, filter_kwargs, extract_fct=extract_mean_and_percentiles):
    """
    Given an experiment's descriptors, finds the computed data and extract the interesting quantities.
    The function also saves the produced dictionary of results as a JSON file and returns the directory.
    See :meth:`~src.eval.simulate.dict2str` on how the experiment data is found.
    
    :param model_kwargs: the used model arguments for :meth:`~src.filters.models.ExtendedSingerModel` to generate 
            the trajectories in the experiment.
    :param filterclass: for which filter the data should be collected
    :param filter_kwargs: which filter arguments were used when initiating the filter
    :param extract_fct: a function that takes an array of floats and returns a dictionary. 
            Defaults to :meth:`~src.eval.simulate.extract_mean_and_percentiles`.
    """
    path = params2path(model_kwargs=model_kwargs, filterclass=filterclass, filter_kwargs=filter_kwargs)
    results = {}
    if Path(path+'/result.json').exists():
        with open(path+'/result.json', 'r') as file:
            results = json.load(file)
    for filename in os.listdir(path):
        category = '.'.join(filename.split('.')[:-1])
        if not category in results:
            results[category] = {}
        if filename.endswith(".pickle") and category != "seeds":
            with open(path +"/"+ filename, 'rb') as readfile:
                arr = array('i')
                while True:
                    if not readfile.peek(4): # if at EOF break
                        break
                    arr.frombytes(pickle.load(readfile))
                extracted_dict = extract_fct(arr)
                prec = HISTPREC if in_hist_list(category) else ESTPREC
                for key, vals in extracted_dict.items():
                    results[category][key] = vals/prec
    with open(path+'/result.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)
    return results


#: Mapping the data categories to headers in the table
category_map = {
    "comp_time_filter":"\\text{Runtime [ms]}", "comp_time_model":"\\text{Runtime Model [ms]}", "comp_time_rest":"\\text{Runtime Rest [ms]}",
    "euc_error":"\\text{Position Errors [AU]}", "likelihoods":"\\text{Log GT likelihood}","likelihoods_pos":"\\text{Log GT Position likelihood}",
    "comp_time_smoother_RTS":"\\text{Runtime [ms]}", "comp_time_model_smoother_RTS":"\\text{Runtime Model [ms]}",
    "euc_error_RTS":"\\text{Position Errors [AU]}", "likelihoods_RTS":"\\text{Log GT likelihood}","likelihoods_pos_RTS":"\\text{Log GT Position likelihood}",
    "comp_time_smoother_STS":"\\text{Runtime [ms]}", "comp_time_model_smoother_STS":"\\text{Runtime Model [ms]}",
    "euc_error_STS":"\\text{Position Errors [AU]}", "likelihoods_STS":"\\text{Log GT likelihood}","likelihoods_pos_STS":"\\text{Log GT Position likelihood}"
}

#: rescales the measurements of compute time from seconds to milliseconds
rescale_fct = lambda key: 1000 if in_hist_list(key) else 1

#: Lists the percentiles to show for a short form of the table.
simple_col_sequence = [("comp_time_filter", ["mean"]), 
                       ("euc_error",       ["mean", "perc_25", "perc_50", "perc_75", "perc_95", "perc_99"]), 
                       ("likelihoods",     ["mean", "perc_1",  "perc_5",  "perc_25", "perc_50", "perc_75"]), 
                       ("likelihoods_pos", ["mean", "perc_1",  "perc_5",  "perc_25", "perc_50", "perc_75"])]
#: Lists the order of the rows each method shall appear in 
simple_row_sequence = ["StudentTFilter_GT", "StudentTFilter", "Huang_SSM", "Agamennoni_VBF", "chang_RKF", 
                       "chang_ARKF", "Saerkkae_VBF", "roth_STF", "KalmanFilter"]

def subcategory2str(sub):
    """
    Transforms the subcategory of a data category into Latex code.
    """
    if sub[:5] == "perc_":
        p = int(sub[5:])
        if p==0:
            return "\\text{min}"
        if p==100:
            return "\\text{max}"
        return f"{p}\\%"
    return "\\text{"+sub+"}"

def find_str_key(target, key_list):
    """
    Finds the key in the key_list (list of strings) that most closely resembles the ``target`` string.
    This key has the longest (non-consecutive) substring equivalent to a prefix of ``target``.

    I.e. find_str_key("Huang_SSM;nu_SSM_3.305-SSM_log", ["Huang_SSM;SSM_log-nu_SSM_3.305","Huang_SSM"])
    would find the prefix "Huang_SSM;nu_SSM_3.305" and return "Huang_SSM;SSM_log-nu_SSM_3.305". 
    Note that the keys representing the methods include the parameters they used.

    Note that if you provide a subset of the parameters, they have to be in the same order
    """
    def make_prefix_comp_function(string):
        def comp_function(string2):
            idx1 = 0
            for idx2 in range(len(string2)):
                if string2[idx2] == string[idx1]:
                    idx1 += 1
            return idx1
        #return lambda string2: len(os.path.commonprefix([string, string2]))
        return comp_function
    return max(key_list, key=make_prefix_comp_function(target))

def larger_better_only_likelihood(keystring):
    """
    Simple function that returns ``True`` if larger data in the category given by keystring
    signifies better performance. In our case, computation time and Euclidean error should
    be as small as possible, while the likelihood of the ground truth states should be
    as high as possible.
    """
    return "likelihood" in keystring 
def not_compare_to_GT(keystring):
    """
    Simple function that declares if a method should be considered for the best performing method
    per column, and hence boldened in the table. Here, all methods that do not carry 'GT' (Ground Truth)
    in their name are considered.
    """
    return "GT" in keystring
def Roman(number):
    """
    Transforms an integer in its Roman numeral equivalent string
    """
    out = ""
    for val,rom in [(1000,'M'), (900,'CM'), (500,'D'), (400, 'CD'), 
                    (100,'C'), (90,'XC'), (50, 'L'), (40,'XL'), 
                    (10, 'X'), (9,'IX'), (5,'V'), (4,'IV'), (1,'I')]:
        while number >= val:
            out += rom
            number -= val
    return out
def simple_colors(name_str):
    """
    Returns a color based on the name of a method ``name_str``.
    """
    names =  ["StudentTFilter_GT", "StudentTFilter", "Huang_SSM", "Agamennoni_VBF", 
              "chang_RKF", "chang_ARKF", "Saerkkae_VBF", "roth_STF", "KalmanFilter"]
    colors = ['#800000', '#ff4500', '#000075', '#006400', 
              '#4363d8', '#42d4f4', '#00ff00', '#f58231', '#9a9a9a',]
    for i in range(len(names)):
        if names[i] in name_str:
            return colors[i]
    return "000000"

def make_tables(results, col_sequence=None, row_sequence=None, method2color=simple_colors,
                category2str=category_map, subcategory2str=subcategory2str, 
                rescale_fct=rescale_fct, larger_better=larger_better_only_likelihood,
                not_compare=not_compare_to_GT, number_fmt=nice_sci_nbr):
    """
    Creates latex code for a table containing the wanted subset of the computed data.

    :param results: The results dictionary returned by :meth:`~src.eval.simulate.collect_best_results`
    :param col_sequence: The sequence of columns in the table, see :attr:`~src.eval.simulate.simple_col_sequence`
    :param row_sequence: The sequence of columns in the table, see :attr:`~src.eval.simulate.simple_row_sequence`
    :param method2color: a map mapping method names to colors in which they are plotted, see :meth:`~src.eval.simulate.simple_colors`
    :param category2str: a map mapping category names to latex headers, see :attr:`~src.eval.simulate.category_map`
    :param subcategory2str: a map mapping subcategory names to latex subheaders, see :attr:`~src.eval.simulate.subcategory2str`
    :param rescale_fct: a map for additional rescaling of the data, takes a categroy string 
    :param larger_better: a map declaring for each category if larger data values signifies better performance, see :meth:`~src.eval.simulate.larger_better_only_likelihood`
    :param not_compare: whether a row should not be compared to find the best performance in each column, see :meth:`~src.eval.simulate.not_compare_to_GT`
    :param number_fmt: a number formatter that maps float values to latex code, see :meth:`~src.eval.simulate.nice_sci_nbr`
    """
    if row_sequence is None:
        row_sequence = [k for k in results.keys()]
    if col_sequence is None:
        col_sequence = sorted([(key, list(val.keys())) for key, val in results[row_sequence[0]].items() ])
    if type(category2str) == dict:
        category2strdict = category2str
        category2str = lambda key: category2strdict[key]
        def category2str(key):
            try:
                return category2strdict[key]
            except:
                print("problem with key: ", key)
                return category2strdict[key.replace("STF", "STS")]


    nbr_cols = sum([len(c[1]) for c in col_sequence])
    table_str = ""
    for i in range(len(row_sequence)):
        color = method2color(row_sequence[i]); color = color if color[0]!='#' else color[1:]
        table_str += "\\definecolor{RowColor"+str(i)+"}{HTML}{"+color+"}\n"      # define colors of the rows if colors are given
    table_str += "\\begin{array}{rc*{"+str(nbr_cols)+"}{|c}}\n\t"               # defining the columns of the table
    table_str += "\\multirow{2}{*}{} & \\multirow{2}{*}{Method} & "             # defining the supercategories in the first row
    table_str += " & ".join(["\\multicolumn{"+f"{len(col_sequence[j][1])}"+"}{|c}{ "+category2str(col_sequence[j][0])+" }"
                             for j in range(len(col_sequence)) if len(col_sequence[j][1]) > 0] ) + "\\\\\n\t"
    table_str += "& & "+ " & ".join([subcategory2str(key)              # defining the subcategories in the second row (like mean/min/25%...)
                                     for j in range(len(col_sequence)) for key in col_sequence[j][1] ])+ "\\\\\n"
    #for row in row_sequence:
    #    print(f"row: {row}")
    #    for col_key, col_vals in col_sequence:
    #        print(f" col: {col_key}")
    #        for col_val in col_vals:
    #            print(f"   {col_val}",end=" ")
    #            print(results[row][col_key][col_val])
            
    data = [[   results[row][col_key][col_val] * rescale_fct(col_key)
                for col_key, col_vals in col_sequence for col_val in col_vals] 
            for row in row_sequence]
    best_in_col = [ (np.nanargmax if larger_better(col_key) else np.nanargmin)(
                        [(np.NAN if not_compare(row) else results[row][col_key][col_val]) for row in row_sequence]) 
                   for col_key, col_vals in col_sequence for col_val in col_vals ]
    for i in range(len(row_sequence)):
            filterclass, filterkwargs = key2filterkwargs( find_str_key(row_sequence[i], results.keys()) )
            filtername_ltx = make_latex_filtername(filterclass, filterkwargs)
            table_str += "\t\t\\hline \\rule{0ex}{2.3ex} \\color{RowColor"+str(i)+"} \\text{"+ Roman(i+1) +"} & \\color{RowColor"+str(i)+"}" +filtername_ltx+" & "
            table_str += " & ".join([("\\bm{"+number_fmt(data[i][j])+"}" if i==best_in_col[j] else number_fmt(data[i][j])) for j in range(len(data[i]))]) 
            table_str += "\\\\\n"
        
    table_str += "\\end{array}"
    return table_str

def make_latex_filtername(filterclass, filter_kwargs={}, model_kwargs=None):
    """
    Produces a latex-frindly filtername for the filterclass
    """
    if (    filterclass == filters.basic.KalmanFilter 
            and 'gating' in filter_kwargs and filter_kwargs['gating'] < 1):
        return r"\text{gated Kalman Filter}"
    model_kwargs = {'T':1, 'seed':0} if model_kwargs is None else model_kwargs
    model, (x0, P0), (x,y,v,e,t) = simulate_Singer(**model_kwargs)
    try:
        filter = filterclass(model=model, mean=x0, covar=P0, current_time=0, **filter_kwargs)
        filtername = filter.label_long()
    except:
        filtername =  filterclass.label()
    old_filtername = "\\text{" + filtername +"}"
    nxt_d = old_filtername.find('$')
    if nxt_d > 0:
        ltx_filtername = old_filtername[:nxt_d]
        while nxt_d > 0:
            nxt_nxt_d = old_filtername.find('$', nxt_d+1)
            ltx_filtername += "}\\," + old_filtername[nxt_d+1:nxt_nxt_d] +"\\text{"
            nxt_d = old_filtername.find('$', nxt_nxt_d+1)
            ltx_filtername += old_filtername[nxt_nxt_d+1:nxt_d] if nxt_d > 0 else old_filtername[nxt_nxt_d+1:]
    else:
        ltx_filtername = old_filtername
    ltx_filtername = ltx_filtername.replace("%","\\%")
    return ltx_filtername


def show_latex(latex_string, packages=[], math=False):
    """
    A method that can render and display latex code in an ipython notebook.
    Note that this needs a working pdflatex command from the comand line.

    :param latex_string: the string to render
    :param packages: additional packages required for the compilation. Have to be present on your machine
    :param math: whether the code should be set in math mode
    """
    string = "\\documentclass[varwidth=\\maxdimen]{standalone}\n" 
    for p in packages:
        string += "\\usepackage{"+p+"}\n"
    string += "\\begin{document}\n"
    if math: 
        string+= "$$"+latex_string+"$$"
    string += "\n\n\\end{document}"
    with open("../data/figures/table.tex", "w") as f:
        f.write(string)
    try: # Compile the LaTeX file using pdflatex
        subprocess.run(["pdflatex", "-halt-on-error", "-output-directory", "../data/figures/", "../data/figures/table.tex"], 
                       stdout=subprocess.DEVNULL, 
                       stderr=subprocess.PIPE, check=True)
        # Clean up temporary files if compiled successfully
        subprocess.run(["rm", "../data/figures/table.tex", "../data/figures/table.aux", "../data/figures/table.log"])
        try:
            scale = (5,200) # upscale the rendered image from the pdf
            pixmap = fitz.open("../data/figures/table.pdf").load_page(0).get_pixmap(matrix=fitz.Matrix(scale[0], scale[1])) # transform the pdf file into a pixel map
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            image = image.resize((pixmap.width//scale[0], max(pixmap.height//scale[1],100)))
            image.save("../data/figures/table.png")
            display(image)
        except Exception:
            print("could display the table inline. Check 'data/figures/table.pdf' instead.")
    except subprocess.CalledProcessError as e:
        print("Requires the presence of the 'pdflatex' command and the latex packages 'amsmath', 'multirow', 'bm' and 'xcolor' on your machine")
        print("The detailed error is: ")
        print("   ", e, end="\n\n") # Error occurred, print error message
        if os.path.exists("../data/figures/table.log"):
            with open("../data/figures/table.log") as f:
                while True:
                    peek = f.readline()
                    if not peek: # EOF
                        break
                    if peek[0] == '!':
                        print("with Latex's error: \n\n", peek)
                        print("".join(f.readlines()), end="\n\n")
                        break
            subprocess.run(["rm", "../data/figures/table.aux", "../data/figures/table.log"]) # Clean up temporary files
            print()
        raise e

def data2bxplot(data_dict, scale=1):
    """
    Creates a boxplot data dictionary consistent with 
    `matplotlib.axes.Axes.bxp <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.bxp.html>`_ , 
    from a category subdirectory of the result directory.
    """
    return {'label': '', 'fliers': [data_dict["perc_0"] *scale, data_dict["perc_1"]  *scale,
                                    data_dict["perc_99"]*scale, data_dict["perc_100"]*scale],
            'whislo': data_dict["perc_5"] *scale, 'whishi': data_dict["perc_95"]*scale,
            'q1':     data_dict["perc_25"]*scale, 'q3':     data_dict["perc_75"]*scale,
            'mean':   data_dict["mean"]   *scale, 'med':    data_dict["perc_50"]*scale,}
def category2linestyle(category):
    """
    creates a linestyle for drawn lines based on the category string
    """
    if "RTS" in category:
        return "--"
    return "-"
def make_boxplot(results, ax, categories, row_sequence=None,
                 data2bxplot=data2bxplot, maxboxoutlier=2,
                 method2color=simple_colors, category2linestyle=category2linestyle,
                 category2str=category_map, rescale_fct=rescale_fct):
    """
    Creates the boxplots associated with :meth:`~src.eval.simulate.make_tables` into the given axes.

    :param results: The results dictionary returned by :meth:`~src.eval.simulate.collect_best_results`
    :param ax: The axes in which to draw the boxplots
    :param categories: which categories should be plotted in the axes. If multiple given, 
            then the boxplots are clustered by common method
    :param row_sequence: The sequence of methods to plot left to right, see :attr:`~src.eval.simulate.simple_row_sequence`
    :param data2bxplot: see :meth:`~src.eval.simulate.data2bxplot`
    :param maxboxoutlier: controls how far fliers can be before they are discarded for considering axes limits
    :param method2color: a map mapping method names to colors in which they are plotted, see :meth:`~src.eval.simulate.simple_colors`
    :param category2linestyle: a map mapping category names to linestyles, see :attr:`~src.eval.simulate.category2linestyle`
    :param category2str: a map mapping category names to latex headers, see :attr:`~src.eval.simulate.category_map`
    :param rescale_fct: a map for additional rescaling of the data, takes a category string 
    """
    if type(categories) != list:
        categories = [categories]
    lcats = len(categories)
    if type(category2str) == dict:
        ax.set_title(category2str[categories[0]].replace("\\text{","").replace("}",""))
    else:
        ax.set_title(category2str(categories[0]).replace("\\text{","").replace("}",""))
    if row_sequence is None:
        row_sequence = [k for k in results.keys()]
    
    pos = 0; box_datas = []
    for i in range(len(row_sequence)):
        for category in categories:
            box_data = data2bxplot(data_dict = results[row_sequence[i]][category], 
                                   scale = rescale_fct(category) )
            box = ax.bxp([box_data], positions=[pos], widths=[0.8])
            box_datas.append(box_data)
            pos += 1
            color = method2color(row_sequence[i])
            for desc in ['whiskers', 'caps', 'boxes', 'medians']:
                for Line in box[desc]:
                    Line.set(color=color, linestyle=category2linestyle(category), linewidth=0.5)
            for Line in box['fliers']:
                Line.set(markerfacecolor=color, markeredgecolor=color, markersize=2)
    # adjusting ylims so that whiskers and very bad methods do not contort the plot
    perc_Q1 = np.percentile(a=[b['q1'] for b in box_datas], q=25)
    perc_Q3 = np.percentile(a=[b['q3'] for b in box_datas], q=75)
    median_box_span = np.median([b['q3']-b['q1'] for b in box_datas])
    # making sure that the boxes take up, on average, at most 1/3 of the space 
    box_ylim = (perc_Q1-median_box_span, perc_Q3+median_box_span)
    # make sure that at least all medians are visible as long as they are not increasing the 
    # height of the plot by more than 2 times
    median_max, median_min = (sum(box_ylim)/2,sum(box_ylim)/2)
    for box in box_datas:
        box_span = (box_ylim[1]-box_ylim[0])
        dist_to_box_ylim = max(box_ylim[0]-box['med'], box['med']-box_ylim[1])/box_span
        if dist_to_box_ylim < maxboxoutlier:
            median_max = max(median_max, box['med']+0.1*box_span)
            median_min = min(median_min, box['med']-0.1*box_span)
    # ... as long as they are not increasing the 
    ax.set_ylim( min(box_ylim[0],median_min), max(box_ylim[1],median_max) )
    # add labels to each method via Roman numerals for each method (each row in the table)
    # if multiple categories are plotted for each method, only add one label in the middle
    ax.set_xticks([lcats*i+(lcats-1)/2 for i in range(len(row_sequence))],
                  [Roman(i+1) for i in range(len(row_sequence))])
    ticklabels = ax.get_xticklabels()
    for i in range(len(row_sequence)):
        ticklabels[i].set_color(method2color(row_sequence[i]))

