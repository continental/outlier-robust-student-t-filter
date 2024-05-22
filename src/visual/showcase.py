# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Implements an interactive graphical interface to test the different methods dynamically.

See the file ```tests/showcase.ipynb``.
"""

import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets
import os
from IPython import display
from .. import filters, eval, utils

print("-   Loading File 'showcase.py'")

rowlayout = widgets.Layout(margin='0px 10px 10px 0px', padding='5px 5px 5px 5px')
boxlayout = widgets.Layout(margin='0px 10px 10px 0px', padding='5px 5px 5px 5px')
def organise_widgets_in_grid(widget_list, rowlayout=rowlayout, boxlayout=boxlayout, rows=None):
    r"""
    Organises the given list of :math:`k` widgets into a table format with ``rows`` number of rows 
    if not ``None``, otherwise with :math:`\sqrt{\lfloor k \rfloor}` number of rows.
    
    The result will be a ``ipywidgets.VBox`` (vertically stacked box) of ``ipywidgets.HBox`` 's 
    (horizontally stacked box of widgets).

    :param widget_list: A list of widgets to organise into a table
    :param rowlayout: The specific layout used for the individual rows, defaults to a default row layout
    :param boxlayout: The specific layout used for the overall table, defaults to a default box layout
    :param rows: The specific number of rows in the table, if known, defaults to None
    :return: A widget encompassing the given widgets as a table
    """

    k = len(widget_list)
    rows = ((1 if k < 4 else 2) if k < 9 else (3 if k < 16 else 4)) if rows is None else rows
    cols = int(np.ceil(k/rows))
    rows = int(np.ceil(k/cols)) if cols > 0 else rows # if e.g. k=6 and rows was given as 4, we will fill 3 rows of 2
    rowboxes = []
    for i in range(rows):
        rowboxes.append(widgets.HBox(widget_list[i*cols:(i+1)*cols], layout=rowlayout))
    return widgets.VBox(rowboxes, layout=boxlayout) if rows > 1 else rowboxes[0]


ExtendedSingerModelParameters = [
    ("alpha",   widgets.BoundedFloatText, {"description":"α", "value":0.5, "min":0.01,
                                           "max":1, "step":0.01, "layout": widgets.Layout(width='175px')}),
    ("beta",    widgets.BoundedFloatText, {"description":"β",  "value":0.25, "min":0.0,
                                           "max":1.,  "step":0.01, "layout": widgets.Layout(width='175px')}),
    ("sigma2",  widgets.BoundedFloatText, {"description":"Q's σ^2", "value":1, "min":0.1,
                                           "max":10, "step":0.1,# "continuous_update": False,
                                           "layout": widgets.Layout(width='175px')}),
    ("Nbr_sens",widgets.BoundedIntText,   {"description":"sensor Nbr", "value":1, "min":1, "max":5,
                                           "step":1, "layout": widgets.Layout(width='175px')}),
    ("R_inhom", widgets.BoundedFloatText, {"description":"R deform", "value":1, "min":1, "max":30,
                                           "step":0.1, "layout": widgets.Layout(width='175px')}),
    ("R_phase", widgets.BoundedFloatText, {"description":"R phase", "value":0, "min":-0.25, "max":0.25,
                                           "step":0.01, "layout": widgets.Layout(width='175px')}),
    ("sigmaR",  widgets.BoundedFloatText, {"description":"R's σ^2", "value":0.25, "min":0.001,
                                           "max":2, "step":0.001, "layout": widgets.Layout(width='175px')}),
    ("R_distr", widgets.Dropdown,  {"description": "Obs. dist.", "value":"Student-T",
                                    "options":["Student-T","Normal"],
                                    "disabled": False, "layout": widgets.Layout(width='175px')}),
    ("R_nu",    widgets.BoundedFloatText, {"description":"R's ν", "value":1, "min":0.1, "max":50,
                                           "step":0.1, "layout": widgets.Layout(width='175px')}),
    ("T",       widgets.BoundedIntText,   {"description":"Nbr steps", "value":500, "min":1,
                                           "max":5000, "step":1, "layout": widgets.Layout(width='175px')}),
    ("dt",      widgets.BoundedFloatText, {"description":"Δt", "value":0.2, "min":0.01,
                                           "max":2, "step":0.01, "layout": widgets.Layout(width='175px')}),
    ("skip",    widgets.BoundedIntText,   {"description":"skip obs", "value":0, "min":0, "max":5,
                                           "step":1, "layout": widgets.Layout(width='175px')}),
]
"""
Defines the necessary parameters for the :class:`ExtendedSingerModelManager`. 

This is only a recommendation and can be adapted if necessary.

It consists of a list of tuples ``(name, widgetClass, initDict)``. ``name`` will make the widget 
accessible as ``ModelManager.name``, ``widgetClass`` defines the type of widget this will be, i.e.
``ipywidgets.BoundedFloatText``, and ``initDict`` is a dictionary with all the parameters to 
successfully create a widget of type ``widgetClass``
"""

class ExtendedSingerModelManager():
    """
    Manages the central simulation and provides the simulation data.
    Specific to the :doc:`src.filters.models.ExtendedSingerModel`.
    """

    plot : callable(object) 
    """
    A function called for each replot. Takes a single argument that has no effect.
    """
    reseeding : callable(object) 
    """
    A function called to find a new seed on which all simulations are based on. 
    Takes a single argument that has no effect.
    """

    
    def __init__(self, trajectory_ax, Model_params, seed=None, rows=None):
        """
        Initialises the model manager.

        :param trajectory_ax: a ``matplotlib.pyplot.axes`` to where the simulated ground truth 
                trajectories and sensor data's are plotted
        :param Model_params: A list of tuples describing the widgets controlling the simulation 
                parameters. See :attr:`ExtendedSingerModelParameters`.
        :param seed: The initial seed for the simulation. Can be a integer or a numpy.rand.Generator. 
                Defaults to None, in which case a random initial seed is taken.
        :param rows: In how many rows the widgets of the params should be presented, see :func:`organise_widgets_in_grid`.
        """

        self.init_seed = np.random.default_rng(seed).integers(2147483647)
        self.update_axes = True
        self.errticklocs, self.errticklabels = ([],[])
        self.figure_dim = trajectory_ax.get_figure().get_size_inches()
        
        def plot(change=-1):
            #print("REPLOTTING MODEL")
            model, (x0, P0), (x,y,v,e,t) = eval.simulate.simulate_Singer(seed=self.init_seed, init_state = np.zeros((6,)),
                        alpha=self.alpha.value, beta=self.beta.value, sigma2=self.sigma2.value, Nbr_sens=self.Nbr_sens.value,
                        R_inhom=self.R_inhom.value, R_phase=self.R_phase.value, sigmaR=self.sigmaR.value, R_distr=self.R_distr.value,
                        R_nu=self.R_nu.value, T=self.T.value, dt=self.dt.value
                    )
            self.model = model
            self.x0 = x0
            self.P0 = P0
            self.x, self.y, self.v, self.e = (x,y,v,e)

            self.GT_data.set(  data=(self.x[:, 0],      self.x[:, 1]))
            for i in range(self.Nbr_sens.value):
                self.meas_data[i].set(data=(self.y[:, 2*i], self.y[:, 2*i+1]))
            for i in range(self.Nbr_sens.value,len(self.meas_data)):
                self.meas_data[i].set(data=([],[]))

            predict = [self.x[-1]]
            for i in range(self.predict_cnt.value):
                predict.append( self.model._F(self.dt.value*(self.skip.value+1)) @ predict[-1] )
            predict = np.asarray(predict)
            self.predict_data.set(  data=(predict[:, 0],  predict[:, 1]))

            if self.update_axes:
                xin, xax = (min(np.min(self.x[:, 0]),np.min(predict[:,0])), max(np.max(self.x[:, 0]),np.max(predict[:,0])))
                yin, yax = (min(np.min(self.x[:, 1]),np.min(predict[:,1])), max(np.max(self.x[:, 1]),np.max(predict[:,1])))
                fac = 0.05 + min(1,10/self.T.value)
                trajectory_ax.set_xlim((xin-fac*(xax-xin), xax+fac*(xax-xin)))
                trajectory_ax.set_ylim((yin-fac*(yax-yin), yax+fac*(yax-yin)))
            
            #trajectory_ax.draw_artist(self.GT_data)
            #trajectory_ax.draw_artist(self.meas_data)

            for listener in self.listeners:
                listener.update()

            #trajectory_ax.figure.canvas.blit()
            #trajectory_ax.figure.canvas.flush_events()
            #print(" --- DONE REPLOTING")
        self.plot = plot

        widget_list = []
        for name, widget, widget_params in Model_params:
            new_widget = widget(**widget_params)
            new_widget.observe(self.plot, 'value')
            widget_list.append(new_widget)
            self.__dict__[name] = new_widget
        def hide_Rnu(change):
            new_visible = (change.new == "Student-T")
            try:
                self.R_nu.layout.visibility = ("visible" if new_visible else "hidden")
            except:
                print("parameter 'R_nu' does not exist")
        self.hide_Rnu = hide_Rnu
        try:
            self.R_distr.observe(self.hide_Rnu, 'value')
        except:
            print("parameter 'R_distr' does not exist")
        
        def reseeding(change):
            self.init_seed = np.random.default_rng(self.init_seed).integers(2147483647)
            print("new seed: ", self.init_seed)
            self.plot()
        self.reseeding = reseeding
        self.new_seed = widgets.Button(**{"description":"reseed", "disabled":False, "icon":'dice' })
        self.new_seed.on_click(self.reseeding)
        self.predict_cnt = widgets.BoundedIntText(description="predict", value=0, min=0, max=100, step=1, layout=widgets.Layout(width="145px"))
        self.predict_cnt.observe(self.plot, 'value')

        self.param_box = organise_widgets_in_grid(widget_list=widget_list, rows=rows)
        self.param_box = widgets.HBox([widgets.VBox([self.new_seed, self.predict_cnt]), self.param_box], layout={"justify-content":"flex-start", "align_items":"center"})
        
        #print("Created all Model Widgets")

        self.listeners = []

        self.meas_data    = [trajectory_ax.plot([], [], "o", color=["blue", "orange", "red", "green", "black"][i], label="observations", ms=1, alpha=0.5)[0] for i in range(5)]
        self.GT_data      = trajectory_ax.plot([], [], "k-", label="ground truth")[0]
        self.predict_data = trajectory_ax.plot([], [], "k--", label="ground truth\nprediction")[0]
        self.Rscale = None
        self.plot()

    def show_params(self):
        """
        Shows the widgets below the current cell. Only works in notebooks.
        """
        display.display(self.param_box)

    def add_listener(self, listener):
        """
        Adds a listener who will be updated whenever the simulation data changes. 
        That is ``listener.update()`` will be called for each listener whenever ``self.plot()`` is
        executed.
        """
        self.listeners.append(listener)
    
    def get_observations(self):
        """
        Gives the current observation data.
        """
        return (self.y, np.arange(1,self.y.shape[0]+1) * self.dt.value * (self.skip.value+1))
    def get_groundTruth(self):
        """
        Returns the ground truth agent positions.
        """
        return self.x[::self.skip.value+1,:,:]
    def get_noise(self):
        """
        Returns the measurement noise.
        """
        return self.e[self.skip.value::self.skip.value+1,:,:]





FilterList = [
    [filters.proposed.StudentTFilter_GT, # 0
        {"color": "#800000", "linestyle": "-", "alpha":0.75},
        [   ("nu", widgets.BoundedFloatText, {"description":"ν", "value":1, "min":0.1, "max":50, 
                                              "step":0.01, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {"GTx": model.get_groundTruth(), "GTe":model.get_noise()} 
    ],
    [filters.proposed.StudentTFilter, # 1
        {"color": "#ff4500", "linestyle": "-", "alpha":0.75},
        [   ("nu", widgets.BoundedFloatText, {"description":"ν", "value":3.26, "min":0.1, "max":50,
                                              "step":0.01, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {} 
    ],
    [filters.proposed.StudentTFilter_analytic, # 2
        {"color": "#f9bf10", "linestyle": "-", "alpha":0.75},
        [   ("nu", widgets.BoundedFloatText, {"description":"ν", "value":3.26, "min":0.1, "max":50,
                                              "step":0.01, "layout": widgets.Layout(width='175px')}),
            ("N", widgets.BoundedIntText, {"description":"N", "value":50000, "min":5000, "max":5000000,
                                           "step":1, "layout": widgets.Layout(width='175px')}),
            ("M", widgets.BoundedIntText, {"description":"M", "value":20, "min":20, "max":1000,
                                           "step":1, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {} 
    ],
    [filters.proposed.StudentTFilter_Newton, # 3
        {"color": "#ff4500", "linestyle": "--", "alpha":0.75},
        [   ("nu", widgets.BoundedFloatText, {"description":"ν", "value":3.26, "min":0.1, "max":50, 
                                              "step":0.01, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {} 
    ],
    [filters.proposed.StudentTFilter_SF, # 4
        {"color": "#e63e00", "linestyle": "-", "alpha":0.75},
        [   ("nu", widgets.BoundedFloatText, {"description":"ν", "value":3.26, "min":0.1, "max":50, 
                                              "step":0.01, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {"comp": [2]*model.Nbr_sens.value} 
    ],
    [filters.robust.Huang_SSM, # 5
        {"color": "#000075", "linestyle": "-", "alpha":0.5},
        [   ("separate", widgets.ToggleButton, {"description":"separate Algo", "value":True,
                                                "layout": widgets.Layout(width='170px')}),
            ("SSM", widgets.Dropdown,  {"description": "SSM", "value":"log", 
                                        "options":['log', 'exp', 'sqrt', 'sq', 'lin'],
                                        "layout": widgets.Layout(width='175px')}),
            ("nu_SSM", widgets.BoundedFloatText, {"description":"ν SSM", "value":3.79, "min":0.1, "max":10,
                                                  "step":0.01, "layout": widgets.Layout(width='175px')}),
            ("process_non_gaussian", widgets.ToggleButton, {"description":"process outliers", "value":False,
                                                            "layout": widgets.Layout(width='170px')}),
            ("sigma_SSM",  widgets.BoundedFloatText, {"description":"σ SSM", "value":1, "min":0.1, "max":10,
                                                      "step":0.01, "layout": widgets.Layout(width='175px')}),
            ("gating", widgets.BoundedFloatText, {"description":"gating χ", "value":1, "min":0.95, "max":1,
                                                  "step":0.001, "layout": widgets.Layout(width='175px')}),
        ],
        lambda model: {} 
    ],
    [filters.robust.chang_RKF, # 6
        {"color": "#4363d8", "linestyle": "-", "alpha":0.5},
        [   ("alpha", widgets.BoundedFloatText, {"description":"α", "value":0.064, "min":0.001, "max":0.1,
                                                 "step":0.001, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {} 
    ],
    [filters.proposed.chang_RKF_SF, # 7
        {"color": "#4363d8", "linestyle": "--", "alpha":0.5},
        [   ("alpha", widgets.BoundedFloatText, {"description":"α", "value":0.064, "min":0.001, "max":0.1,
                                                 "step":0.001, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {"comp": [2]*model.Nbr_sens.value} 
    ],
    [filters.robust.chang_ARKF, # 8
        {"color": "#42d4f4", "linestyle": "-", "alpha":0.5}, 
        [   ("alpha", widgets.BoundedFloatText, {"description":"α", "value":0.021, "min":0.001, "max":0.1,
                                                 "step":0.001, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {} 
    ],
    [filters.robust.Agamennoni_VBF, # 9
        {"color": "#006400", "linestyle": "-", "alpha":0.5},
        [   ("nu", widgets.BoundedFloatText, {"description":"ν", "value":12.15, "min":0.1, "max":100,
                                              "step":0.01, "layout": widgets.Layout(width='175px')}),
            ("gating", widgets.BoundedFloatText, {"description":"gating χ", "value":1, "min":0.95, "max":1,
                                                  "step":0.001, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {} 
    ],
    [filters.robust.Saerkkae_VBF, # 10
        {"color": "#00ff00", "linestyle": "-", "alpha":0.5},
        [   ("rho", widgets.BoundedFloatText, {"description":"ρ", "value":0.001, "min":0, "max":1,
                                               "step":0.001, "layout": widgets.Layout(width='175px')}),
            ("gating", widgets.BoundedFloatText, {"description":"gating χ", "value":1, "min":0.95, "max":1,
                                                  "step":0.001, "layout": widgets.Layout(width='175px')})
        ],
        lambda model: {} 
    ],
    [filters.robust.roth_STF, # 11
        {"color": "#f58231", "linestyle": "-"},
        [   ("state_nu", widgets.BoundedFloatText, {"description":"state ν", "value":5.54, "min":0.1, "max":20,
                                                    "step":0.01, "layout": widgets.Layout(width='175px')}),
            ("process_gamma", widgets.BoundedFloatText, {"description":"process ν", "value":12.72, "min":0.1,
                                                         "layout": widgets.Layout(width='175px'),
                                                         "max":20, "step":0.01, }),
            ("obs_delta", widgets.BoundedFloatText, {"description":"observation ν", "value":2.27, "min":0.1,
                                                    "layout": widgets.Layout(width='175px'),
                                                    "max":20, "step":0.01}),
        ],
        lambda model: {} 
    ],
    [filters.basic.KalmanFilter, # 12
        {"color": "#9a9a9a", "linestyle": "-", "alpha":0.5},
        [   ("gating", widgets.BoundedFloatText, {"description":"gating χ", "value":0.997, "min":0.95, "max":1,
                                                  "step":0.001, "layout": widgets.Layout(width='175px')}),
        ],
        lambda model: {} 
    ],
]
"""
Lists the filters which shall be investigated.

This is only a recommendation and can be adapted if necessary.

Each list element contains a tuple of ``(FilterClass, styleDict, paramList, modelParamFkt)``. 
``FilterClass`` is the class of the corresponding filter. 
``styleDict`` is a dictionary that will be used in the pyplot functions - e.g., as in ``plt.plot(data, **styleDict)``.
``paramList`` is a list with widgets similarly to :attr:`ExtendedSingerModelParameters` but will be 
used for :class:`FilterManager`. 
``modelParamFkt`` is a function that extracts additional ground-truth data from the model to provide the filter.
Returns a dictionary with additional arguments for the filter's __init__ function.

In more detail, ``paramList`` is a list of tuples 
``(name, widgetClass, initDict)``. ``name`` will make the widget accessible as 
``FilterManager.name`` - Important: the ``names`` have to correspond to keyword parameters of the 
corresponding ``FilterClass.__init__()`` method.
``widgetClass`` defines the type of widget this will be, i.e.
``ipywidgets.BoundedFloatText``. And ``initDict`` is a dictionary with all the parameters to 
successfully create a widget of type ``widgetClass``
"""


widget_boxlayout = widgets.Layout(
    #display='flex',
    #justify_content='flex-start',
    #flex_flow='column nowrap',
    align_items='stretch',
    width='100%'
)
class FilterManager():
    """
    Manages the widgets controlling parameters of a single filter method.
    It also interfaces with the simulations managed by a :class:`ExtendedSingerModelManager` to get
    the simulation data.
    """

    def __init__(self, filterclass, style, params, paramFct, index, modelmanager, emptyax, trajectory_ax, 
                 time_ax, error_ax, likelihood_ax, likelihood_pos_ax, iter_ax=None, boxlayout=widget_boxlayout):
        """
        Initialises the FilterManager

        :param filterclass: Class of the filter
        :param style: Linestyle for how the filter should be plotted to distinguish the different methods.
        :param params: A list of tuples ``(name, widgetClass, initDict)`` to create ipywidgets managing
                the parameters used in initialising the filters, see :attr:`FilterList`.
        :param paramFct: a function that extracts additional ground-truth data from the model to provide the filter.
                Returns a dictionary with additional arguments for the filter's __init__ function.
        :param index: The index of the filter, e.g., the 5th created filter. Used for regulating the 
                order in which the filters are ordered in the boxplots etc.
        :param modelmanager: The instance of :class:`ExtendedSingerModelManager` managing the simulation data this
                filter is listening to.
        :param emptyax: A ``matplotlib.pyplot.axes`` which should not be plotted (in detail) since
                it is only used to create new ``boxplots`` or ``violinplots`` whose vertices are then
                copied to update the internal figures.
        :param trajectory_ax: A ``matplotlib.pyplot.axes`` illustrating the 2d filtered trajectory.
        :param time_ax: A ``matplotlib.pyplot.axes`` illustrating the needed processing time for the filter,
                The solid part of the bar plots represents the time spent only in the filtering method,
                and the transparent part the remaining post- and preprocessing time (i.e., getting information 
                of the dynamical model).
        :param error_ax: A ``matplotlib.pyplot.axes`` illustrating the distribution over the Euclidean 
                error between the ground truth positions and filtered mean positions (so no velocities or
                accelerations).
        :param likelihood_ax: A ``matplotlib.pyplot.axes`` illustrating the distribution over the 
                likelihoods that the ground truth was created by the filtered estimated distribution.
        :param likelihood_pos_ax: A ``matplotlib.pyplot.axes`` illustrating the distribution over the 
                likelihoods that the ground truth positions were created by the filtered estimated 
                distribution marginalised onto the positions.
        :param iter_ax: A ``matplotlib.pyplot.axes`` illustrating the distribution over needed iterations
                (for iterating methods).
        :param boxlayout: The layout of the parameter widget box, defaults to a simple layout.
        """
        self.filterclass = filterclass
        self.label = filterclass.__name__.replace("_", " ").replace(",","\n")
        self.label_short = filterclass.label()
        self.paramFct = paramFct
        self.modelmanager = modelmanager
        modelmanager.add_listener(self)
        self.emptyax = emptyax
        self.trajectory_ax = trajectory_ax
        self.time_ax = time_ax
        self.error_ax = error_ax
        self.likelihood_ax = likelihood_ax
        self.likelihood_pos_ax = likelihood_pos_ax
        self.iter_ax = iter_ax
        self.index = index

        self.need_redraw = True
        # main plot
        self.plot_data = trajectory_ax.plot([], [], label=self.label, **style)[0]
        self.plot_maxerror = trajectory_ax.plot([], [], "r--", alpha=0.75, zorder = 100)[0]
        self.plot_maxerror_ind = trajectory_ax.plot([], [], "ro", alpha=0.5, zorder = 100)[0]
        self.plot_data.set_visible(False)
        self.plot_maxerror.set_visible(False)
        self.plot_maxerror_ind.set_visible(False)
        self.predict_data = trajectory_ax.plot([], [], **style)[0]
        self.predict_data.set_linestyle("--")
        self.predict_data.set_visible(False)
        # time plot
        no_alpha_style = {key:val for key,val in style.items() if key!="alpha"}
        self.time_bar = time_ax.bar(x=index, height=0, width=0.8, **no_alpha_style)[0]
        self.rtime_bar = time_ax.bar(x=index, height=0, width=0.8, **no_alpha_style, alpha=0.5)[0]#
        self.stime = time_ax.plot([index-0.4, index+0.4], [0,0], color="black", solid_capstyle="butt")[0]
        self.time_bar.set_visible(False)
        self.rtime_bar.set_visible(False)
        self.stime.set_visible(False)
        xmin, xmax = time_ax.get_xlim()
        dmin = max(index/10, 0.5)
        time_ax.set_xlim( min(xmin,-dmin), max(xmax, index+dmin) )
        time_ax.set_ylim()
        # error plot
        self.error_box = error_ax.boxplot([], positions=[index])
        self.error_vio = error_ax.violinplot([0], positions=[index], showextrema=False)["bodies"][0]
        self.likeli_box = likelihood_ax.boxplot([], positions=[index])
        self.likeli_vio = likelihood_ax.violinplot([0], positions=[index], showextrema=False)["bodies"][0]
        self.likeli_pos_box = likelihood_pos_ax.boxplot([], positions=[index])
        self.likeli_pos_vio = likelihood_pos_ax.violinplot([0], positions=[index], showextrema=False)["bodies"][0]
        self.plot_iter = self.iter_ax is not None
        if self.plot_iter:
            self.iter_box = iter_ax.boxplot([], positions=[index])
        
        modelmanager.errticklocs.append(index)
        modelmanager.errticklabels.append("")
        self.errticksidx = len(modelmanager.errticklocs)-1
        self.error_ax.set_xticks(ticks=modelmanager.errticklocs, labels=modelmanager.errticklabels, rotation="vertical")
        self.likelihood_ax.set_xticks(ticks=[], labels=[], rotation="vertical")
        self.likelihood_pos_ax.set_xticks(ticks=modelmanager.errticklocs, labels=modelmanager.errticklabels, rotation="vertical")
        self.iter_ax.set_xticks(ticks=modelmanager.errticklocs, labels=modelmanager.errticklabels, rotation="vertical")
        for desc in ['whiskers', 'caps', 'boxes', 'medians']:
            for box in [self.error_box[desc], self.likeli_box[desc], self.likeli_pos_box[desc]]+(
                    [self.iter_box[desc]] if self.iter_ax is not None else []   ):
                for Line in box:
                    Line.set(**no_alpha_style)
                    Line.set_visible(False)
        for vio in [self.error_vio, self.likeli_vio, self.likeli_pos_vio]:
            vio.set(**no_alpha_style)
            vio.set(visible=False)

        # Show/ hide figure
        self.visibility = False
        def switch_visibility(button=None):
            new_visibility = not self.visibility
            #print("changing ", self.label," visibility to ", new_visibility)
            self.vis_button.icon = 'eye' if new_visibility else 'eye-slash'
            if new_visibility and self.need_redraw:
                self.compute()

            self.plot_data.set_visible(new_visibility)
            self.plot_maxerror.set_visible(new_visibility and self.show_max_error)
            self.plot_maxerror_ind.set_visible(new_visibility and self.show_max_error)
            self.predict_data.set_visible(new_visibility)
            self.modelmanager.errticklabels[self.errticksidx] = self.label_short if new_visibility else ""
            self.error_ax.set_xticks(ticks=self.modelmanager.errticklocs, labels=self.modelmanager.errticklabels, rotation="vertical")
            self.likelihood_pos_ax.set_xticks(ticks=self.modelmanager.errticklocs, labels=self.modelmanager.errticklabels, rotation="vertical")
            if self.plot_iter:
                iterLabel = self.iter_ax.get_xticklabels()
                iterLabel[self.index] = f"{self.label_short} ({int(self.itersum):d})" if new_visibility else ""
                self.iter_ax.set_xticks(ticks=self.modelmanager.errticklocs, labels=iterLabel, rotation="vertical")
                fig = self.iter_ax.get_figure()
                figure_dim = self.modelmanager.figure_dim
                label_length = max([label.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).height for label in self.iter_ax.get_xticklabels()])
                trajectory_label_length  = self.trajectory_ax.get_xticklabels()[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted()).height
                fig.set_size_inches(figure_dim[0], figure_dim[1]+max(0,label_length-2*trajectory_label_length))

            #computation time bar data
            self.time_bar.set_visible(new_visibility)
            self.rtime_bar.set_visible(new_visibility)
            self.stime.set_visible(new_visibility)
            for desc in ['whiskers', 'caps', 'boxes', 'medians']:
                for box in [self.error_box[desc], self.likeli_box[desc], self.likeli_pos_box[desc]]+(
                        [self.iter_box[desc]] if self.iter_ax is not None else []   ):
                    for Line in box:
                        Line.set_visible(new_visibility)
            for vio in [self.error_vio, self.likeli_vio, self.likeli_pos_vio]:
                vio.set(visible=new_visibility)
            
            self.visibility = new_visibility
            #print(" - successfully changed visibility!")
        
        def show_max_error_call(button=None):
            self.show_max_error = not self.show_max_error
            self.err_button.icon = 'eye' if self.show_max_error else 'eye-slash'
            self.plot_maxerror.set_visible(self.visibility and self.show_max_error)
            self.plot_maxerror_ind.set_visible(self.visibility and self.show_max_error)


        self.switch_visibility = switch_visibility
        self.show_max_error_call = show_max_error_call

        self.vis_button = widgets.Button(description=self.label, disabled=False, button_style='',
                                         icon='eye-slash', 
                                         layout=widgets.Layout(width='175px') )
        self.vis_button.on_click(self.switch_visibility)
        self.show_max_error = False
        self.err_button = widgets.Button(description="max error", disabled=False, button_style='',
                                        tooltip='show where the greatest discrepancy between filtered estimate and ground truth is (Positions only). Indicated by a red dot and dashed line', icon='eye-slash', 
                                        layout=widgets.Layout(width='100px') )
        self.err_button.on_click(self.show_max_error_call)
        self.smoother = widgets.Dropdown(description="Smoother: ", value = "None", 
                                         options = ["None", "Kalman", "StudentT"], disabled = False, 
                                         layout = widgets.Layout(width='175px'))
        
        # Remaining parameters
        def compute(val=-1):
            #print("compute ", self.label)
            params = {name: widget.value for name, widget in self.widget_dict.items()}
            #print("  with params: ", params)
            self.filter = self.filterclass(model=self.modelmanager.model, mean=np.zeros((6,)), 
                                      covar=self.modelmanager.P0, 
                                      current_time=0, exp_hist_len=self.modelmanager.T.value, 
                                      **params, **self.paramFct(self.modelmanager))
            obs, times = self.modelmanager.get_observations()
            self.filter.filter(obs=obs, times=times)
            est, times = self.filter.get_all_state_distr()
            hist       = self.filter.get_history()
            comp_time = hist["comp_time_filter"].sum()
            rest_time = 0 # hist["comp_time_model"].sum() + hist["comp_time_rest"].sum()

            if self.smoother.value != "None":
                if self.smoother.value == "Kalman":
                    smoother = filters.basic.KalmanSmoother(self.filter)
                elif self.smoother.value == "StudentT":
                    smoother = filters.proposed.StudentTSmoother(self.filter)
                est, times, smoother_hist = smoother.smoothen()
                comp_time_smoother = smoother_hist["comp_time_smoother"].sum()
            else:
                comp_time_smoother = 0


            estx, esty = (est.mean()[:, 0, 0], est.mean()[:,1,0])
            
            self.plot_data.set_data(estx, esty)

            predict = [est.mean()[-1]]
            for i in range(self.modelmanager.predict_cnt.value):
                predict.append( self.modelmanager.model._F(self.modelmanager.dt.value*(self.modelmanager.skip.value+1)) @ predict[-1] )
            predict = np.asarray(predict)
            self.predict_data.set(  data=(predict[:, 0],  predict[:, 1]))
            #self.trajectory_ax.draw_artist(self.plot_data)
            #trajectory_ax.figure.canvas.blit()
            #trajectory_ax.figure.canvas.flush_events()
            
            self.time_bar.set_height( comp_time )
            self.rtime_bar.set_y( comp_time )
            self.rtime_bar.set_height( rest_time )
            self.stime.set_ydata( [comp_time_smoother,comp_time_smoother] )
            #time_ax.draw_artist(self.time_bar)
            #time_ax.draw_artist(self.rtime_bar)
            #time_ax.figure.canvas.blit()
            #time_ax.figure.canvas.flush_events()

            GT = self.modelmanager.get_groundTruth()
            error = GT - est.mean()[:]
            ignore_first = 5 # initially the ground truth is given the methods, to prevent these from badly impacting the distributions we will ignore them
            metric = np.sqrt(error[ignore_first:,0,0]**2 + error[ignore_first:,1,0]**2)
            max_dist = np.argmax(metric)+ignore_first
            self.plot_maxerror.set_data([estx[max_dist], GT[max_dist,0,0]],[esty[max_dist], GT[max_dist,1,0]])
            self.plot_maxerror_ind.set_data([estx[max_dist]],[esty[max_dist]])
            likeli = est.logpdf(GT)[ignore_first:]
            likeli_pos = est.marginal([0,1]).logpdf(GT[:,[0,1],:])[ignore_first:]
            self.plot_iter = (self.iter_ax is not None) and ("iterations" in hist.keys())
            if self.plot_iter:
                iters = hist["iterations"][1:,0]
                self.itersum = sum(iters)
                if self.visibility:
                    iterLabel = self.iter_ax.get_xticklabels()
                    iterLabel[self.index] = f"{self.label_short} ({int(self.itersum):d})"
                    self.iter_ax.set_xticks(ticks=self.modelmanager.errticklocs, labels=iterLabel, rotation="vertical")

            #print(self.filterclass.__name__, " likeli:     ", likeli.flatten())
            #print(self.filterclass.__name__, " likeli_pos: ", likeli_pos.flatten())

            for metric, error_box, error_vio, ax in [
                    (metric,     self.error_box,      self.error_vio,      self.error_ax), 
                    (likeli,     self.likeli_box,     self.likeli_vio,     self.likelihood_ax), 
                    (likeli_pos, self.likeli_pos_box, self.likeli_pos_vio, self.likelihood_pos_ax)]:
                new_boxplot = self.emptyax.boxplot(metric, positions=[self.index], widths=0.5, whis=[5,95], sym='' )
                for desc in ['whiskers', 'caps', 'boxes', 'medians']:
                    own_lines = error_box[desc]
                    new_lines = new_boxplot[desc]
                    for i in range(len(own_lines)):
                        own_lines[i].set_data(*new_lines[i].get_data())
                dy = abs(new_boxplot['caps'][0].get_ydata()[0]-new_boxplot['caps'][1].get_ydata()[0])
                dy = min(dy, (ax.get_ylim()[1]-ax.get_ylim()[0]))
                new_violin = self.emptyax.violinplot(metric, positions=[self.index], widths=0.8, 
                                                     showextrema=False, points=1000, bw_method=dy/100)["bodies"][0]
                error_vio.set_verts([new_violin.get_paths()[0].vertices])
            
            if self.plot_iter:
                new_boxplot = self.emptyax.boxplot(iters, positions=[self.index], widths=0.5, whis=[5,95], sym='' )
                for desc in ['whiskers', 'caps', 'boxes', 'medians']:
                    own_lines = self.iter_box[desc]
                    new_lines = new_boxplot[desc]
                    for i in range(len(own_lines)):
                        own_lines[i].set_data(*new_lines[i].get_data())

            if self.filterclass == filters.proposed.StudentTFilter:
                self.time_ax.set_ylim(0,4*(comp_time+rest_time))
                self.error_ax.set_ylim(0, 2*self.error_box["caps"][1].get_ydata()[0])
                ymin = self.likeli_box["caps"][0].get_ydata()[0]; ymax = max(ymin+0.01,self.likeli_box["caps"][1].get_ydata()[0])
                self.likelihood_ax.set_ylim(ymin-0.50*(ymax-ymin), ymax+0.25*(ymax-ymin))
                ymin = self.likeli_pos_box["caps"][0].get_ydata()[0]; ymax = max(ymin+0.01,self.likeli_pos_box["caps"][1].get_ydata()[0])
                self.likelihood_pos_ax.set_ylim(ymin-0.50*(ymax-ymin), ymax+0.25*(ymax-ymin))
                if False and plot_iter:
                    ymin, ymax = (0, self.iter_box["caps"][1].get_ydata()[0])
                    self.iter_ax.set_ylim(ymin-0.05*(ymax-ymin), ymax+0.25*(ymax-ymin))
            
            self.vis_button.tooltip = self.filter.desc()
            self.need_redraw = False

        self.compute = compute
        
        
        self.widget_dict = {}
        for name, widget, widget_params in params:
            new_widget = widget(**widget_params)
            new_widget.observe(self.compute, 'value')
            self.widget_dict[name] = new_widget
        
        self.smoother.observe(self.compute, 'value')
        
        self.widget_box = widgets.HBox([self.vis_button, self.err_button, self.smoother,
                                        organise_widgets_in_grid(widget_list=list(self.widget_dict.values()))],
                                        layout=boxlayout)
        compute()
    
    
    def update(self):
        """
        Is called by the :class:`ExtendedSingerModelManager` whenever the filter has to be updated.
        """
        if self.visibility:
            self.compute()
        else:
            self.need_redraw = True
