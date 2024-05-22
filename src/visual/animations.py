# Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Jonas Noah Michael Neuhöfer
"""
Gives some implementations to create explainatory animations.
"""

import numpy as np
import re
from matplotlib.path import Path
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.transforms as transforms
from matplotlib.legend_handler import HandlerPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

print("-   Loading File 'animations.py'")

def SVGPath_to_PLTPath(path: str, scale=1, closed=True, xoff=0.5, yoff=0.5):
    """
    Transforms the SVG Path into a matplotlib Path instance 

    :param path: A string describing the SVG according to http://www.w3.org/2000/svg
    :param scale: scales the SVG to fit into the [-xoff*scale,(1-xoff)*scale]x[-yoff*scale,(1-yoff)*scale] box, defaults to 1.
            Note that the width to height ration in the SVG is preserved 
    :param closed: Whether to add CLOSEPOLY flags at the end of each linesegment to signify closed curves, defaults to True.
    :param xoff: see scale, defaults to 0.5.
    :param yoff: see scale, defaults to 0.5.
    :return: the Path instance
    """
    split = re.split('([a-zA-Z])', path) # split at each word of at least 1 letter, case-insensitve
    i = 0; ctrl = None
    (cur_x, cur_y) = (0,0)
    (last_ccx, last_ccy) = (None, None) # the last cubic control points
    (last_qcx, last_qcy) = (None, None) # the last quadratic control points
    (minx,miny,maxx,maxy) = (None,None,None,None)
    startx, starty = (None,None)
    xcoords = np.zeros((0,))
    ycoords = np.zeros((0,))
    codes = np.zeros((0,))
    while i < len(split)-1:
        ctrl = None
        while( ctrl is None ):
            ctrl = re.search('[a-z]', split[i], re.I) # first letter of the string, if no letter present: None
            i += 1
        ctrl = ctrl.group(0)
        if ctrl not in ['z', 'Z']:
            matches = re.findall(r'(-?[0-9]+(\.[0-9]+)?|(-?\.[0-9]+))', split[i])
            coords = np.array([float(x[0]) for x in matches])
            j = 0
            while j < len(coords):
                if ctrl == 'M':
                    if closed and startx is not None:
                        xcoords = np.append(xcoords, [startx], axis=0); ycoords = np.append(ycoords, [starty], axis=0); codes = np.append(codes, [Path.CLOSEPOLY])
                    cur_x = coords[j]; cur_y = coords[j+1]; j += 2
                    xcoords = np.append(xcoords, [cur_x], axis=0); ycoords = np.append(ycoords, [cur_y], axis=0); codes = np.append(codes, [Path.MOVETO])
                    startx = cur_x; starty = cur_y; ctrl = 'L'
                elif ctrl == 'm':
                    if closed and startx is not None:
                        xcoords = np.append(xcoords, [startx], axis=0); ycoords = np.append(ycoords, [starty], axis=0); codes = np.append(codes, [Path.CLOSEPOLY])
                    cur_x += coords[j]; cur_y += coords[j+1]; j += 2
                    xcoords = np.append(xcoords, [cur_x], axis=0); ycoords = np.append(ycoords, [cur_y], axis=0); codes = np.append(codes, [Path.MOVETO])
                    startx = cur_x; starty = cur_y; ctrl = 'l'
                elif ctrl == 'L':
                    cur_x = coords[j]; cur_y = coords[j+1]; j += 2
                    xcoords = np.append(xcoords, [cur_x], axis=0); ycoords = np.append(ycoords, [cur_y], axis=0); codes = np.append(codes, [Path.LINETO])
                elif ctrl == 'l':
                    cur_x += coords[j]; cur_y += coords[j+1]; j += 2
                    xcoords = np.append(xcoords, [cur_x], axis=0); ycoords = np.append(ycoords, [cur_y], axis=0); codes = np.append(codes, [Path.LINETO])
                elif ctrl == 'H':
                    cur_x = coords[j]; j += 1
                    xcoords = np.append(xcoords, [cur_x], axis=0); ycoords = np.append(ycoords, [cur_y], axis=0); codes = np.append(codes, [Path.LINETO])
                elif ctrl == 'h':
                    cur_x += coords[j]; j += 1
                    xcoords = np.append(xcoords, [cur_x], axis=0); ycoords = np.append(ycoords, [cur_y], axis=0); codes = np.append(codes, [Path.LINETO])
                elif ctrl == 'V':
                    cur_y = coords[j]; j += 1
                    xcoords = np.append(xcoords, [cur_x], axis=0); ycoords = np.append(ycoords, [cur_y], axis=0); codes = np.append(codes, [Path.LINETO])
                elif ctrl == 'v':
                    cur_y += coords[j]; j += 1
                    xcoords = np.append(xcoords, [cur_x], axis=0); ycoords = np.append(ycoords, [cur_y], axis=0); codes = np.append(codes, [Path.LINETO])
                elif ctrl == 'C':
                    xcoords = np.append(xcoords, coords[j:j+6:2], axis=0); ycoords = np.append(ycoords, coords[j+1:j+6:2], axis=0); codes = np.append(codes, [Path.CURVE4, Path.CURVE4, Path.CURVE4])
                    last_ccx = coords[j+2]; last_ccy = coords[j+3]
                    cur_x = coords[j+4]; cur_y = coords[j+5]; j += 6
                elif ctrl == 'c':
                    xcoords = np.append(xcoords, coords[j:j+6:2]+cur_x, axis=0); ycoords = np.append(ycoords, coords[j+1:j+6:2]+cur_y, axis=0); codes = np.append(codes, [Path.CURVE4, Path.CURVE4, Path.CURVE4])
                    last_ccx = cur_x+coords[j+2]; last_ccy = cur_y+coords[j+3]
                    cur_x += coords[j+4]; cur_y += coords[j+5]; j += 6
                elif ctrl == 'S':
                    new_ccx = 2*cur_x-last_ccx if last_ccx is not None else cur_x; new_ccy = 2*cur_y-last_ccy if last_ccy is not None else cur_y
                    xcoords = np.append(xcoords, [new_ccx,coords[j],coords[j+2]], axis=0); ycoords = np.append(ycoords, [new_ccy,coords[j+1],coords[j+3]], axis=0); codes = np.append(codes, [Path.CURVE4, Path.CURVE4, Path.CURVE4])
                    last_ccx = coords[j]; last_ccy = coords[j+1]
                    cur_x = coords[j+2]; cur_y = coords[j+3]; j += 4
                elif ctrl == 's':
                    new_ccx = 2*cur_x-last_ccx if last_ccx is not None else cur_x; new_ccy = 2*cur_y-last_ccy if last_ccy is not None else cur_y
                    xcoords = np.append(xcoords, [new_ccx,cur_x+coords[j],cur_x+coords[j+2]], axis=0); ycoords = np.append(ycoords, [new_ccy,cur_y+coords[j+1],cur_y+coords[j+3]], axis=0); codes = np.append(codes, [Path.CURVE4, Path.CURVE4, Path.CURVE4])
                    last_ccx = cur_x+coords[j]; last_ccy = cur_y+coords[j+1]
                    cur_x = coords[j+2]; cur_y = coords[j+3]; j += 4

                elif ctrl == 'Q':
                    xcoords = np.append(xcoords, coords[j:j+4:2], axis=0); ycoords = np.append(ycoords, coords[j+1:j+4:2], axis=0); codes = np.append(codes, [Path.CURVE3, Path.CURVE3])
                    last_qcx = coords[j]; last_qcy = coords[j+1]
                    cur_x = coords[j+2]; cur_y = coords[j+3]; j += 4
                elif ctrl == 'q':
                    xcoords = np.append(xcoords, cur_x+coords[j:j+4:2], axis=0); ycoords = np.append(ycoords, cur_y+coords[j+1:j+4:2], axis=0); codes = np.append(codes, [Path.CURVE3, Path.CURVE3])
                    last_qcx = cur_x+coords[j]; last_qcy = cur_y+coords[j+1]
                    cur_x += coords[j+2]; cur_y += coords[j+3]; j += 4
                elif ctrl == 'T':
                    new_qcx = 2*cur_x-last_qcx if last_qcx is not None else cur_x; new_qcy = 2*cur_y-last_qcy if last_qcy is not None else cur_y
                    xcoords = np.append(xcoords, [new_qcx,coords[j]], axis=0); ycoords = np.append(ycoords, [new_qcy,coords[j+1],], axis=0); codes = np.append(codes, [Path.CURVE3, Path.CURVE3])
                    last_qcx = new_qcx; last_qcy = new_qcy
                    cur_x = coords[j]; cur_y = coords[j+1]; j += 2
                elif ctrl == 't':
                    new_qcx = 2*cur_x-last_qcx if last_qcx is not None else cur_x; new_qcy = 2*cur_y-last_qcy if last_qcy is not None else cur_y
                    xcoords = np.append(xcoords, [new_qcx,cur_x+coords[j]], axis=0); ycoords = np.append(ycoords, [new_qcy,cur_y+coords[j+1],], axis=0); codes = np.append(codes, [Path.CURVE3, Path.CURVE3])
                    last_qcx = new_qcx; last_qcy = new_qcy
                    cur_x += coords[j]; cur_y += coords[j+1]; j += 2
                else:
                    print(f"UNKOWN CONTROL '{ctrl}'")
                    break
                if ctrl not in ['C', 'c', 'S', 's']:
                    last_ccx = None; last_ccy = None
                elif ctrl not in ['Q', 'q', 'T', 't']:
                    last_qcx = None; last_qcy = None
                minx = minx if minx is not None and minx < cur_x else cur_x
                miny = miny if miny is not None and miny < cur_y else cur_y
                maxx = maxx if maxx is not None and maxx > cur_x else cur_x
                maxy = maxy if maxy is not None and maxy > cur_y else cur_y
            #print("                   ", " ".join([ f"{xcoords[i]:.2f} {ycoords[i]:.2f}" for i in range(-j//2, 0) ]))
        else:
            pass
            #print(f"GOT '{ctrl}'")
    # normalise the image by mapping the viewBox to [-scale/2,scale/2]x[-scale/2,scale/2]
    if closed and startx is not None:
        xcoords = np.append(xcoords, [startx], axis=0); ycoords = np.append(ycoords, [starty], axis=0); codes = np.append(codes, [Path.CLOSEPOLY])
    maxdim = max(maxx-minx, maxy-miny)
    xcoords = (xcoords - xoff*maxx- (1-xoff)*minx)/maxdim*scale
    ycoords = (ycoords - yoff*maxy- (1-yoff)*miny)/maxdim*scale
    path = Path(np.stack([xcoords,ycoords],axis=1), codes=codes)
    return path

#: An SVG grafic for a car used to symbolise a moving agent.
#: Source: https://uxwing.com/car-top-view-icon/, 
#: License: All icons are free to use for any personal and commercial projects without any attribution or credit 
SVGcar = """
    M42.3 110.94
    c2.22 24.11 2.48 51.07 1.93 79.75-13.76.05-24.14 1.44-32.95 6.69-4.96 2.96-8.38 6.28-10.42 12.15-1.37 4.3-.36 7.41 2.31 8.48 4.52 1.83 22.63-.27 28.42-1.54 2.47-.54 4.53-1.28 5.44-2.33.55-.63 1-1.4 1.35-2.31 1.49-3.93.23-8.44 3.22-12.08.73-.88 1.55-1.37 2.47-1.61-1.46 62.21-6.21 131.9-2.88 197.88 0 43.41 1 71.27 43.48 97.95 41.46 26.04 117.93 25.22 155.25-8.41 32.44-29.23 30.38-50.72 30.38-89.54 5.44-70.36 1.21-134.54-.79-197.69.69.28 1.32.73 1.89 1.42 2.99 3.64 1.73 8.15 3.22 12.08.35.91.8 1.68 1.35 2.31.91 1.05 2.97 1.79 5.44 2.33 5.79 1.27 23.9 3.37 28.42 1.54 2.67-1.07 3.68-4.18 2.31-8.48-2.04-5.87-5.46-9.19-10.42-12.15-8.7-5.18-18.93-6.6-32.44-6.69-.75-25.99-1.02-51.83-.01-77.89
    C275.52-48.32 29.74-25.45 42.3 110.94
    z
    M111.93 20.06
    C101.49 37.86 75.41 61.78 54.36 77.59
    C62.75 48.67 83.52 30.68 111.93 20.06
    z
    m89.14-4.18
    c28.41 10.62 49.19 28.61 57.57 57.53-21.05-15.81-47.13-39.73-57.57-57.53
    z
    M71.29 388.22
    l8.44-24.14
    c53.79 8.36 109.74 7.72 154.36-.15
    l7.61 22.8
    c-60.18 28.95-107.37 32.1-170.41 1.49
    z
    M256.55 354.09
    C244.8 307.99 241.92 281.11 254.56 233.48 261.35 267.51 262.41 319.99 256.55 354.09
    z
    M70.18 238.83
    l-10.34-47.2
    c45.37-57.48 148.38-53.51 193.32 0
    l-12.93 47.2
    c-57.58-14.37-114.19-13.21-170.05 0
    z
    M56.45 354.09
    c-5.86-34.1-4.8-86.58 1.99-120.61 12.63 47.63 9.76 74.51-1.99 120.61
    z"""

class Car:
    """
    An animateable agent.
    """
    patch = None
    length = 0
    front_pos = None
    rear_pos = None
    ax = None

    def __init__(self, state, ax, model, length=1, predict=0, past=0, dt=0.5, **patch_kwargs) -> None:
        """
        :param state: the state of the car, i.e., x/y positions, velocities, accelerations consistent with :class:`ExtendedSingerModelManager`.
                In particular, the positions will be the coordinates of the center of the axle between the front wheels.
        :param ax: the axis the car is plotted into 
        :param model: the dynamic model controlling the agent's movement, e.g., an instance of :class:`ExtendedSingerModelManager`.
        :param length: the distance between front and rear wheel axles
        :param predict: how many timesteps to predict into the future according to the dynamic model
        :param past: how many past positions (of the rear axle center) should be kept and plotted
        :param dt: the time step size used in the predictions
        :param patch_kwargs: the parameters for creating a drawable PathPatch of the car, such as facecolor, alpha, ...  
        """
        carPath = SVGPath_to_PLTPath(SVGcar,scale=-length, yoff=0.1)
        self.patch = PathPatch(carPath, transform=ax.transData, **patch_kwargs)
        self.length =  0.9*length
        self.front_pos = state
        self.ax = ax
        self.model = model

        v_scale = self.length/(state[2]**2 + state[3]**2+1e-30)**0.5
        self.rear_pos = state-np.array([state[2]*v_scale, state[3]*v_scale,0,0,0,0])
        ax.add_patch(self.patch)
        trans = transforms.Affine2D().rotate(-np.arctan2(state[2],state[3])).translate(state[0], state[1]) + ax.transData
        self.patch.set_transform(trans)
        self.pred = predict
        self._predict = ax.plot([0],[0], color="red", zorder=-10)[0]
        self.dt = dt
        self.pred_Fdt = model._F(dt)
        self.predict()
        self.past = past
        self.past_rears = np.full((past,6), self.rear_pos)
        self._past = LineCollection(([(0,0)],), color="black", **patch_kwargs)
        ax.add_collection(self._past)
        if 'fc' in patch_kwargs:
            self.cmap = LinearSegmentedColormap.from_list('custom_cmap', [patch_kwargs['fc'], patch_kwargs['fc'],[1,1,1,0]])
        else:
            self.cmap = LinearSegmentedColormap.from_list('custom_cmap', [[1,1,1,0], [1,1,1,0.55],[1,1,1,1]])

        #self._past = ax.plot([0],[0], color="black", zorder=-10)[0]
        #self._anchors = ax.plot([pos[0], self.rear_pos[0]], [pos[1], self.rear_pos[1]], "ok")[0]
    
    def predict(self,predict=None,dt=None):
        """
        Predicts the future movement of the car, which will be plotted as dotted lines in front of the car.

        :param predict: How many timesteps into the future to predict. If ``None`` defaults to the value set 
                when initialising the ``Car`` instance.
        :param dt: the time step size used in the predictions. If ``None`` defaults to the value set 
                when initialising the ``Car`` instance.
        """
        if predict is not None:
            self.pred = predict
        if dt is not None:
            self.dt
            self.pred_Fdt = self.model._F(dt)
        predictions = [self.front_pos]
        for i in range(self.pred):
            predictions.append( self.pred_Fdt @ predictions[-1] )
        predictions = np.array(predictions)
        self._predict.set_data(predictions[:,0], predictions[:,1])
    
    def keep_past(self,past=None):
        """
        Updates the list of past rear positions.

        :param past: The new length of the list of past rear positions
        """
        if past is not None:
            # update length of kept past
            if self.past > past:
                self.past_rears = self.past_rears[-past:]
            elif self.past < past:
                if self.past == 0:
                    self.past_rears = np.full((past,6), self.rear_pos)
                else:
                    self.past_rears = np.concatenate([np.full((past-self.past,6), self.past_rears[0]), self.past_rears])
            self.past = past
            if self.past > 0:
                self._past.set_color(self.cmap(np.linspace(1,0,self.past)))
        self.past_rears = self.past_rears[max(self.past_rears.shape[0]-self.past-1,0):]
        self._past.set_segments([ self.past_rears[i:i+2,:2] for i in range(0,self.past) ])
    
    def update(self, new_pos):
        """
        Updates the state of the car given the new front axle position. 
        The rear axle position follows the shortest path between the old rear position and the new front position.
        """
        new_diff_rear = new_pos[:] - self.rear_pos[:]
        new_diff_rear /= (new_diff_rear[0]**2 + new_diff_rear[1]**2+1e-30)**0.5
        self.rear_pos = new_pos - self.length*new_diff_rear
        self.front_pos = new_pos
        trans = transforms.Affine2D().rotate(-np.arctan2(new_diff_rear[0],new_diff_rear[1])).translate(new_pos[0], new_pos[1]) + self.ax.transData
        self.patch.set_transform(trans)
        self.past_rears = np.append(self.past_rears,self.rear_pos[None,:], axis=0)
        self.predict()
        self.keep_past()
        #self._anchors.set_data([new_pos[0], self.rear_pos[0]], [new_pos[1], self.rear_pos[1]])

class HandlerPathPatch(HandlerPatch):
    """
    Custom handler class to be able to put the Car SVG into a legend.
    """
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        """
        Creates the artist for the legend displaying the car symbol.

        :param legend: The legend
        :param orig_handle: the original :class:`Car` instance the handler is supposed to index
        :param width: the width of the artist, corresponding to the length of the car
        :param trans: an additional transform applied to the artist
        """
        path = SVGPath_to_PLTPath(SVGcar,scale=-width*0.9, yoff=0)
        path = path.transformed( transforms.Affine2D().rotate_deg(90) + transforms.Affine2D().translate(3, 3) )
        p = PathPatch(path, 
                      lw=orig_handle.get_linewidth(),
                      edgecolor=orig_handle.get_edgecolor(),
                      facecolor=orig_handle.get_facecolor(),
                      linestyle=orig_handle.get_linestyle(),
                      capstyle=orig_handle.get_capstyle(),
                      joinstyle=orig_handle.get_joinstyle(),
                      fill=orig_handle.get_fill(),
                      snap=orig_handle.get_snap(),
                      visible=orig_handle.get_visible())
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def upsampling(x, t, model, dt=0.05):
    """
    Upsamples the given trajectory states ``x`` at timesteps ``t`` produced by the linear model 
    ``model`` into a trajectory that has a consistent time-difference ``dt``. 
    Can be used to interpolate new states between existing states.

    :param x: the trajectory states
    :param t: the timesteps of the trajectory states
    :param model: The linear model whos _F(dt) matrix produces the deterministic relationship between
            :math:`x_t` and :math:`x_{t + dt}` 
    :param dt: The target time-difference between new trajectory states, defaults to 0.05
    """

    eps = min(1e-8, dt/1000) #the difference between two timesteps such that they will be considered the same.
    newt = t[0]
    newx = [x[0]]
    k_last = 0; k_next = 1
    F_matrices = {}
    def F(dt):
        if dt not in F_matrices:
            F_matrices[dt] = model._F(dt)
        return F_matrices[dt]
    while newt < t[-1]:
        while t[k_last+1]-eps < newt:
            k_last += 1
        while t[k_next]+eps < newt+dt:
            k_next += 1
        nextt = newt+dt
        dt_forward = newt - t[k_last]; dt_back = nextt - t[k_next]
        if dt_forward <= eps:
            dt_forward = 0
            newt = t[k_last]
        if dt_back >= -eps:
            dt_back = 0
            nextt = t[k_next]
        Dt = t[k_next] - t[k_last]
        newx.append( (newt-t[k_last])/Dt*( (F(dt_forward) @ x[k_last]) if dt_forward > eps else x[k_last]) 
                    +(t[k_next]-newt)/Dt*( (F(dt_back)    @ x[k_next]) if dt_back   < -eps else x[k_next]) )
        newt = nextt
    newx.append(F(newt-t[-1])@x[-1])
    return np.array(newx), np.arange(len(newx))*dt+t[0]


def arrow_for_outlier(view, obs, l, minx, miny, maxx, maxy):
    """
    Generates an arrow pointing at the observations ``obs`` from the viewpoint ``view``
    whose tip is on the box defined by ``[minx,maxx]``x``[miny,maxy]``. The length of the
    arrow is given by ``l``.

    :return: ``x,y,dx,dy,out``: The x and y coordinates of the tail of the arrow at its length dx, dy. 
            ``out`` is a boolean value that is ``True`` if the observation is actually not included in ``[minx,maxx]``x``[miny,maxy]``.
    """
    view = np.asarray(view).reshape((2,))
    obs = np.asarray(obs).reshape((2,))
    dir = obs-view
    # directions from viewpoints to frame corners, to figure out which frame boundary the line
    # between viewpoint and observation crosses. Going clockwise from top-right, bottom-right, bottom-left, top-left
    corner_dir = np.array([[maxx,maxx,minx,minx],[maxy,miny,miny,maxy]]) - view[:,None]
    normal_dir = np.array([corner_dir[1], -corner_dir[0]])
    angles = np.sum(normal_dir*dir[:,None], axis=0)
    if angles[0] >= 0 and angles[1] < 0:
        # line intersects right frame boundary
        s = (maxx-view[0])/dir[0]
        arrowtip = np.array([maxx, view[1] + s*dir[1]])
    elif angles[1] >= 0 and angles[2] < 0:
        # line intersects bottom frame boundary
        s = (miny-view[1])/dir[1]
        arrowtip = np.array([view[0] + s*dir[0], miny])
    elif angles[2] >= 0 and angles[3] < 0:
        # line intersects left frame boundary
        s = (minx-view[0])/dir[0]
        arrowtip = np.array([minx, view[1] + s*dir[1]])
    else:
        # line intersects top frame boundary
        s = (maxy-view[1])/dir[1]
        arrowtip = np.array([view[0] + s*dir[0], maxy])
    length = l/(np.linalg.norm(view-arrowtip)+1e-5)
    arrowtail = arrowtip+length*(view-arrowtip)
    return arrowtail[0], arrowtail[1], (arrowtip[0]-arrowtail[0])/2, (arrowtip[1]-arrowtail[1])/2, s<1
