# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def add_colorbar(fig, ax, image, label='', aspect=0.05, pad=0.4, side='right'):
    width = axes_size.AxesY(ax, aspect=aspect)
    pad_fraction = axes_size.Fraction(pad, width)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size=width, pad=pad_fraction)
    fig.colorbar(image, cax=cax, label=label)


def set_axis_aspect(ax, aspect):
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(aspect*abs((xright - xleft)/(ybottom - ytop)))
