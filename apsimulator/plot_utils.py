# -*- coding: utf-8 -*-
# This file is part of the APSimulator API.
# Author: Lars Frogner
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


def figure(*args, **kwargs):
    return plt.figure(*args, **kwargs)


def subplots(*args, **kwargs):
    return plt.subplots(*args, **kwargs)


def tight_layout(*args, **kwargs):
    plt.tight_layout(*args, **kwargs)


def savefig(*args, **kwargs):
    plt.savefig(*args, **kwargs)


def render(output_path=False):
    if output_path:
        savefig(output_path)
    else:
        plt.show()


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


def plot_image(fig, ax, values, vmin=None, vmax=None, xlabel='', ylabel='', title='', origin='lower', extent=None, cmap=plt.get_cmap('gray'), colorbar=True, clabel=''):

    image = ax.imshow(values,
                      extent=extent,
                      vmin=vmin, vmax=vmax,
                      origin=origin,
                      interpolation='none',
                      cmap=cmap)
    if colorbar:
        add_colorbar(fig, ax, image, label=clabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
