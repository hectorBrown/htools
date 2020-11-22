# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 23:13:33 2020

@author: Hector
"""
import numpy as np, matplotlib.pyplot as plt
def default_plot(x,y,xlab=None,ylab=None,yerr=None,xerr=None,title=None,figsize=(10,10), capsize=4, fmt=None):
    x = np.array(x)
    y = np.array(y)
    plt.figure(figsize=figsize)
    if not xlab is None: plt.xlabel(xlab)
    if not ylab is None: plt.ylabel(ylab)
    if not title is None:
        plt.title(title)
    elif not xlab is None and not ylab is None:
        plt.title(ylab + " against " + xlab)
    if len(x.shape) == 1:
        if yerr is None and xerr is None:
            if fmt is None:
                plt.plot(x,y)
            else:
                plt.plot(x,y,fmt)
        elif yerr is None and not xerr is None:
            if fmt is None: fmt = 'o'
            plt.errorbar(x,y,xerr=xerr,fmt=fmt,capsize=capsize)
        elif not yerr is None and xerr is None:
            if fmt is None: fmt = 'o'
            plt.errorbar(x,y,yerr=yerr,fmt=fmt,capsize=capsize)
        else:
            if fmt is None: fmt = 'o'
            plt.errorbar(x,y,yerr=yerr,xerr=xerr,fmt=fmt,capsize=capsize)
    else:
        if yerr is None: yerr = [None] * len(x)
        if xerr is None: xerr = [None] * len(x)
        if fmt is None: fmt = [None] * len(x)
        for i in range(len(x)):
            if yerr[i] is None and xerr[i] is None:
                if fmt[i] is None:
                    plt.plot(x[i],y[i])
                else:
                    plt.plot(x[i],y[i],fmt[i])
            elif yerr[i] is None and not xerr[i] is None:
                if fmt[i] is None: fmt[i] = 'o'
                plt.errorbar(x[i],y[i],xerr=xerr[i],fmt=fmt[i],capsize=capsize)
            elif not yerr[i] is None and xerr[i] is None:
                if fmt[i] is None: fmt[i] = 'o'
                plt.errorbar(x[i],y[i],yerr=yerr[i],fmt=fmt[i],capsize=capsize)
            else:
                if fmt[i] is None: fmt[i] = 'o'
                plt.errorbar(x[i],y[i],yerr=yerr[i],xerr=xerr[i],fmt=fmt[i],capsize=capsize)
    return plt.figure(1)
def eqfit(x,y,err=None):
    if not err is None:
        err = np.array(err)
        return np.poly1d(np.polyfit(x,y,1,w=1/err))
    else:
        return np.poly1d(np.polyfit(x,y,1))
