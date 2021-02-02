
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils

#E(2)-steerable CNNs - librar"y:
from e2cnn import gspaces    
from e2cnn import nn as G_CNN   
import e2cnn

#Plotting in 2d/3d:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#Tools:
import datetime
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append('../../')
#Own files:
import Kernel_and_GP_tools as GP
import My_Tools

#HYPERPARAMETERS and set seed:
torch.set_default_dtype(torch.float)

def plot_context_and_VF(ax,X_Context=None,Y_Context=None,X_Target=None,Y_Target=None,scale=20,colormap='viridis',facecolor='black',color=None,x1_lim=[-10,10],x2_lim=[-10,10]):
    '''
    Inputs: ax - matplotlib.axes._subplots.AxesSubplot - axis to plot on
            X_Context, Y_Context, X_Target, Y_Target - torch.Tensor - shape (n_c,2)/ (n_t,2) context and target set to plot
            scale - float - scale for plotting quiver arrows
            colormap - string - name of colormap to use 
            x1_lim,x2_lim - [float,float] - range of x1- and x2-axis
    Output: plot the context set in red and the target set (i.e. the ground truth) together in one plot
    '''
    #Set limits of axis and background color, get the width:
    ax.set_xlim(x1_lim)
    ax.set_ylim(x2_lim)
    if facecolor is not None:
        ax.set_facecolor(facecolor)
    width=ax.get_xlim()[1]-ax.get_xlim()[0]
    #Plot target vector field
    if X_Target is not None and Y_Target is not None:
        if color is not None:
            ax.quiver(X_Target[:,0],X_Target[:,1],Y_Target[:,0],Y_Target[:,1],color=color,
            pivot='mid',scale_units='width',scale=width*scale,headlength=4, headwidth = 2,width=0.005,alpha=1)
        else:
            ax.quiver(X_Target[:,0],X_Target[:,1],Y_Target[:,0],Y_Target[:,1],Y_Target.norm(dim=1),
            cmap=cm.get_cmap(colormap),pivot='mid',scale_units='width',scale=width*scale,headlength=4, headwidth = 2,width=0.005,alpha=1)
    #Plot context set:
    if X_Context is not None and Y_Context is not None:
        ax.quiver(X_Context[:,0],X_Context[:,1],Y_Context[:,0],Y_Context[:,1],
            color='red',pivot='mid',label='Context set',scale_units='width',scale=width*scale,headlength=4, headwidth = 2,width=0.005)   
    return(ax)

def plot_Covs(ax,X_Target,Cov_Mat,scale=20,x1_lim=[-10,10],x2_lim=[-10,10],alpha=1.,facecolor='orange',edgecolor='grey',linewidth=1.):
    '''
    Inputs: ax - matplotlib.axes._subplots.AxesSubplot - axis to plot on
            X_Target - torch.Tensor - shape (n_t,2) - locations of the target set
            Cov_Mat - torch.Tensor - shape (n_t, 2 ,2) - covariance matrices 
            scale - float - scale for plotting quiver arrows
            x1_lim,x2_lim - [float,float] - range of x1- and x2-axis
            alpha - float - permeability of covariance ellipses
            facecolor - str - name of color to use to plot covariance ellipses
            edgecolor - str - name of the color to use to plot the edges of the covariance ellipses
    Output: Plots the covariance matrices as ellipses in R2 
    '''
    #Get the 66%-quantile for the chi-square distribution:
    chi_sq_quantile=math.sqrt(2.157)  #for 95%:5.99#

    if X_Target is not None and Cov_Mat is not None:
        #Set window:
        ax.set_xlim(x1_lim)
        ax.set_ylim(x2_lim)
        
        #Go over all target points and plot ellipse of continour lines of density of distributions:
        for j in range(X_Target.size(0)):
            #Get covarinace matrix:
            A=Cov_Mat[j]
            #Decompose A:
            eigen_decomp=torch.eig(A,eigenvectors=True)
            #Get the eigenvector corresponding corresponding to the largest eigenvalue:
            u=eigen_decomp[1][:,0]

            #Get the angle of the ellipse in degrees:
            angle=360*torch.atan(u[1]/u[0])/(2*math.pi)
        
            #Get the width and height of the ellipses (eigenvalues of A):
            D=eigen_decomp[0][:,0]
            width=2*chi_sq_quantile*torch.sqrt(D[0])/scale
            height=2*chi_sq_quantile*torch.sqrt(D[1])/scale
            #Plot the Ellipse:
            E=Ellipse(xy=X_Target[j,].numpy(),width=width,height=height,angle=angle,alpha=alpha,zorder=0,facecolor=facecolor,edgecolor=edgecolor,linewidth=linewidth)
            #E.set_color(colormap(traces[j].item()))
            #E.set_facecolor(colormap(traces[j].item()))
            ax.add_patch(E)
    return(ax)

def compare_predict_vs_truth(ax,X_Context=None,X_Target=None,Predict_1=None,Predict_2=None,Cov_Mat=None,scale=20,x1_lim=[-10,10],x2_lim=[-10,10],color='orange',diff_color='black'):
    '''
    Inputs: ax - matplotlib.axes._subplots.AxesSubplot - axis to plot on
            X_Context,X_Target,  Predict_1, Predict_2 - torch.Tensor - shape (n_c,2)/ (n_t,2) context and target points, 2 predictions to compare
            Cov_Mat - torch.Tensor - shape (n_t,2,2) - covariance matrices to plot
            scale - float - scale for plotting quiver arrows
            x1_lim,x2_lim - [float,float] - range of x1- and x2-axis
    Output: ax - plots the difference between Predict_1 and Predict_2 over the confidence ellipses
    '''
    #Set window and get width:
    ax.set_xlim(x1_lim)
    ax.set_ylim(x2_lim)
    width=ax.get_xlim()[1]-ax.get_xlim()[0]
    #Plot target set and covariance ellipses:
    if X_Target is not None and Cov_Mat is not None:
        ax=plot_Covs(ax,X_Target,Cov_Mat,alpha=.6,scale=scale,x1_lim=x1_lim,x2_lim=x2_lim,facecolor=color,edgecolor='black')
    #Plot the difference between the two given vector fields Predict_1, Predict_2 on the target set:
    if X_Target is not None and Predict_1 is not None and Predict_2 is not None:
        Diff=Predict_1-Predict_2
        #ax.tricontourf(X_Target[:,0],X_Target[:,1],Diff.norm(dim=1),alpha=0.8,method='cubic',cmap=cm.get_cmap(colormap))
        ax.quiver(X_Target[:,0],X_Target[:,1],Diff[:,0],Diff[:,1], 
            color=diff_color,pivot='tail',label='Predict',scale_units='width',scale=scale*width,width=0.005,headwidth=1.5,headlength=3)
    #Plot the locations of the context points:
    if X_Context is not None:
        ax.scatter(X_Context[:,0],X_Context[:,1],color='red',marker='x')
    return(ax)

def compare_VF(ax,X_Context=None,X_Target=None,Predict_1=None,Predict_2=None,scale=20,x1_lim=[-10,10],x2_lim=[-10,10],color_1=cm.get_cmap('viridis')(0.9),color_2='magenta'):
    '''
    Inputs: ax - matplotlib.axes._subplots.AxesSubplot - axis to plot on
            X_Context,X_Target,  Predict_1, Predict_2 - torch.Tensor - shape (n_c,2)/ (n_t,2) context and target points, 2 predictions to compare
            scale - float - scale for plotting quiver arrows
            x1_lim,x2_lim - [float,float] - range of x1- and x2-axis
    Output: plots predict_1 and predict_2 on top of each other
    '''
    #Set background color and window, get width:
    #ax.set_facecolor('black')
    ax.set_xlim(x1_lim)
    ax.set_ylim(x2_lim)
    width=ax.get_xlim()[1]-ax.get_xlim()[0]

    #Plot the first vector field:
    if Predict_1 is not None:
        ax.quiver(X_Target[:,0],X_Target[:,1],Predict_1[:,0],Predict_1[:,1],
            color=color_1,pivot='mid',label='Predict',scale_units='width',scale=width*scale,headlength=6, headwidth = 6,width=0.01) 
    #Plot the second vector field:
    if Predict_2 is not None:
        ax.quiver(X_Target[:,0],X_Target[:,1],Predict_2[:,0],Predict_2[:,1],alpha=1.,
            color=color_2,pivot='mid',label='Predict',scale_units='width',scale=width*scale,headlength=6, headwidth = 6,width=0.01) 
    #Plot the context set:
    if X_Context is not None:
        ax.scatter(X_Context[:,0],X_Context[:,1],color='red',marker='x')
    return(ax)

def plot_diff(ax,X_Context,X_Target=None,Predict_1=None,Predict_2=None,scale=20,x1_lim=[-10,10],x2_lim=[-10,10]):
    '''
    Inputs: ax - matplotlib.axes._subplots.AxesSubplot - axis to plot on
            X_Context,X_Target,  Predict_1, Predict_2 - torch.Tensor - shape (n_c,2)/ (n_t,2) context and target points, 2 predictions to compare
            scale - float - scale for plotting quiver arrows
            x1_lim,x2_lim - [float,float] - range of x1- and x2-axis
    Output: plots the difference between Predict_1 and Predict_2 and a contour plot of the norm of the difference
    '''
    #Set the window, get the width and set the background color:
    ax.set_xlim(x1_lim)
    ax.set_ylim(x2_lim)
    width=ax.get_xlim()[1]-ax.get_xlim()[0]
    ax.set_facecolor('black')

    #Plot the difference between the two vector fields as a contour plot of the norms and a vector field:
    if Predict_1 is not None and Predict_2 is not None:
        Diff=Predict_1-Predict_2
        norms=Diff.norm(dim=1)
        tricontour=ax.tricontourf(X_Target[:,0],X_Target[:,1],norms,cmap=cm.get_cmap('cividis'), extend='both')
        ax.quiver(X_Target[:,0],X_Target[:,1],Diff[:,0],Diff[:,1],pivot='mid',scale_units='width',scale=width*scale,headlength=4, headwidth = 2,width=0.005,alpha=1)
    #Plot the context set:
    if X_Context is not None:
        ax.scatter(X_Context[:,0],X_Context[:,1],color='red',marker='x')
    return(ax,tricontour)

def diff_covs(ax,X_Context=None,X_Target=None,Cov_Mat_1=None,Cov_Mat_2=None,scale=20,x1_lim=[-10,10],x2_lim=[-10,10]):
    '''
    Inputs: ax - matplotlib.axes._subplots.AxesSubplot - axis to plot on
            X_Context,X_Target - torch.Tensor - shape (n_c,2)/ (n_t,2) context and target points
            Cov_Mat_1,Cov_Mat_2 - torch.Tensor - shape (n_t,2) - two sets of covariance matrices on the target points
            scale - float - scale for plotting quiver arrows
            x1_lim,x2_lim - [float,float] - range of x1- and x2-axis
    Output: plots Cov_Mat_1 and Cov_Mat_2 on top of each other to compare
    '''
    #Set the window:
    ax.set_xlim(x1_lim)
    ax.set_ylim(x2_lim)
    #Plot the first set of covariance ellipses:
    if X_Target is not None and Cov_Mat_1 is not None:
        ax=plot_Covs(ax,X_Target,Cov_Mat_1,alpha=1,scale=scale,x1_lim=x1_lim,x2_lim=x2_lim,facecolor='darkturquoise',edgecolor=None)
    #Plot the second set of covariance ellipses:
    if X_Target is not None and Cov_Mat_2 is not None:
        ax=plot_Covs(ax,X_Target=X_Target,Cov_Mat=Cov_Mat_2,alpha=0.5,scale=scale,x1_lim=x1_lim,x2_lim=x2_lim,facecolor='firebrick',edgecolor=None)#'black')
    #Plot context set:
    if X_Context is not None:
        ax=ax.scatter(X_Context[:,0],X_Context[:,1],color='red',marker='x')
    return(ax)
