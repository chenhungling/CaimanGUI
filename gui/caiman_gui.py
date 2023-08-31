#!/usr/bin/env python
'''
GUI taking CaImAn processed 1-photon data (hdf5 file) displaying spatial/temporal components
of different groups ('All', 'Accepted', 'Rejected', 'Unassigned') under various mode:
  - 'reset': initialization (show contours of the current group)
  - 'neurons': show colormap of the current group, display mouse clicked multiple components
  - 'correlation': show mouse clicked single componet and all other components
                   whose color is scaled to the pair's temporal correlation
  - 'accepted': display accepted components (keyboard left/right selection)
  - 'neighbors': display neighbors correlation of accepted components (keyboard left/right selection)
  
@author: Hung-Ling
'''
import os
import sys
import json
import cv2
import numpy as np
import pyqtgraph as pg
from pyqtgraph import FileDialog
from pyqtgraph.Qt import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree
from scipy.ndimage.measurements import center_of_mass
from scipy import sparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi

from memap_player import memap_window

## Interpret image data as 'col-major' (default pyqtgraph) or 'row-major' (numpy array)
pg.setConfigOptions(imageAxisOrder='row-major')

# %% Create a subclass MainWindow from the parent class QtGui.QMainWindow
class MainWindow(QtGui.QMainWindow):
    def __init__(self, datapath=None, jsonpath=None):
        super(MainWindow, self).__init__()  # Inherit the constructor, methods and properties of the parent class
        self.resize(1400,1000)  # width, height
        self.setWindowTitle('Caiman GUI Lite for 1-photon data')
        self.statusBar().showMessage('Ready')
        
        ## Discrete colors (cycle of 12) for displaying selected cells
        colors = plt.cm.Set3(np.linspace(0,1,12),bytes=True)[:,:3]  # Shape (12,3) dtype='uint8'
        self.colors = colors.tolist()  # Convert data type to list of int (to circumvent cv2 TypeError) 
        ## Continuous colormap used to setup a colorbar for metric scores
        cmap = plt.cm.jet(np.linspace(0,1,8),bytes=True)[:,:3]
        
        ## ------------ Initialize internal variables -------------------------
        self.loaded = False  # True if the caiman data is loaded
        self.config = None  # Dictionary stores the configuration for NWB file
        self.config_loaded = False  # True if NWB configuration is loaded
        self.mode = 'reset'  # 'reset'|'neurons'|'correlation'|'accepted'|'neighbors'
        self.this_cell = None  # The cell index being selected (by mouse click on img1). None if no cell was selected
        self.selected_cells = []  # List of currently selected cells
        self.neighbor_cells = []  # List of neighbor cells (mode 'neighbors')
        self.last_cell = None  # The cell index of the previously selected cell (used in mode 'accepted'|'neighbors')
        self.yx = np.array([-1,-1])  # Mosue clicked position [y,x] coordinates
        
        ## ------------ Central Widget Layout --------------------------------
        cw = QtGui.QWidget()  # Create a central widget to hold everything
        self.layout = QtGui.QGridLayout()  # Create and store the grid layout
        cw.setLayout(self.layout)
        self.setCentralWidget(cw)  
        
        ## ------------ Add Widgets ------------------------------------------
        self.t1 = ParameterTree(showHeader=False)  # For loading data dispaying parameters
        self.t2 = ParameterTree(showHeader=False)  # For action parameters
        self.p1 = pg.PlotWidget()  # For imaging FOV
        self.p2 = pg.PlotWidget()  # For other image (scatter plot...)
        self.p3 = pg.PlotWidget()  # For fluorescence traces
        ## Add widgets to the layout in their proper positions
        self.layout.addWidget(self.t1, 0, 0)  # top-left
        self.layout.addWidget(self.t2, 1, 0)  # bottom-left
        self.layout.addWidget(self.p1, 0, 1)  # top-middle
        self.layout.addWidget(self.p2, 0, 2)   # top-right
        self.layout.addWidget(self.p3, 1, 1, 1, 2)  # bottom-left, spanning 2 columns
        ## Set widget size
        self.t1.setMinimumHeight(320)
        self.t1.setMinimumWidth(270)
        self.t1.setMaximumWidth(360)
        self.t2.setMinimumHeight(320)
        self.t2.setMinimumWidth(270)
        self.t2.setMaximumWidth(360)
        
        ## ------------ Create plot area (ViewBox + axes) --------------------
        self.img1 = pg.ImageItem()
        self.p1.addItem(self.img1)
        self.p1.setAspectLocked()
        self.p1.invertY()  # Set Y axis pointing downward
        self.colorbar = pg.GradientLegend((10,300),(10,10))  # size, offset
        self.colorbar.setParentItem(self.img1)
        gradient = QtGui.QLinearGradient()  # Customize colorbar (jet)
        for i in range(len(cmap)):
            pos = np.linspace(0,1,len(cmap))[i]
            gradient.setColorAt(pos, QtGui.QColor(*cmap[i]))
        self.colorbar.setGradient(gradient)
        self.colorbar.setLabels({'Min':0, 'Max':1})
        self.scatter = pg.ScatterPlotItem(size=12)  # Spot size
        self.scatter.sigClicked.connect(self.scatter_clicked)
        self.p2.addItem(self.scatter)
        self.p2.setLabels(bottom='Rval', left='SNR')
        self.p3.setTitle('Mode: %s' %self.mode)
        self.p3.setLabel('bottom', 'Time (s)')
        self.p3.setMouseEnabled(x=True, y=False)  # Enable only horizontal zoom for displaying traces
        
        ## ------------ Setup munu and parameter tree -------------------------
        self.make_menu()
        self.make_parameter_tree()
        ## ------------ Load data --------------------------------------------
        if datapath is not None:
            self.fname = datapath
            self.load_data(click=False)
        if jsonpath is not None:
            self.fname_json = jsonpath
            self.load_json(click=False)
        ## ------------ Link mouse event -------------------------------------
        # self.p1.mousePressEvent = self.mouse_clicked
        ##  A general rule in Qt is that if you override one mouse event handler, you must override all of them ??
        # self.p1.mouseReleaseEvent = lambda *args: None  # "do nothing" function
        # self.p1.mouseMoveEvent = lambda *args: None
        self.p1.scene().sigMouseClicked.connect(self.mouse_clicked)
    
    def make_menu(self):
        ## Load caiman hdf5 data
        openHDF5 = QtGui.QAction('Open...', self)
        openHDF5.setShortcut('Ctrl+O')
        openHDF5.setStatusTip('Open caiman hdf5 file')
        openHDF5.triggered.connect(lambda: self.load_data(click=True))
        ## Load json for nwb configuration
        loadJSON = QtGui.QAction('Load json...', self)
        loadJSON.setShortcut('Ctrl+L')
        loadJSON.setStatusTip('Load JSON for NWB file configuration')
        loadJSON.triggered.connect(lambda: self.load_json(click=True))
        ## Save (overwrite original hdf5 file)
        saveData = QtGui.QAction('Save', self)
        saveData.setShortcut('Ctrl+S')
        saveData.setStatusTip('Save caiman hdf5 file')
        saveData.triggered.connect(lambda: self.save_data(new=False))
        ## Save as new file (hdf5 or nwb)
        saveAs = QtGui.QAction('Save as...', self)
        saveAs.setShortcut('Ctrl+Shift+S')
        saveAs.setStatusTip('Save current file as...')
        saveAs.triggered.connect(lambda: self.save_data(new=True))
        ## View memap window
        viewMemap = QtGui.QAction('Movie', self)
        viewMemap.setShortcut('Ctrl+M')
        viewMemap.setStatusTip('Display movie from mmap file...')
        viewMemap.triggered.connect(lambda: memap_window(self))
        ## Make main menu
        menu = self.menuBar()
        file_menu = menu.addMenu('&File')
        file_menu.addAction(openHDF5)
        file_menu.addAction(loadJSON)
        file_menu.addAction(saveData)
        file_menu.addAction(saveAs)
        view_menu = menu.addMenu('&View')
        view_menu.addAction(viewMemap)
        
    def make_parameter_tree(self):
        param1 = [
            {'name':'NWB config', 'type':'group','children':[
                {'name':'Sess desc', 'type':'str'},
                {'name':'Sess start t', 'type':'str'},
                {'name':'Experimenter', 'type':'str'},
                {'name':'Exp desc', 'type':'str'}]},
            {'name':'RESET', 'type':'action'},
            {'name':'NEURONS', 'type':'action'},
            {'name':'CORRELATION', 'type':'action'},
            {'name':'Image', 'type':'list', 'values':['Corr','PNR','Max','Mean','Std'], 'value':'PNR'},
            {'name':'Metric', 'type':'list', 'values':['Rval','SNR','Mean paircorr','Max paircorr'], 'value':'Rval'},
            {'name':'Trace', 'type':'list', 'values':['Raw','Denoised','Spike'], 'value':'Raw'},
            {'name':'ACCEPTED', 'type':'action'},
            {'name':'NEIGHBORS', 'type':'action'},
            {'name':'Contour thr', 'type':'float', 'value':0.2, 'limits':(0,1), 'step':0.01},
            {'name':'Contour pix', 'type':'int', 'value':1, 'limits':(1,6), 'step':1},
            {'name':'Dist pix', 'type':'int', 'value':100, 'limits':(0,1000), 'step':5},
            {'name':'Cell ID', 'type':'int', 'value':-1, 'limits':(-1,1000), 'step':1}
        ]
        self.par1 = Parameter.create(name='Parameters Display', type='group', children=param1)
        self.t1.setParameters(self.par1, showTop=True)
        self.par1.child('NWB config').sigTreeStateChanged.connect(self.change_config)
        self.par1.param('RESET').sigActivated.connect(self.reset_button)
        self.par1.param('NEURONS').sigActivated.connect(self.neurons_button)
        self.par1.param('CORRELATION').sigActivated.connect(self.correlation_button)
        self.par1.param('Image').sigValueChanged.connect(self.change_image)
        self.par1.param('Metric').sigValueChanged.connect(self.change_metric)
        self.par1.param('Trace').sigValueChanged.connect(self.draw_trace)
        self.par1.param('ACCEPTED').sigActivated.connect(self.accepted_button)
        self.par1.param('NEIGHBORS').sigActivated.connect(self.neighbors_button)
        self.par1.param('Contour thr').sigValueChanged.connect(self.draw_fov_overall)
        self.par1.param('Contour pix').sigValueChanged.connect(self.draw_fov_overall)
        self.par1.param('Dist pix').sigValueChanged.connect(self.draw_fov_overall)
        self.par1.param('Cell ID').sigValueChanged.connect(self.change_cell)
        
        param2 = [
            {'name':'View components', 'type':'list', 'values':['All','Accepted','Rejected','Unassigned'], 'value':'All'},
            {'name':'Filter components', 'type':'bool', 'value':True, 'tip':'Filter components'},          
            {'name':'Quality thr','type':'group','children':[
                {'name':'Rval high', 'type':'float', 'value':0.85, 'limits':(-1,1), 'step':0.01},
                {'name':'Rval low', 'type':'float', 'value':-1, 'limits':(-1,1), 'step':0.01},
                {'name':'SNR high', 'type':'float', 'value':2, 'limits':(0,20), 'step':0.1},
                {'name':'SNR low', 'type':'float', 'value':0, 'limits':(0,20), 'step':0.1},
                {'name':'CNN high', 'type':'float', 'value':0.99, 'limits':(0,1), 'step':0.01},
                {'name':'CNN low', 'type':'float', 'value':0.1, 'limits':(0,1), 'step':0.01}]},
            {'name':'ADD GROUP', 'type':'action'},
            {'name':'REMOVE GROUP', 'type':'action'},
            {'name':'MERGE', 'type':'action'},
            {'name':'ADD SELECTED', 'type':'action'},
            {'name':'REMOVE SELECTED', 'type':'action'},
            {'name':'Info', 'type':'text'}
        ]
        self.par2 = Parameter.create(name='Parameters Action', type='group', children=param2)
        self.t2.setParameters(self.par2, showTop=True)
        self.par2.param('View components').sigValueChanged.connect(self.change_list)
        self.par2.param('Filter components').sigValueChanged.connect(self.change_list)
        self.par2.child('Quality thr').sigTreeStateChanged.connect(self.change_list)
        self.par2.param('MERGE').sigActivated.connect(self.merge_components)
        self.par2.param('ADD GROUP').sigActivated.connect(self.add_group)
        self.par2.param('REMOVE GROUP').sigActivated.connect(self.remove_group)
        self.par2.param('ADD SELECTED').sigActivated.connect(self.add_selected)
        self.par2.param('REMOVE SELECTED').sigActivated.connect(self.remove_selected)
        
    def load_data(self, click=True):
        self.loaded = False
        if click:
            fname = FileDialog().getOpenFileName(
                caption='Load CNMF Object', filter='HDF5 (*.h5 *.hdf5)')[0]  # ;;NWB (*.nwb)
            self.fname = fname
        try:
            self.cnmf = load_CNMF(self.fname)
            self.loaded = True
            self.statusBar().showMessage('Loading '+self.fname)
        except Exception:
            self.statusBar().showMessage('Loading '+self.fname+' failed. Try other file...')
        
        if self.loaded:
            dims = self.cnmf.estimates.dims  # (y,x)
            A = self.cnmf.estimates.A  # csc_matrix shape (N,K) where N=xy
            img_components = [A[:,i].reshape(dims,order='F').toarray() for i in range(A.shape[1])]
            self.cms = np.array([center_of_mass(comp) for comp in img_components])  # Shape (K,2) centroid (y,x) of each component
            self.img_components = np.stack(
                [normalize_image(comp) for comp in img_components], axis=0)
            ## Store a background image (initially PNR image)
            self.image = normalize_image(self.cnmf.pnr, stretch_prct=True, rgb=True)
            
            ## Compute pairwise correlation and store the metrics (initially r_values)
            # corr_matrix = np.corrcoef(self.cnmf.estimates.C)  # Denoised traces
            corr_matrix = np.corrcoef(self.cnmf.estimates.C + self.cnmf.estimates.YrA)
            np.fill_diagonal(corr_matrix, np.NaN)
            self.corr_matrix = corr_matrix
            rval = self.cnmf.estimates.r_values
            self.colorbar.setLabels({f'{rval.min():.2f}':0, f'{rval.max():.2f}':1})
            self.metric = (rval-rval.min())/(rval.max()-rval.min())
            
            ## Build accepted and rejected list
            accepted_empty = True
            if hasattr(self.cnmf.estimates, 'accepted_list'):
                accepted_empty = (len(self.cnmf.estimates.accepted_list)==0)  # False if accepted_list already contains neuron indices 
            if accepted_empty:
                self.cnmf.estimates.accepted_list = np.array([], dtype=int)
                self.cnmf.estimates.rejected_list = np.array([], dtype=int)
            self.par2.param('View components').setValue('All', blockSignal=self.change_list)
            self.reset_button()
            self.print_info()
            
            ## Set realistic limits
            max_dist = np.sqrt(np.sum(np.array(dims)**2))
            self.par1.param('Dist pix').setLimits((0,int(max_dist)))
            self.par1.param('Cell ID').setLimits((-1,A.shape[1]-1))
            ## Display quality threshold (this may trigger change_list)
            quality = self.cnmf.params.quality
            self.par2.child('Quality thr').param('Rval high').setValue(quality['rval_thr'])
            self.par2.child('Quality thr').param('Rval low').setValue(quality['rval_lowest'])
            self.par2.child('Quality thr').param('SNR high').setValue(quality['min_SNR'])
            self.par2.child('Quality thr').param('SNR low').setValue(quality['SNR_lowest'])
            self.par2.child('Quality thr').param('CNN high').setValue(quality['min_cnn_thr'])
            self.par2.child('Quality thr').param('CNN low').setValue(quality['cnn_lowest'])
            self.par2.param('Filter components').setValue(True)
            self.statusBar().showMessage('Loaded: '+self.fname)
            
    def load_json(self, click=True):
        self.config_loaded = False
        if click:
            fname_json = FileDialog().getOpenFileName(
                caption='Load json file for NWB configuration', filter='JSON (*.json)')[0]
            self.fname_json = fname_json
        try:   
            with open(self.fname_json, 'r') as f:
                self.config = json.load(f)
            ## Set initial NWB metadata using the json file    
            self.par1.child('NWB config').param('Sess desc').setValue(
                self.config['nwbfile']['session_description'])
            self.par1.child('NWB config').param('Sess start t').setValue(
                self.config['nwbfile']['session_start_time'])
            self.par1.child('NWB config').param('Experimenter').setValue(
                self.config['nwbfile']['experimenter'])
            self.par1.child('NWB config').param('Exp desc').setValue(
                self.config['nwbfile']['experiment_description'])
            self.config_loaded = True
        except Exception:
            self.statusBar().showMessage('Loading '+self.fname_json+' failed. Try other file...')
        
    def change_config(self):
        '''Update self.config dictionary from user input
        '''
        if self.config_loaded:
            self.config['nwbfile']['session_description'] = \
                self.par1.child('NWB config').param('Sess desc').value()
            self.config['nwbfile']['session_start_time'] = \
                self.par1.child('NWB config').param('Sess start t').value()
            self.config['nwbfile']['experimenter'] = \
                self.par1.child('NWB config').param('Experimenter').value()
            self.config['nwbfile']['experiment_description'] = \
                self.par1.child('NWB config').param('Exp desc').value()

    # %% Various setup
    def reset_init(self, accepted=False):
        self.p1.setTitle('FOV')
        self.p2.setTitle('Metrics')
        self.p3.clearPlots()
        self.p3.setTitle('Mode: %s' %self.mode)
        if accepted:
            accepted_list = self.cnmf.estimates.accepted_list
            if len(accepted_list) > 0:
                self.this_cell = accepted_list[0]
                self.selected_cells = [self.this_cell]
                self.neighbor_cells = []
                self.last_cell = accepted_list[0]
                self.par1.param('Cell ID').setValue(self.this_cell)
        else:
            self.this_cell = None
            self.selected_cells = []
            self.neighbor_cells = []
            self.last_cell = None
            self.par1.param('Cell ID').setValue(-1, blockSignal=self.change_cell)
            
    def reset_button(self):
        self.mode = 'reset'
        self.reset_init()
        self.draw_contours()
        self.p1.autoRange()
        self.draw_scatter()
        self.p2.autoRange()
        
    def neurons_button(self):
        self.mode = 'neurons'
        self.reset_init()
        self.draw_colormap()
        self.draw_scatter()
        
    def correlation_button(self):
        self.mode = 'correlation'
        self.reset_init()
        self.draw_contours()
        self.draw_scatter()
    
    def accepted_button(self):
        self.mode = 'accepted'
        self.reset_init(accepted=True)
        self.par2.param('Filter components').setValue(False)
        self.par2.param('View components').setValue('Accepted')
        self.draw_fov_overall()
        self.draw_scatter()
        self.draw_trace()
    
    def neighbors_button(self):
        self.mode = 'neighbors'
        self.reset_init(accepted=True)
        self.par2.param('Filter components').setValue(False)
        self.par2.param('View components').setValue('Accepted')
        accepted_list = self.cnmf.estimates.accepted_list
        if len(accepted_list) > 0:
            radius = self.par1.param('Dist pix').value() 
            distances = np.sqrt(np.sum(
                (self.cms[self.this_cell] - self.cms[accepted_list])**2, axis=1))
            self.neighbor_cells = np.setdiff1d(
                accepted_list[distances<radius], self.this_cell)
        self.draw_fov_overall()
        self.draw_scatter()
        self.draw_trace()
        
    def print_info(self):
        K = self.cnmf.estimates.C.shape[0]
        idx_components = self.cnmf.estimates.idx_components
        accepted_list = self.cnmf.estimates.accepted_list
        self.par2.param('Info').setValue(
            os.path.split(self.fname)[-1] + '\n' 
            + '-'*32 + '\n' 
            + f'Total components: {K}\n'
            + f'Current group: {len(idx_components)}\n'
            + f'Accepted: {len(accepted_list)}\n')  # + str([*accepted_list])
            
    def change_image(self):
        img_to_plot = self.par1.param('Image').value()
        if img_to_plot=='PNR':
            self.image = normalize_image(self.cnmf.pnr, stretch_prct=True, rgb=True)
        elif img_to_plot=='Corr':
            self.image = normalize_image(self.cnmf.cn_filter, stretch_prct=True, rgb=True)
        elif hasattr(self.cnmf,'image_max') and img_to_plot=='Max':
            self.image = normalize_image(self.cnmf.image_max, stretch_prct=True, rgb=True)
        elif hasattr(self.cnmf,'image_mean') and img_to_plot=='Mean':
            self.image = normalize_image(self.cnmf.image_mean, stretch_prct=True, rgb=True)
        elif hasattr(self.cnmf,'image_std') and img_to_plot=='Std':
            self.image = normalize_image(self.cnmf.image_std, stretch_prct=True, rgb=True)
        self.draw_fov_overall()
    
    def change_metric(self, plot=True):
        metric = self.par1.param('Metric').value()
        if metric=='Rval':
            scores = self.cnmf.estimates.r_values
        elif metric=='SNR':
            scores = self.cnmf.estimates.SNR_comp
        elif metric=='Mean paircorr':
            scores = np.nanmean(self.corr_matrix, axis=1)
        elif metric=='Max paircorr':
            scores = np.nanmax(self.corr_matrix, axis=1)
        self.colorbar.setLabels({f'{scores.min():.2f}':0, f'{scores.max():.2f}':1})
        self.metric = (scores-scores.min())/(scores.max()-scores.min())  # Normalized to [0,1]
        if plot:
            self.draw_fov_overall()
            self.draw_scatter()
        
    # %% Plotting        
    def draw_contours(self):
        '''Draw contours of all cells in the current group (idx_components) with color scaled to the metric
        '''
        thisImg = self.image.copy()
        thrsh = self.par1.param('Contour thr').value()  # Contour threshold (0,1)
        thick = self.par1.param('Contour pix').value()  # Contour thickness (int pixel)
        idx_components = self.cnmf.estimates.idx_components
        if len(idx_components) > 0:
            colors = plt.cm.jet(self.metric, bytes=True)[:,:3].tolist()  # All colors
            for idx in idx_components:
                img = self.img_components[idx]
                contour = cv2.findContours(cv2.threshold(img,int(255*thrsh),255,0)[1],
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(thisImg, contour, -1, colors[idx], thick)
        self.img1.setImage(thisImg, autoLevels=False)
    
    def draw_colormap(self):
        '''Draw colormap of all cells in the current group (idx_components) with color scaled to the metric
        '''
        thisImg = self.image[:,:,0].copy()  # dtype=np.uint8
        dims = self.cnmf.estimates.dims
        idx_components = self.cnmf.estimates.idx_components
        img_rgb = color_cells(thisImg, self.cnmf.estimates.A, dims,
                              self.metric, idx_components)
        self.img1.setImage(img_rgb, autoLevels=False)
        
    def draw_fov_update(self):
        '''
        Update FOV by drawing contours of the selected cells (mouse clicked)
        and colormap of the other cells in the current group (idx_components).
        Mode 'neurons' or 'correlation'
        '''
        thisImg = self.image[:,:,0].copy()
        idx_components = self.cnmf.estimates.idx_components
        if len(idx_components) > 0:
            dims = self.cnmf.estimates.dims
            idx_remain = np.setdiff1d(idx_components, self.selected_cells)
            if self.mode == 'neurons':
                thisImg = color_cells(thisImg, self.cnmf.estimates.A, dims,
                                      self.metric, idx_remain)
            elif self.mode == 'correlation':
                scores = self.corr_matrix[self.this_cell]
                self.colorbar.setLabels({f'{np.nanmin(scores):.2f}':0, f'{np.nanmax(scores):.2f}':1})
                scores = (scores-np.nanmin(scores))/(np.nanmax(scores)-np.nanmin(scores))
                thisImg = color_cells(thisImg, self.cnmf.estimates.A, dims,
                                      scores, idx_remain)
            thrsh = self.par1.param('Contour thr').value()
            thick = self.par1.param('Contour pix').value()
            for i, idx in enumerate(self.selected_cells):
                ii = i % len(self.colors)  # Remainder
                img = self.img_components[idx]
                contour = cv2.findContours(cv2.threshold(img,int(255*thrsh),255,0)[1],
                                           cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
                cv2.drawContours(thisImg, contour, -1, self.colors[ii], thick)
        else:
            thisImg = np.dstack([thisImg]*3)
        self.img1.setImage(thisImg, autoLevels=False)

    def draw_fov_keyupdate(self):
        ''' Mode 'accepted' or 'neighbors'
        Draw the contour of this_cell and colormap of other cells in the accepted list.
        '''
        thisImg = self.image[:,:,0].copy()
        accepted_list = self.cnmf.estimates.accepted_list
        if len(accepted_list) > 0:
            dims = self.cnmf.estimates.dims
            thrsh = self.par1.param('Contour thr').value()
            thick = self.par1.param('Contour pix').value()
            if self.mode == 'accepted':
                idx_remain = np.setdiff1d(accepted_list, self.this_cell)        
                thisImg = color_cells(thisImg, self.cnmf.estimates.A, dims,
                                      self.metric, idx_remain)
            elif self.mode == 'neighbors':
                scores = self.corr_matrix[self.this_cell]
                self.colorbar.setLabels({f'{np.nanmin(scores):.2f}':0, f'{np.nanmax(scores):.2f}':1})
                scores = (scores-np.nanmin(scores))/(np.nanmax(scores)-np.nanmin(scores))
                ## Draw colormap of other cells
                thisImg = color_cells(thisImg, self.cnmf.estimates.A, dims,
                                      scores, self.neighbor_cells)
            ## Draw contour of this_cell
            img = self.img_components[self.this_cell]
            contour = cv2.findContours(cv2.threshold(img,int(255*thrsh),255,0)[1],
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(thisImg, contour, -1, (255,0,255), thick)  # self.colors[3] -> red (plt.cm.Set3)
        else:
            thisImg = np.dstack([thisImg]*3)
        self.img1.setImage(thisImg, autoLevels=False)
    
    def draw_fov_overall(self):
        if self.mode == 'reset':
            self.draw_contours()
        elif self.mode == 'neurons':
            if self.this_cell is None:  # No specific neuron has been selected
                self.draw_colormap()
            else:  # One neuron has been selected by mouse click
                self.draw_fov_update()
        elif self.mode == 'correlation':
            if self.this_cell is None:
                self.draw_contours()
            else:
                self.draw_fov_update()
        elif self.mode in {'accepted', 'neighbors'}:
            if self.this_cell is None:
                self.draw_contours()
            else:
                self.draw_fov_keyupdate()

    def draw_scatter(self):
        '''Scatter plot of neuron quality (Rval & SNR) and color spots with the specified metric
        '''
        self.scatter.clear()
        idx_components = self.cnmf.estimates.idx_components
        rval = self.cnmf.estimates.r_values
        snr = self.cnmf.estimates.SNR_comp
        rgba = plt.cm.jet(self.metric, bytes=True, alpha=0.8)  # Shape (K,4) dtype=np.uint8
        idx_other = np.setdiff1d(np.arange(len(snr)), idx_components)
        rgba[idx_other] = np.array([128,128,128,204])  # Gray for other cells
        spots = []
        for i in range(len(rval)):
            spots.append({'pos':(rval[i], snr[i]), 'data':i,
                          'brush':rgba[i]})
        self.scatter.setData(spots)
        for i in self.selected_cells:
            self.scatter.points()[i].setPen('w', width=2)
            
    def draw_trace(self):
        '''Plot fluorescence traces of selected cells with predefined color cycle (self.colors -> plt.cm.Set3)
        '''
        self.p3.clearPlots()
        fr = self.cnmf.params.data['fr']
        T = self.cnmf.estimates.C.shape[1]
        trace = self.par1.param('Trace').value()
        if self.mode in {'neurons', 'correlation'}:
            for i, idx in enumerate(self.selected_cells):  # Replot for all cells
                if trace == 'Raw':
                    f = self.cnmf.estimates.C[idx] + self.cnmf.estimates.YrA[idx]
                elif trace == 'Denoised':
                    f = self.cnmf.estimates.C[idx]
                elif trace == 'Spike':
                    f = self.cnmf.estimates.S[idx]
                if self.mode == 'neurons':
                    ii = i % len(self.colors)  # Remainder (cyclic colors if selected more than 12 cells)
                    self.p3.plot(np.arange(T)/fr, i+f/f.max(), pen=self.colors[ii])  # Normalized trace and shifted vertically 
                elif self.mode == 'correlation':  # Only one cell selected is possible
                    self.p3.plot(np.arange(T)/fr, f, pen=self.colors[0])  # Original intensity
        elif self.mode in {'accepted', 'neighbors'}:
            idx = self.this_cell
            if trace == 'Raw':
                f = self.cnmf.estimates.C[idx] + self.cnmf.estimates.YrA[idx]
            elif trace == 'Denoised':
                f = self.cnmf.estimates.C[idx]
            elif trace == 'Spike':
                f = self.cnmf.estimates.S[idx]
            if self.mode == 'accepted':
                self.p3.plot(np.arange(T)/fr, f, pen='m')  # self.colors[0]
            elif self.mode == 'neighbors':
                self.p3.plot(np.arange(T)/fr, f/f.max(), pen='m')  # self.colors[3]
                scores = self.corr_matrix[self.this_cell]
                scores = (scores-np.nanmin(scores))/(np.nanmax(scores)-np.nanmin(scores))
                colors = plt.cm.jet(scores, bytes=True)[:,:3].tolist()
                if len(self.neighbor_cells) > 0:
                    orders = np.argsort(scores[self.neighbor_cells])[::-1]  # Sort from high to low correlation
                    for i, idx in enumerate(self.neighbor_cells[orders]):  # High correlation component is close to the selected cell
                        if trace == 'Raw':
                            f = self.cnmf.estimates.C[idx] + self.cnmf.estimates.YrA[idx]
                        elif trace == 'Denoised':
                            f = self.cnmf.estimates.C[idx]
                        elif trace == 'Spike':
                            f = self.cnmf.estimates.S[idx]
                        self.p3.plot(np.arange(T)/fr, i+1+f/f.max(), pen=colors[idx])
                
    # %% Mouse and keyboard interaction
    def update_selection(self, this_cell):
        '''Update this_cell, selected_cells and displaying items after mouse clicked on img1 or scatter plot item.
        '''
        if self.mode == 'neurons':
            if this_cell in self.selected_cells:  # Already selected, remove this_cell
                self.selected_cells.remove(this_cell)  # Click twice removes selected cell
                self.scatter.points()[this_cell].resetPen()
                if len(self.selected_cells) > 0:
                    self.this_cell = self.selected_cells[-1]
                else:
                    self.this_cell = None
            else:  # Not been selected, append this_cell
                self.selected_cells.append(this_cell)  # Multiple cells are selected
                self.scatter.points()[this_cell].setPen('w', width=2)
                self.this_cell = this_cell
        elif self.mode == 'correlation':
            for i in self.selected_cells:  # There should be only one cell or empty in the list
                self.scatter.points()[i].resetPen()
            self.scatter.points()[this_cell].setPen('w', width=2)
            self.selected_cells = [this_cell]  # Override as only one cell selected is possible in correlation mode
            self.this_cell = this_cell
            
        if self.this_cell is None:
            self.p1.setTitle('')
            self.p2.setTitle('')
            self.par1.param('Cell ID').setValue(-1, blockSignal=self.change_cell)
        else:
            self.p1.setTitle('Component %d' %self.this_cell)  # *self.yx
            self.p2.setTitle('Rval: %.3f SNR: %.2f'
                             %(self.cnmf.estimates.r_values[self.this_cell],
                               self.cnmf.estimates.SNR_comp[self.this_cell]))
            self.par1.param('Cell ID').setValue(self.this_cell,
                                                blockSignal=self.change_cell)
        self.draw_fov_overall()
        self.draw_trace()
            
    def mouse_clicked(self, event):
        '''Determine the mouse clicked position (y,x) on img1 to infer this_cell
        '''
        if self.mode in {'neurons', 'correlation'}:
            dims = self.cnmf.estimates.dims
            # pos = self.img1.mapFromScene(event.pos())  # event from self.p1.mousePressEvent -> override all mouse press event
            pos = self.img1.mapFromScene(event.scenePos())  # event from self.p1.scene().sigMouseClicked -> preserve other mouse utilities e.g. drag to span... 
            x, y = pos.x(), pos.y()
            if x>=0 and x<=dims[1] and y>=0 and y<=dims[0]:  # Do nothing if user clicks outside the image
                # i = int(np.clip(y, 0, dims[0] - 1))  # Value outside the interval are clipped to the edges
                # j = int(np.clip(x, 0, dims[1] - 1))
                self.yx = np.array([y, x])
                idx_components = self.cnmf.estimates.idx_components
                distances = np.sum((self.yx - self.cms[idx_components])**2, axis=1)  # **0.5
                this_cell = idx_components[np.argmin(distances)]  # components_to_plot
                self.update_selection(this_cell)
    
    def scatter_clicked(self, plot, points):
        '''Get the cell ID of the clicked spot on the scatter plot. Argument points is a list of points under the clicked mouse curser.
        '''
        if self.mode in {'neurons', 'correlation'}:
            this_cell = points[0].data()
            self.update_selection(this_cell)
        
    def keyPressEvent(self, event):
        '''Override the existing method to activate left/right key to scroll through the cell ID
        '''
        if self.mode in {'accepted', 'neighbors'}:
            accepted_list = self.cnmf.estimates.accepted_list
            K2 = len(accepted_list)
            if K2 > 0 and self.this_cell is not None:
                # if event.modifiers() !=  QtCore.Qt.ShiftModifier:
                self.last_cell = self.this_cell
                last_i = np.where(accepted_list==self.last_cell)[0].item()
                if event.key() == QtCore.Qt.Key_Left:
                    this_i = np.clip(last_i-1, 0, K2-1)
                    self.this_cell = accepted_list[this_i]
                elif event.key() == QtCore.Qt.Key_Right:
                    this_i = np.clip(last_i+1, 0, K2-1)
                    self.this_cell = accepted_list[this_i]
                self.selected_cells = [self.this_cell]
                self.par1.param('Cell ID').setValue(
                    self.this_cell, blockSignal=self.change_cell)
                self.p1.setTitle('Component %d' %self.this_cell)
                self.p2.setTitle('Rval: %.3f SNR: %.2f'
                                 %(self.cnmf.estimates.r_values[self.this_cell],
                                   self.cnmf.estimates.SNR_comp[self.this_cell]))
                self.scatter.points()[self.last_cell].resetPen()
                self.scatter.points()[self.this_cell].setPen('w', width=2)
                ## Compute neighbor cells
                if self.mode == 'neighbors':
                    radius = self.par1.param('Dist pix').value() 
                    distances = np.sqrt(np.sum(
                        (self.cms[self.this_cell] - self.cms[accepted_list])**2, axis=1))
                    self.neighbor_cells = np.setdiff1d(
                        accepted_list[distances<radius], self.this_cell)
                self.draw_fov_keyupdate()
                self.draw_trace()
    
    def change_cell(self):
        '''
        Act when the Cell ID parameter is changed by user (update only single cell selection).
        Block this signal when this_cell is determined by mouse clicked and keyboard interaction!!
        '''
        self.this_cell = self.par1.param('Cell ID').value()
        self.p1.setTitle('Component %d' %self.this_cell)
        self.p2.setTitle('Rval: %.3f SNR: %.2f'
                         %(self.cnmf.estimates.r_values[self.this_cell],
                           self.cnmf.estimates.SNR_comp[self.this_cell]))
        for i in self.selected_cells:  # There should be only one cell or empty in the list
            self.scatter.points()[i].resetPen()
        self.scatter.points()[self.this_cell].setPen('w', width=2)
        self.selected_cells = [self.this_cell]
        ## Compute neighbor cells
        if self.mode == 'neighbors':
            accepted_list = self.cnmf.estimates.accepted_list
            radius = self.par1.param('Dist pix').value() 
            distances = np.sqrt(np.sum(
                (self.cms[self.this_cell] - self.cms[accepted_list])**2, axis=1))
            self.neighbor_cells = np.setdiff1d(
                accepted_list[distances<radius], self.this_cell)
        self.draw_fov_overall()
        self.draw_trace()
        
    # %% Merge components
    def merge_components(self):
        '''Merge components in the selected_cells list and update cnmf object.
        '''
        if len(self.selected_cells) > 1:
            K = self.cnmf.estimates.C.shape[0]
            K1 = K - len(self.selected_cells) + 1  # Number of total components after merge
            A_merge = self.cnmf.estimates.A[:,self.selected_cells]
            C_merge = self.cnmf.estimates.C[self.selected_cells,:] + \
                self.cnmf.estimates.YrA[self.selected_cells,:]
            computedA, computedC = merge_iteration(A_merge, C_merge)
            deconvC, bl, c1, g, sn, sp, lam = constrained_foopsi(
                computedC, g=None, **self.cnmf.params.get_group('temporal'))
            keep = np.setdiff1d(np.arange(K), self.selected_cells)
            ## Update estimates
            self.cnmf.estimates.A = sparse.hstack([self.cnmf.estimates.A[:,keep], computedA])
            self.cnmf.estimates.C = np.vstack([self.cnmf.estimates.C[keep,:], deconvC])
            self.cnmf.estimates.YrA = np.vstack([self.cnmf.estimates.YrA[keep,:], computedC-deconvC])
            self.cnmf.estimates.S = np.vstack([self.cnmf.estimates.S[keep,:], sp])
            self.cnmf.estimates.bl = np.hstack([self.cnmf.estimates.bl[keep], bl])
            self.cnmf.estimates.c1 = np.hstack([self.cnmf.estimates.c1[keep], c1])
            self.cnmf.estimates.sn = np.hstack([self.cnmf.estimates.sn[keep], sn])
            self.cnmf.estimates.g = np.vstack([self.cnmf.estimates.g[keep], g])  # Shape (K1,p) where p=1 or 2
            self.cnmf.estimates.nr = K1
            ## Update metrics (approximately, not re-evaluated -> need C-memmap)
            rval = np.mean(self.cnmf.estimates.r_values[self.selected_cells])
            snr = np.max(self.cnmf.estimates.SNR_comp[self.selected_cells])
            self.cnmf.estimates.r_values = np.hstack([self.cnmf.estimates.r_values[keep], rval])
            self.cnmf.estimates.SNR_comp = np.hstack([self.cnmf.estimates.SNR_comp[keep], snr])
            ## Update img_components, center of mass, metrics (pairwise correlation...)
            dims = self.cnmf.estimates.dims
            img_merged = computedA.reshape(dims, order='F').toarray()
            self.img_components = np.concatenate(
                [self.img_components[keep,:,:], normalize_image(img_merged)[np.newaxis,:,:]], axis=0)
            self.cms = np.vstack([self.cms[keep,:], np.array(center_of_mass(img_merged))])  # Shape (K1,2)
            self.corr_matrix = np.corrcoef(self.cnmf.estimates.C)
            np.fill_diagonal(self.corr_matrix, np.NaN)
            ## Update accepted list, figures...
            self.cnmf.estimates.accepted_list = update_list(K, self.cnmf.estimates.accepted_list, self.selected_cells)
            self.cnmf.estimates.rejected_list = update_list(K, self.cnmf.estimates.rejected_list, self.selected_cells)
            self.this_cell = K1-1  # The merged component
            self.selected_cells = [K1-1]
            self.change_metric(plot=False)
            self.change_list(None, None)
            
    # %% Change idx_components, accepted_list and save data
    def add_group(self):
        '''Add all current components to the accepted list
        '''
        self.cnmf.estimates.accepted_list = \
            np.union1d(self.cnmf.estimates.accepted_list, self.cnmf.estimates.idx_components)  # union of two arrays
        self.cnmf.estimates.rejected_list = \
            np.setdiff1d(self.cnmf.estimates.rejected_list, self.cnmf.estimates.idx_components)  # unique values in arg1 that are not in arg2
        self.change_list(None, None)
        
    def remove_group(self):
        '''Remove all current components from the accepted list and put them into the rejected list
        '''
        self.cnmf.estimates.rejected_list = \
            np.union1d(self.cnmf.estimates.rejected_list, self.cnmf.estimates.idx_components)
        self.cnmf.estimates.accepted_list = \
            np.setdiff1d(self.cnmf.estimates.accepted_list, self.cnmf.estimates.idx_components)
        self.change_list(None, None)
        
    def add_selected(self):
        '''Add the current selected component to the accepted list
        '''
        self.cnmf.estimates.accepted_list = \
            np.union1d(self.cnmf.estimates.accepted_list, self.selected_cells)
        self.cnmf.estimates.rejected_list = \
            np.setdiff1d(self.cnmf.estimates.rejected_list, self.selected_cells)
        self.this_cell = None
        self.selected_cells = []
        self.change_list(None, None)
        
    def remove_selected(self):
        '''Remove the current selected component from the accepted list and put it into the rejected list
        '''
        self.cnmf.estimates.rejected_list = \
            np.union1d(self.cnmf.estimates.rejected_list, self.selected_cells)
        self.cnmf.estimates.accepted_list = \
            np.setdiff1d(self.cnmf.estimates.accepted_list, self.selected_cells)
        if self.mode in {'neurons', 'correlation'}:
            self.this_cell = None
            self.p1.setTitle('')
            self.p2.setTitle('')
        elif self.mode in {'accepted','neighbors'}:
            if self.this_cell != self.last_cell:
                self.this_cell = self.last_cell  ## Keep this_cell visible as last_cell
            else:
                self.this_cell = self.cnmf.estimates.accepted_list[0]
            self.p1.setTitle('Component %d' %self.this_cell)
            self.p2.setTitle('Rval: %.3f SNR: %.2f'
                             %(self.cnmf.estimates.r_values[self.this_cell],
                               self.cnmf.estimates.SNR_comp[self.this_cell]))
        self.selected_cells = []
        self.change_list(None, None)
        
    def change_list(self, param, changes):  # param, changes are positional (required!!) arguments connected to the Parameter class, not used here
        '''Act when quality thresholds are modified or Filter components, View components state is changed
        '''
        K = self.cnmf.estimates.C.shape[0]
        accepted_list = self.cnmf.estimates.accepted_list
        rejected_list = self.cnmf.estimates.rejected_list
        if self.par2.param('Filter components').value():
            set_par = self.par2.child('Quality thr').getValues()
            par_dict = {'rval_thr': set_par['Rval high'][0],
                        'rval_lowest': set_par['Rval low'][0],
                        'min_SNR': set_par['SNR high'][0],
                        'SNR_lowest': set_par['SNR low'][0],
                        'min_cnn_thr': set_par['CNN high'][0],
                        'cnn_lowest': set_par['CNN low'][0]}
            self.cnmf.params.quality.update(par_dict)  # Renew caiman thresholds
            # estimates.filter_components(mov, params_obj, dview=None,
            #                             select_mode=select_mode)  # Need mov here ??
            ## ======== Filter Components ============ ##
            ## Rval/SNR low and high thresholds (CNN not used here)
            rval = self.cnmf.estimates.r_values
            snr = self.cnmf.estimates.SNR_comp
            low_thr = (rval>=set_par['Rval low'][0]) & (snr>=set_par['SNR low'][0])
            high_thr = (rval>=set_par['Rval high'][0]) | (snr>=set_par['SNR high'][0])
            good_idx = np.arange(K)[low_thr & high_thr]
        else:  # Not to filter components (Original Caiman GUI: filter with default setting -> this makes thing complicated...)
            good_idx = np.arange(K)
        select_mode = self.par2.param('View components').value()  # 'All'|'Accepted'|'Rejected'|'Unassigned'
        if select_mode == 'Accepted':
            good_idx = np.intersect1d(good_idx, accepted_list)
        elif select_mode == 'Rejected':
            good_idx = np.intersect1d(good_idx, rejected_list)
        elif select_mode == 'Unassigned':
            good_idx = np.setdiff1d(good_idx,
                                    np.union1d(rejected_list, accepted_list)) 
        bad_idx = np.setdiff1d(np.arange(K), good_idx)
        self.cnmf.estimates.idx_components = good_idx
        self.cnmf.estimates.idx_components_bad = bad_idx
        self.draw_fov_overall()
        self.draw_scatter()
        self.draw_trace()
        self.print_info()
        
    def save_data(self, new=True):
        if new:
            fname_save = FileDialog().getSaveFileName(filter='HDF5 (*.hdf5);;NWB (*.nwb)')[0]
        else:
            fname_save = self.fname
        if os.path.splitext(fname_save)[1] == '.hdf5':
            self.cnmf.save(fname_save)
        elif os.path.splitext(fname_save)[1] == '.nwb':
            import nwb
            nwb.save_nwb(self.cnmf, fname_save, self.config, raw_data_file=None)
        self.statusBar().showMessage('Saved: '+fname_save)
        
# %% Useful functions
def normalize_image(img, stretch_prct=False, prct=(1,99), rgb=False):
    '''
    Normalize image to 'uint8' for use in OpenCV

    Parameters
    ----------
    img : numpy 2D array
        Input image.
    stretch_prct : bool
        Whether to apply a percentile stretching. The default is False (min-max streching).
    prct : tuple of two values between 0 and 100
        The low and high percentile used if stretch_prct. The default is (1,99)
    rgb : bool
        Whether to return a RGB stack. The default is False.

    Returns
    -------
    img2 : numpy 2D or 3D array
        Normalized image, shape (h,w) or (h,w,3).
    '''
    if stretch_prct:
        min_, max_ = np.percentile(img, prct)
        img2 = np.clip((img.copy()-min_)/(max_-min_)*255,0,255).astype('uint8')
    else:  # min/max normalization
        min_, max_ = img.min(), img.max()
        img2 = ((img.copy()-min_)/(max_-min_)*255).astype('uint8')
    if rgb:
        img2 = np.dstack([img2]*3)
    return img2

def color_cells(background, A, dims, scores, list_cells):
    '''
    Overlay a gray scale background image with a list of colored cells. 

    Parameters
    ----------
    background : numpy 2d array, shape (y,x)
    A : scipy.sparse.csc_matrix, shape (N,K) where N=xy and K total number of components
    dims : list or tuple, (y,x) pixels of the FOV
    scores : numpy 1d array, shape (K,)
        Metric used to color cells
    list_cells : list or numpy 1d array
        List of cells to color

    Returns
    -------
    img_rgb : numpy 3d array, shape (y,x,3)
        Background image with colored cells
    '''
    if len(list_cells) > 0:
        # scores = (scores-scores.min())/(scores.max()-scores.min())
        rgb = plt.cm.jet(scores)[:,:3]  # 0 to 1
        hsv = mcolors.rgb_to_hsv(rgb)  # 0 to 1
        hue_list = (hsv[:,0]*179).astype(np.uint8)  # OpenCV hue range is [0,179]
        H = np.zeros(np.prod(dims), dtype=np.uint8)
        S = np.zeros(np.prod(dims), dtype=np.uint8)
        for idx in list_cells:
            img = A[:,idx]  # csc_matrix ('data','indices','indptr') here column vector
            weight = (img.data/np.max(img.data)*255).astype(np.uint8)
            overlay = (weight > S[img.indices])  # Overlay only in region where this component has larger weight than previous
            pos = img.indices[overlay]  # Position (i.e. indices) where to overlay
            H[pos] = hue_list[idx]  # Color the most weighted cell
            S[pos] = weight[overlay]
        H = H.reshape(dims, order='F')
        S = S.reshape(dims, order='F')  # 0 saturation -> white
        img_hsv = np.dstack([H,S,background])  # 0 value (lightness) -> black
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    else:
        img_rgb = np.dstack([background]*3)
    return img_rgb

def merge_iteration(A_merge, C_merge):
    '''
    Perform rank 1 nonnegative matrix factorization to merge components.
    
    Parameters
    ----------
    A_merge : scipy.sparse.csc_matrix
        Matrix of spatial components to merge, shape (N,r)
        where N=xy the number of pixels, r>=2 the number of components to merge.
    C_merge : numpy.ndarray, shape (r,T)
        Array of temporal components to merge.

    Returns
    -------
    computedA : scipy.csc_matrix, shape (N,1)
        Merged spatial component
    computedC : numpy.ndarray, shape (T,)
        Merged temporal component
    '''
    ## Start by initializing a merged spatial component (computedA) calculated as
    ## a weighted sum of all spatial components to merge (A_merge)
    ## Weights are choosed to scale with the corresponding temporal component C_merge,
    ## and are normalized to 1 (norm2)
    ## Note that norm2 of computedA is not necessarilly 1 due to overlapped A_merge
    C2 = np.mean(C_merge**2, axis=1)  # Shape (r,) sum of squares
    nC = np.sqrt(C2/C2.sum())  # Shape (r,) weight of temporal component such that nC**2 is summed to 1
    computedA = A_merge.dot(nC)  # Initialized A, shape (N,) Note that matrix vector product dot() returns numpy.ndarray !!
    ## Updata merged temporal component (computedC) and computedA with 10 iterations
    for _ in range(10):
        computedC = (A_merge.T.dot(computedA)).dot(C_merge) / (computedA.dot(computedA))
        computedA = A_merge.dot(C_merge.dot(computedC)) / (computedC.dot(computedC))
        computedA = np.maximum(computedA, 0)  # Spatial component is nonnegative
    
    normA = np.sqrt(computedA.dot(computedA))
    computedA /= normA  # Make norm2 of computedA 1
    computedC *= normA
    return sparse.csr_matrix(computedA).T, computedC

def update_list(K, list_idx, merge_idx):
    '''
    Parameters
    ----------
    K : int
        Number of components before merging operation.
    list_idx : numpy 1D array of int
        List of component indices before merging operation.
    merge_idx : numpy 1D array of int
        List of indices of components to merge.

    Returns
    -------
    list_merged : numpy 1D array of int
        The component indices after merging for the same group of components represented by list_idx.
        The merged component appends to the last one and is in the list if all the parent components (merged)
        were in the list
    '''
    keep = np.ones(K, dtype=bool)  # Components to keep
    keep[merge_idx] = False
    list_tmp = np.setdiff1d(list_idx, merge_idx)
    bool_tmp = np.zeros(K, dtype=bool)
    bool_tmp[list_tmp] = True
    bool_tmp = bool_tmp[keep]
    if all([i in list_idx for i in merge_idx]):
        bool_tmp = np.hstack([bool_tmp, True])
    else:
        bool_tmp = np.hstack([bool_tmp, False])
    list_merged = np.where(bool_tmp)[0]
    return list_merged
    
# %% Execute application event loop
if __name__ == '__main__':
    ## Initializing Qt
    app = QtGui.QApplication(sys.argv)
    ## Instantiate the MainWindow class
    win = MainWindow()
    # win = MainWindow(jsonpath='config_nwb.json')
    # win = MainWindow(datapath=r'D:\Miniscope\2019-11-20-11-55-39_video1_memmap__d1_714_d2_728_d3_1_order_C_frames_6130_.hdf5')
    win.show()
    sys.exit(app.exec_())
