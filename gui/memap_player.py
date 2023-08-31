# -*- coding: utf-8 -*-
"""
Window displaying motion-corrected movie (F-order memap file).
When called from the parent GUI, time series (motion shift and fluorescence of the selected component)
can be shown jointly with the vertical line indicating the current frame.

@author: Hung-Ling
"""
import os
import sys
import numpy as np
import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

pg.setConfigOptions(imageAxisOrder='row-major')

# %%
class MemapPlayer(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(MemapPlayer, self).__init__(parent)
        self.resize(800,800)  # width, height
        self.setWindowTitle('View registered frames')
        self.layout = QtGui.QGridLayout()
        cw = QtGui.QWidget()
        cw.setLayout(self.layout)
        self.setCentralWidget(cw)
        ## -------- Some internal variables/parameters -----------------------
        self.loaded = False
        self.cframe = 0  # Current frame index
        self.fps = 20.0  # Frame per second
        self.dt = 1  # Temporal binning
        self.sigma = 0  # Highpass Gaussian filter size (0 for original frames)
        self.medSize = 1  # Apply median filter
        self.prct = [1, 99.9]  # Percentile for stretching the contrast
        self.dframe = 10  # Number of frames to jump using left/right keys
        self.this_cell = -1
        ## -------- Button to open memap file --------------------------------
        openButton = QtGui.QPushButton('OPEN')
        self.fileLabel = QtGui.QLabel('No file chosen')
        self.layout.addWidget(openButton,0,0,1,2)  # i-row, j-col, nrow, ncol
        self.layout.addWidget(self.fileLabel,0,2,1,14)
        openButton.clicked.connect(self.open_memap)
        ## -------- Parameters -----------------------------------------------
        rateLabel = QtGui.QLabel('Frame rate:')
        self.layout.addWidget(rateLabel,1,2,1,1)
        self.rate = QtGui.QLineEdit()
        self.rate.setFixedWidth(50)
        self.rate.setAlignment(QtCore.Qt.AlignCenter)  # AlignRight
        self.rate.setText(str(self.fps))
        self.rate.textChanged.connect(self.change_params)
        self.layout.addWidget(self.rate,1,3,1,1)
        self.layout.addItem(QtGui.QSpacerItem(10,20),1,4,1,1)  # Horizontal spacer
        binLabel = QtGui.QLabel('Frame binning:')
        self.layout.addWidget(binLabel,1,5,1,1)
        self.binning = QtGui.QLineEdit()
        self.binning.setValidator(QtGui.QIntValidator(1,100))
        self.binning.setFixedWidth(30)
        self.binning.setAlignment(QtCore.Qt.AlignCenter)  # AlignRight
        self.binning.setText(str(self.dt))
        self.binning.textChanged.connect(self.change_params)
        self.layout.addWidget(self.binning,1,6,1,1)
        self.layout.addItem(QtGui.QSpacerItem(10,20),1,7,1,1)  # Horizontal spacer
        sigmaLabel = QtGui.QLabel('Highpass gSig:')  # QtGui.QCheckBox('Highpass')
        self.layout.addWidget(sigmaLabel,1,8,1,1)
        self.highpass = QtGui.QLineEdit()
        self.highpass.setValidator(QtGui.QIntValidator(0,50))
        self.highpass.setFixedWidth(30)
        self.highpass.setAlignment(QtCore.Qt.AlignCenter)  # AlignRight
        self.highpass.setText(str(self.sigma))
        self.highpass.textChanged.connect(self.change_params)
        self.layout.addWidget(self.highpass,1,9,1,1)
        self.layout.addItem(QtGui.QSpacerItem(10,20),1,10,1,1)  # Horizontal spacer
        medianLabel = QtGui.QLabel('Median filt:')
        self.layout.addWidget(medianLabel,1,11,1,1)
        self.medfilt = QtGui.QLineEdit()
        self.medfilt.setValidator(QtGui.QIntValidator(0,50))
        self.medfilt.setFixedWidth(30)
        self.medfilt.setAlignment(QtCore.Qt.AlignCenter)  # AlignRight
        self.medfilt.setText(str(self.medSize))
        self.medfilt.textChanged.connect(self.change_params)
        self.layout.addWidget(self.medfilt,1,12,1,1)
        self.layout.addItem(QtGui.QSpacerItem(10,20),1,13,1,1)  # Horizontal spacer
        cellLabel = QtGui.QLabel('Component:')
        self.layout.addWidget(cellLabel,1,14,1,1)
        self.component = QtGui.QLineEdit()
        self.component.setValidator(QtGui.QIntValidator(-1,1000))
        self.component.setFixedWidth(40)
        self.component.setAlignment(QtCore.Qt.AlignCenter)  # AlignRight
        self.component.setText(str(self.this_cell))
        self.component.textChanged.connect(self.change_params)
        self.layout.addWidget(self.component,1,15,1,1)
        ## -------- Graphics displaying frames and shifts --------------------
        graph = pg.GraphicsLayoutWidget()
        self.p1 = graph.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        self.img = pg.ImageItem()
        self.contour = pg.IsocurveItem(level=32, pen='m')  # Level at which the isocurve is drawn. Note that img_components are normalized 0-255 
        self.contour.setParentItem(self.img)  # Make sure isocurve is always correctly displayed over image
        self.contour.setZValue(10)
        self.p1.addItem(self.img)
        self.p2 = graph.addPlot(row=1, col=0)
        self.p2.addLegend()
        self.p2.setMouseEnabled(x=True, y=False)
        self.vline = pg.InfiniteLine(angle=90, movable=True)
        self.vline.setValue(0)
        self.vline.sigPositionChanged.connect(self.go_to_frame)
        self.p2.addItem(self.vline, ignoreBounds=True)
        graph.ci.layout.setRowStretchFactor(0,2)  # Stretch row 0 by the factor 2
        self.layout.addWidget(graph,2,0,1,16)
        ## -------- Get shift from parent cnmf object ------------------------
        if hasattr(parent, 'cnmf'):  # Call from caiman_gui1p_lite
            self.img_components = parent.img_components
            self.C = parent.cnmf.estimates.C
            if hasattr(parent.cnmf, 'shifts_rig'):
                self.shifts = parent.cnmf.shifts_rig  # (y,x) shifts, shape (T,2)
            self.plot_trace()
            self.plot_contour()
            
        ## -------- Button play/pauss and slider -----------------------------
        iconSize = QtCore.QSize(24,24)
        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip('Play')
        self.playButton.setCheckable(True)
        self.playButton.setEnabled(False)
        self.pauseButton = QtGui.QToolButton()
        self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip('Pause')
        self.pauseButton.setCheckable(True)
        self.pauseButton.setEnabled(False)
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setTracking(False)
        self.layout.addWidget(self.playButton,3,0,1,1)
        self.layout.addWidget(self.pauseButton,3,1,1,1)
        self.layout.addWidget(self.frameSlider,3,2,1,14)
        frameTitle = QtGui.QLabel('Elapsed time:')
        self.elapsedTime = QtGui.QLabel('0:00.0')
        self.layout.addWidget(frameTitle,4,0,1,2)
        self.layout.addWidget(self.elapsedTime,4,2,1,1)        
        self.playButton.clicked.connect(self.play)
        self.pauseButton.clicked.connect(self.pause)
        self.frameSlider.valueChanged.connect(self.change_slider)
        ## -------- Setup timer ----------------------------------------------
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
    
    # %%
    def open_memap(self):
        fmemap = QFileDialog.getOpenFileName(
            caption='Load memory-mapped file', filter='MMAP (*.mmap)')[0]
        try:
            Yr, dims, T = load_memmap(fmemap)  # Shape (N,T)
            self.movie = Yr.T.reshape((T,)+dims, order='F')  # Shape (T,y,x)
            self.nframe = T
            self.fileLabel.setText('Loaded: '+fmemap)
            self.loaded = True
        except Exception:
            self.fileLabel.setText('Loading '+fmemap+' failed. Try other file...')
        
        if self.loaded:
            self.playButton.setEnabled(True)
            self.frameSlider.setMinimum(0)
            self.frameSlider.setMaximum(self.nframe-1)
            
    # %%
    def change_params(self):
        if self.rate.text():  # Avoid empty string during editing
            self.fps = float(self.rate.text())
        if self.binning.text():
            self.dt = int(self.binning.text())
        if self.highpass.text():
            self.sigma = int(self.highpass.text())
        if self.medfilt.text():
            self.medSize = int(self.medfilt.text())
        if self.component.text():
            self.this_cell = int(self.component.text())
            if hasattr(self, 'img_components'):
                self.plot_contour()
            if hasattr(self, 'C'):
                self.plot_trace()
    
    def change_slider(self):
        if self.vline.value() != int(self.frameSlider.value()):  # Block signal
            self.vline.setValue(int(self.frameSlider.value()))  # This will trigger self.go_to_frame()
        
    # %%    
    def play(self):
        if self.cframe < self.nframe-1:
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.timer.start(0)
    
    def pause(self):
        self.timer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.frameSlider.setEnabled(True)
    
    def next_frame(self):
        '''Read and display the next frame and adjust the current time window in the plot
        '''
        self.cframe += self.dt
        if self.cframe < self.nframe:            
            frame = self.movie[slice(self.cframe-self.dt, self.cframe),:,:].mean(axis=0).astype(np.float32)
            if self.sigma > 0:
                frame -= cv2.GaussianBlur(frame, (4*self.sigma+1,)*2,
                                          sigmaX=self.sigma, sigmaY=self.sigma)
            if self.medSize > 1:
                frame = cv2.medianBlur(frame, self.medSize)
            min_ = np.percentile(frame, self.prct[0]) if self.prct[0]>0 else np.min(frame)
            max_ = np.percentile(frame, self.prct[1]) if self.prct[1]<100 else np.max(frame)
            frame = np.clip(255*(frame-min_)/(max_-min_),0,255).astype(np.uint8)  # Fixed grayscale range    
            self.img.setImage(frame)
            self.vline.setValue(self.cframe)
            self.frameSlider.setValue(self.cframe)
            sec = self.cframe/self.fps  # Elapsed time in second
            decisec = round((sec - np.floor(sec))*10)  # Decisecond
            self.elapsedTime.setText(f'{int(sec)//60}:{int(sec)%60}.{decisec}')
        else:
            self.timer.stop()
            
    def go_to_frame(self):
        '''Jump to the frame specified by the vertical line (during pause only)
        '''
        ## Do nothing during play (playButton is not enabled)
        if self.playButton.isEnabled():
            self.cframe = int(self.vline.value())  # int(self.frameSlider.value())
            self.frameSlider.setValue(self.cframe)
            frame = self.movie[self.cframe,:,:].astype(np.float32)
            if self.sigma > 0:
                frame -= cv2.GaussianBlur(frame, (4*self.sigma+1,)*2,
                                          sigmaX=self.sigma, sigmaY=self.sigma)
            if self.medSize > 1:
                frame = cv2.medianBlur(frame, self.medSize)
            min_ = np.percentile(frame, self.prct[0]) if self.prct[0]>0 else np.min(frame)
            max_ = np.percentile(frame, self.prct[1]) if self.prct[1]<100 else np.max(frame)
            frame = np.clip(255*(frame-min_)/(max_-min_),0,255).astype(np.uint8)  # Fixed grayscale range
            self.img.setImage(frame)
            sec = self.cframe/self.fps  # Elapsed time in second
            decisec = round((sec - np.floor(sec))*10)  # Decisecond
            self.elapsedTime.setText(f'{int(sec)//60}:{int(sec)%60}.{decisec}')
    
    def keyPressEvent(self, event):
        '''Override the existing method to activate left/right key to scroll through the frameSlider
        '''
        if self.playButton.isEnabled():  # During pause only
            # if event.modifiers() !=  QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Left:
                self.cframe = np.clip(self.cframe-self.dframe,0,self.nframe-1)
                self.frameSlider.setValue(self.cframe)
            elif event.key() == QtCore.Qt.Key_Right:
                self.cframe = np.clip(self.cframe+self.dframe,0,self.nframe-1)
                self.frameSlider.setValue(self.cframe)
    
    # %%
    def plot_trace(self):
        self.p2.clearPlots()
        if hasattr(self, 'shifts'):
            self.p2.plot(self.shifts[:,0], pen=(0,128,255), name='y shift')
            self.p2.plot(self.shifts[:,1], pen=(102,204,0), name='x shift')
        if self.this_cell >= 0:
            fluor = self.C[self.this_cell]/np.max(self.C[self.this_cell])*10-10  # Downshifting to separate from motion traces
            self.p2.plot(fluor, pen=(255,51,153), name='fluor')
        
    def plot_contour(self):
        if self.this_cell >= 0:
            self.contour.setData(self.img_components[self.this_cell].astype(np.float32))
    
# %%
def load_memmap(fname, dtype='float32', key=None):
    ''' 
    Load a memory mapped file with customized data type.
    
    Parameters
    ----------
    fname: str
        Full path of the file to be loaded
    dtype: str
        'float32' (default caiman) or 'uint16'
    key: None or array-like (slice, range, ...)
        Used to read only a subset of frame indices
        
    Returns:
    -------
    Yr:
        Memory mapped variable, shape (N,T)
    dims: tuple
        Frame dimensions
    T: int
        Number of frames
    '''
    filename = os.path.split(fname)[-1]
    fpart = filename.split('_')[1:-1]  # The filename encodes the structure of the map
    d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]
    dims = (d1,d2) if d3==1 else (d1,d2,d3)
    shape_mov = (np.uint64(np.prod(dims)), np.uint64(T))
    byte = int(os.path.getsize(fname)/np.prod(dims)/T)
    if byte == 2 and dtype == 'float32':
        dtype = 'uint16'
    elif byte == 4 and dtype == 'uint16':
        dtype = 'float32'
    if key is None:
        Yr = np.memmap(fname, mode='r', shape=shape_mov, order=order, dtype=dtype)
    else:
        Yr = np.memmap(fname, mode='r', shape=shape_mov, order=order, dtype=dtype)[:,key]
        T = Yr.shape[1]
    return Yr, dims, T

# %% Call memap_window(self) from the parent GUI
def memap_window(parent):
    win = MemapPlayer(parent)
    win.show()
    
# %% Execute MemapPlayer() as a standalone GUI
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    win = MemapPlayer()
    win.show()
    sys.exit(app.exec_())
    