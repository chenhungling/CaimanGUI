# -*- coding: utf-8 -*-
"""
Convert caiman processed data (hdf5 file) to NWB format.

@author: Hung-Ling
"""
import json
import numpy as np
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO  # TimeSeries
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from pynwb.ophys import OpticalChannel, ImageSeries, ImageSegmentation, Fluorescence

def save_nwb(self, fname_nwb, config=None, raw_data_file=None):
    '''
    Save caiman data as NWB file. Only accepted components are saved.

    Parameters
    ----------
    self : CNMF object of caiman.source_extraction.cnmf.cnmf.CNMF
        Caiman processed data.
    fname_nwb : str
        Full path to the NWB file.
        Attention : If the file already exists, it will be overwrite.
    config : dict
        Dictionary containing metadata to be store in the NWB file.
        The default is None (will load a default json file).
    raw_data_file : str
        Full path to the imaging data. If this is provided, a link is created.
        The default is None.
    '''
    if config is None:
        ## Read json file
        with open('config-nwb.json', 'r') as f:
            config = json.load(f)

    if config['nwbfile']['identifier'] is None:
        import uuid
        config['nwbfile']['identifier'] = uuid.uuid1().hex  # Generate random ID
        
    ## Convert session_start_time from str to datetime format
    ts = [int(t) for t in config['nwbfile']['session_start_time'].split('-')]
    config['nwbfile']['session_start_time'] = datetime(*ts, tzinfo=tzlocal())
    
    ## Create an NWB file
    nwbfile = NWBFile(**config['nwbfile'])
    
    ## Add Device, OpticalChannel and ImagingPlane
    device = nwbfile.create_device(**config['device'])
    optical_channel = OpticalChannel(**config['channel'])
    img_plane = nwbfile.create_imaging_plane(
        optical_channel=optical_channel,
        device=device,
        **config['plane'])
    
    ## Optional link to the original recording file
    if raw_data_file is not None:
        extension = raw_data_file.split('.')[-1]
        fps = config['plane']['imaging_rate']
        nwbfile.add_acquisition(
            ImageSeries(name='ImageSeries',
                        external_file=[raw_data_file],
                        format=extension,
                        rate=fps,
                        starting_frame=[0]))
    
    ## Create "ophys" processing module
    mod = nwbfile.create_processing_module('ophys', 'Optical physiology processed data')
    
    ## Add ROIs under ImageSegmentation
    img_seg = ImageSegmentation(name='ImageSegmentation')
    mod.add(img_seg)
    ps = img_seg.create_plane_segmentation(
        name='PlaneSegmentation',
        description='Spatial components (ROIs)',
        imaging_plane=img_plane)
    
    ## Save only accepted components
    if hasattr(self.estimates, 'accepted_list') and len(self.estimates.accepted_list)>0:
        accepted_list = self.estimates.accepted_list
    else:
        accepted_list = self.estimates.idx_components
    dims = self.estimates.dims
    
    for k in accepted_list:
        Ak = self.estimates.A[:,k].reshape(dims, order='F').tocoo()  # coo_matrix
        pixel_mask = np.column_stack([Ak.col, Ak.row, Ak.data])  # (x,y,weight) 
        ps.add_roi(pixel_mask=pixel_mask)
    
    ## Add customized columns to store metrics
    ps.add_column('r_values', 'Spatial correlation score',
                  self.estimates.r_values[accepted_list])
    ps.add_column('snr_values', 'Temporal signal-to-noise ratio',
                  self.estimates.SNR_comp[accepted_list])
    if len(self.estimates.cnn_preds)>0:
        ps.add_column('cnn', 'CNN score',
                      self.estimates.cnn_preds[accepted_list])
    
    rt_region = ps.create_roi_table_region(
        region=list(np.arange(len(accepted_list))),
        description='Cell IDs')
    
    ## Add fluorescence
    fl = Fluorescence(name='Fluorescence')
    mod.add(fl)
    
    fr = self.params.data['fr']
    timestamps = np.arange(self.estimates.C.shape[1])/fr
    
    names = ['Raw', 'Denoised', 'Spikes']
    traces = [
        (self.estimates.C[accepted_list]+self.estimates.YrA[accepted_list]).T,
        self.estimates.C[accepted_list].T,
        self.estimates.S[accepted_list].T
        ]  # Time dimension goes first
    for i in range(3):
        fl.create_roi_response_series(
            name=names[i], 
            data=traces[i],
            rois=rt_region,
            timestamps=timestamps,
            unit='a.u.')

    ## Store summary images
    images = Images(name='Images',
                    description='Summary images')
    images.add_image(GrayscaleImage('corr', self.cn_filter,  # name, data
                                    description='Local correlation map'))
    images.add_image(GrayscaleImage('pnr', self.pnr,
                                    description='Peak-to-noise ratio map'))
    
    for proj in ['max', 'mean', 'std']:
        img_name = 'image_'+proj
        if hasattr(self, img_name):
            images.add_image(GrayscaleImage(img_name, eval('self.'+img_name),
                                            description=img_name+' projection image'))
    mod.add(images)
    
    ## Write the NWB file (overwrite if already exists)
    with NWBHDF5IO(fname_nwb, 'w') as io:
        io.write(nwbfile)