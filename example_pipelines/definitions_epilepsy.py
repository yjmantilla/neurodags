import numpy as np
import os
import glob
import scipy
import scipy.io as sio
import mat73
import mne
from cocofeats.loaders import load_meeg
from cocofeats.nodes import register_node,register_node_with_name
from cocofeats.definitions import Artifact, NodeResult
import xarray as xr

def parse_bids(bidsname):
    name = os.path.basename(bidsname)
    entities=name.split('_')
    suffix = entities[-1]
    ext = suffix.split('.')[-1]
    suffix = suffix.split('.')[0]
    entities = entities[:-1]
    d={}
    for item in entities:
        l=item.split('-')
        key=l[0]
        val=l[1]
        d[key]=val
    if not '-' in suffix:
        d['suffix']=suffix
    else:
        a,b=suffix.split('-')
        d[a]=b
    return d


def get_suffix_from_path(x,suffixext):
    subrundir = os.path.dirname(x)
    files = glob.glob(os.path.join(subrundir,f'*_{suffixext}'))
    files = [x for x in files if os.path.isfile(x)]
    query = parse_bids(x)
    for f in files:
        candidate = parse_bids(f)
        winner = True
        if 'suffix' in candidate:
            del candidate['suffix']
        
        for key,val in candidate.items():
            if key in query:
                if query[key]!=candidate[key]:
                    winner = False
                if not winner:
                    break
        if winner:
            return f
    return None


def loadmat(x,kwargs={}):
    try:
        return sio.loadmat(x,**kwargs)
    except:
        return mat73.loadmat(x,**kwargs)

def get_subdict_from_path(x):
    bidspath = x[:x.find('sub-')]
    acq = parse_bids(x)['acq']
    subinfo = loadmat(os.path.join(bidspath,'subinfo.mat',),dict(simplify_cells=True))['SubInfo'][acq]
    subs = [d['Name'] for d in subinfo]
    sub_idx = subs.index(parse_bids(x)['sub'])
    subdict = subinfo[sub_idx]
    return subdict

def get_rundict_from_path(x):
    subdict = get_subdict_from_path(x)
    run = parse_bids(x)['run']
    if isinstance(subdict['SZ'],list):
        subruns =[sz['Name'].replace('_','') for sz in subdict['SZ']]
    else:
        if isinstance(subdict['SZ'],dict):
            subruns = [subdict['SZ']['Name'].replace('_','')]
            subdict['SZ'] = [subdict['SZ']]
        else:
            raise Exception
    idx_run = subruns.index(run)
    rundict = subdict['SZ'][idx_run]
    assert rundict['Name'].replace('_','') == run
    return rundict


def loader_iEEG(x,sfreq=None):
    if '.mat' in x:
        data = loadmat(x)['F']
        subdict = get_subdict_from_path(x)
        rundict = get_rundict_from_path(x)
        if sfreq is None:
            sfreq1 = subdict['sfreq_orig']
            sfreq = loadmat(x)['Time']
            Ts = sfreq[1]-sfreq[0]
            sfreq = 1/Ts
        ch_types = 'eeg'
        ch_names=rundict['Channel']['SEEG']['Name'].tolist()
        assert data.shape[0] == len(ch_names)
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
        raw = mne.io.RawArray(data, info,verbose=False)
    elif '.fif' in x:
        raw = load_meeg(x)
    else:
        raise Exception('File format not supported')
    
    return raw


def load_data(eegpath):
    if '.mat' in eegpath:
        bads = get_suffix_from_path(eegpath,'flag.mat')
        bads = [bool(aux) for aux in np.squeeze(loadmat(bads)['BadChannel']).tolist()]
        labelresect = get_suffix_from_path(eegpath,'labelresect.mat')
        labelresect = [bool(aux) for aux in np.squeeze(loadmat(labelresect)['LabelResect'].tolist()).tolist()]
    else:
        bads = []
        labelresect = []
    raw = loader_iEEG(eegpath)

    if bads:
        # make a DataArray
        bads_xr = xr.DataArray(bads,coords=[raw.ch_names],dims=['spaces'])
    if labelresect:
        labelresect_xr = xr.DataArray(labelresect,coords=[raw.ch_names],dims=['spaces'])
    
    artifacts = {
        '.fif': Artifact(item=raw, writer=lambda path: raw.save(path,overwrite=True))
    }
    
    if labelresect:
        artifacts['.labelresect.nc'] = Artifact(item=labelresect_xr, writer=lambda path: labelresect_xr.to_netcdf(path))
    if bads:
        artifacts['.bads.nc'] = Artifact(item=bads_xr, writer=lambda path: bads_xr.to_netcdf(path))
    
    return NodeResult(artifacts=artifacts)

register_node_with_name('mat2fif', load_data)


def amplitude_normalization_per_segment(mne_object) -> NodeResult:

    if isinstance(mne_object, NodeResult):
        if ".fif" in mne_object.artifacts:
            mne_object = mne_object.artifacts[".fif"].item
        else:
            raise ValueError("NodeResult does not contain a .fif artifact to process.")

    if isinstance(mne_object, str | os.PathLike):
        mne_object = load_meeg(mne_object)

    epo = mne_object.copy()
    #apply the appropiate transformation so that each epoch/channel or channel slice has zero mean and unit variance
    # epochs --> epoch channel slice
    # raw --> channel slice
    data = epo.get_data()
    mean = np.mean(data,axis=-1,keepdims=True)
    std = np.std(data,axis=-1,keepdims=True)
    data = (data - mean)/std
    epo._data = data

    artifacts = {
        ".fif": Artifact(item=epo, writer=lambda path: epo.save(path, overwrite=True))
    }
    out = NodeResult(artifacts=artifacts)
    return out

register_node_with_name('amplitude_normalization_per_segment', amplitude_normalization_per_segment)

def metadata_properties(path_like, str | os.PathLike) -> NodeResult:
    """Extract metadata properties from a file path.

    Parameters
    ----------
    path_like : str | os.PathLike
        File path to extract metadata from.
    Returns
    -------
    NodeResult
        A feature result containing the metadata properties as a NetCDF artifact.
    """
    path_like = str(path_like)
    props = parse_bids(path_like)
    props_xr = xr.Dataset()
    for key, val in props.items():
        props_xr[key] = xr.DataArray([val], dims=["entry"])
    
    artifacts = {
        ".nc": Artifact(item=props_xr, writer=lambda path: props_xr.to_netcdf(path))
    }
    return NodeResult(artifacts=artifacts)