from __future__ import annotations
import torch
import torch.nn.functional as F
from sprpcf.simulation.dispersion import plasmonic_validity_score
PITCH_UM_BOUNDS=(0.8,4.0); D_OVER_LAMBDA_BOUNDS=(0.20,0.90); METAL_NM_BOUNDS=(15.0,80.0); CHANNEL_RADIUS_UM_BOUNDS=(0.10,2.0)
def _bound_penalty(v,lo,hi): return F.relu(lo-v).pow(2).mean()+F.relu(v-hi).pow(2).mean()
def geometry_constraint_loss(geometry,overlap_weight=1.0,boundary_weight=1.0,resonance_wavelength_nm=None,dispersion_weight=0.0):
    if geometry.ndim!=2 or geometry.shape[1]!=4: raise ValueError("geometry must have shape [batch, 4].")
    pitch,ratio,metal,channel=geometry[:,0],geometry[:,1],geometry[:,2],geometry[:,3]
    overlap=F.relu(ratio*pitch-pitch).pow(2).mean(); boundary=_bound_penalty(pitch,*PITCH_UM_BOUNDS)+_bound_penalty(ratio,*D_OVER_LAMBDA_BOUNDS)+_bound_penalty(metal,*METAL_NM_BOUNDS)+_bound_penalty(channel,*CHANNEL_RADIUS_UM_BOUNDS); total=overlap_weight*overlap+boundary_weight*boundary
    if resonance_wavelength_nm is not None and dispersion_weight>0:
        validity=plasmonic_validity_score(resonance_wavelength_nm.reshape(-1).to(dtype=geometry.dtype,device=geometry.device)/1000.0); total=total+dispersion_weight*F.relu(-validity).pow(2).mean()
    return total
