from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np,tensorflow as tf
from sprpcf.edge.denoising import build_denoising_autoencoder,build_ri_predictor
from sprpcf.edge.quantization import TFLiteModelRunner,convert_model_to_int8_tflite
from sprpcf.ml.dataset import read_table
def parse_spectra(frame):
    rows=[np.fromstring(v,sep=",") for v in frame["loss_db_per_cm"].astype(str)]; lengths={len(x) for x in rows}
    if not rows or len(lengths)!=1 or next(iter(lengths))<8: raise ValueError("All spectra must contain the same number of >=8 samples.")
    return np.asarray(rows,dtype=np.float32)
def normalize_spectra(s): mean=s.mean(1,keepdims=True); std=s.std(1,keepdims=True)+1e-6; return ((s-mean)/std).astype(np.float32),mean.astype(np.float32),std.astype(np.float32)
def add_sensor_noise(clean,noise_std=.08,drift_std=.03,seed=7):
    rng=np.random.default_rng(seed); axis=np.linspace(-1,1,clean.shape[1],dtype=np.float32); return (clean+rng.normal(0,noise_std,clean.shape).astype(np.float32)+rng.normal(0,drift_std,(clean.shape[0],1)).astype(np.float32)*axis+.015*np.sin(np.linspace(0,8*np.pi,clean.shape[1],dtype=np.float32))[None,:]).astype(np.float32)
def train_val_split(*arrays,val_fraction=.2,seed=7):
    n=arrays[0].shape[0]
    if n<5: raise ValueError("At least five spectra are required.")
    idx=np.random.default_rng(seed).permutation(n); nv=max(1,int(n*val_fraction)); return [a[idx[nv:]] for a in arrays],[a[idx[:nv]] for a in arrays]
def mse(a,b):return float(np.mean((a-b)**2))
def mae(a,b):return float(np.mean(np.abs(a-b)))
def psnr(a,b):
    e=mse(a,b)
    if e<=1e-12:return float("inf")
    return float(20*np.log10(max(float(np.max(a)-np.min(a)),1e-6))-10*np.log10(e))
def ssim_1d(a,b):
    x=a.reshape(a.shape[0],-1);y=b.reshape(b.shape[0],-1);c1=.01**2;c2=.03**2;mx=x.mean(1);my=y.mean(1);vx=x.var(1);vy=y.var(1);cov=((x-mx[:,None])*(y-my[:,None])).mean(1);return float(np.mean(((2*mx*my+c1)*(2*cov+c2))/((mx**2+my**2+c1)*(vx+vy+c2))))
def r2_columns_np(a,b):
    residual=np.sum((a-b)**2,axis=0);total=np.sum((a-a.mean(0))**2,axis=0);return 1-residual/np.maximum(total,1e-12)
def weighted_regression_loss(target_scale):
    scale=tf.constant(target_scale.reshape(1,-1),dtype=tf.float32)
    def loss(y_true,y_pred):return tf.reduce_mean(tf.square((y_true-y_pred)/scale))
    return loss
def configure_device(device):
    if device=="cpu": tf.config.set_visible_devices([],"GPU");return "/CPU:0"
    if device=="auto": return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    return device
def _tflite_predict_all(path,inputs):
    runner=TFLiteModelRunner(path);out=[];lat=[]
    for sample in inputs:
        start=time.perf_counter();out.append(runner.predict(sample[None,...])[0]);lat.append((time.perf_counter()-start)*1000)
    return np.asarray(out,np.float32),np.asarray(lat,float)
def train_edge_models(data_path,denoiser_out,predictor_out,epochs,batch_size,device,quantize,denoiser_tflite_out,predictor_tflite_out,seed=7):
    tf.keras.utils.set_random_seed(seed);frame=read_table(data_path).dropna(subset=["loss_db_per_cm","analyte_ri","lambda_res_nm"]);clean,_,_=normalize_spectra(parse_spectra(frame));noisy=add_sensor_noise(clean,seed=seed);targets=frame[["analyte_ri","lambda_res_nm"]].to_numpy(np.float32);(nt,ct,tt),(nv,cv,tv)=train_val_split(noisy,clean,targets,seed=seed)
    with tf.device(configure_device(device)):
        den=build_denoising_autoencoder(clean.shape[1]);den.compile(optimizer="adam",loss="mse");den.fit(nt[...,None],ct[...,None],validation_data=(nv[...,None],cv[...,None]),epochs=epochs,batch_size=batch_size,verbose=0);dt=den.predict(nt[...,None],verbose=0);dv=den.predict(nv[...,None],verbose=0);pred=build_ri_predictor(clean.shape[1]);pred.compile(optimizer="adam",loss=weighted_regression_loss(tt.std(0)+1e-6));pred.fit(dt,tt,validation_data=(dv,tv),epochs=epochs,batch_size=batch_size,verbose=0)
    denoiser_out.parent.mkdir(parents=True,exist_ok=True);predictor_out.parent.mkdir(parents=True,exist_ok=True);den.save(denoiser_out,include_optimizer=False);pred.save(predictor_out,include_optimizer=False);fp=pred.predict(dv,verbose=0);r2=r2_columns_np(tv,fp);metrics={"denoising_mse":mse(cv[...,None],dv),"denoising_psnr":psnr(cv[...,None],dv),"denoising_ssim":ssim_1d(cv[...,None],dv),"ri_mae":mae(tv[:,0],fp[:,0]),"ri_r2":float(r2[0]),"lambda_res_mae_nm":mae(tv[:,1],fp[:,1]),"lambda_res_r2":float(r2[1])}
    if quantize:
        convert_model_to_int8_tflite(den,denoiser_tflite_out,nt);convert_model_to_int8_tflite(pred,predictor_tflite_out,dt[:,:,0]);qden,lat=_tflite_predict_all(denoiser_tflite_out,nv[...,None]);qpred,_=_tflite_predict_all(predictor_tflite_out,qden);qr2=r2_columns_np(tv,qpred);metrics.update({"int8_denoising_psnr":psnr(cv[...,None],qden),"int8_ri_mae":mae(tv[:,0],qpred[:,0]),"int8_ri_r2":float(qr2[0]),"int8_lambda_res_mae_nm":mae(tv[:,1],qpred[:,1]),"int8_lambda_res_r2":float(qr2[1]),"int8_latency_ms_p50":float(np.percentile(lat,50)),"int8_latency_ms_p95":float(np.percentile(lat,95)),"int8_denoiser_bytes":float(denoiser_tflite_out.stat().st_size),"int8_predictor_bytes":float(predictor_tflite_out.stat().st_size)})
    return metrics
def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--epochs",type=int,default=25);p.add_argument("--batch-size",type=int,default=64);p.add_argument("--device",default="auto");p.add_argument("--quantize",action="store_true");p.add_argument("--seed",type=int,default=7);p.add_argument("--out",type=Path,default=Path("models/edge_denoiser.keras"));p.add_argument("--ri-out",type=Path,default=Path("models/edge_ri_predictor.keras"));p.add_argument("--denoiser-tflite-out",type=Path,default=Path("models/edge_denoiser_quantized.tflite"));p.add_argument("--ri-tflite-out",type=Path,default=Path("models/edge_ri_predictor_quantized.tflite"));a=p.parse_args();print(json.dumps(train_edge_models(a.data,a.out,a.ri_out,a.epochs,a.batch_size,a.device,a.quantize,a.denoiser_tflite_out,a.ri_tflite_out,a.seed),indent=2))
if __name__=="__main__":main()
