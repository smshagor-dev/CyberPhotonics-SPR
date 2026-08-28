from __future__ import annotations
import argparse
from pathlib import Path
import tensorflow as tf
from sprpcf.edge.quantization import convert_model_to_int8_tflite
from sprpcf.edge.train_denoiser import normalize_spectra,parse_spectra
from sprpcf.ml.dataset import read_table
def convert_to_tflite(model_path,output_path,quantization,calibration_data=None):
    model=tf.keras.models.load_model(model_path)
    if quantization=="int8":
        if calibration_data is None: raise ValueError("Full INT8 export requires --calibration-data.")
        frame=read_table(calibration_data).dropna(subset=["loss_db_per_cm"]); spectra,_,_=normalize_spectra(parse_spectra(frame)); convert_model_to_int8_tflite(model,output_path,spectra); return
    converter=tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization=="fp16": converter.optimizations=[tf.lite.Optimize.DEFAULT]; converter.target_spec.supported_types=[tf.float16]
    output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_bytes(converter.convert())
def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",type=Path,required=True); p.add_argument("--out",type=Path,required=True); p.add_argument("--quantization",choices=["none","fp16","int8"],default="fp16"); p.add_argument("--calibration-data",type=Path); a=p.parse_args(); convert_to_tflite(a.model,a.out,a.quantization,a.calibration_data)
if __name__=="__main__":main()
