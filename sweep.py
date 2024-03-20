from itertools import product 
import os 
# import logging 
import pprint 

import numpy as np 

import librosa 
from whisperx import load_model
from whisperx.utils import WriteVTT 
import gc 

VTT_OPTIONS = {
    "max_line_width": 1000,
    "max_line_count": 100,
    "highlight_words": False,
}

printer = pprint.PrettyPrinter()

DEFAULT_ASR_OPTIONS = {
    "beam_size": 5,
    "best_of": 5, # number of candidates when decoding with non zero temperature. 
    "patience": 1, # optional parameter used when decoding. 
    "length_penalty": 1,
    "repetition_penalty": 1,
    "temperatures": 0 # 0 by default. used when sampling while decoding. 
}

def get_asr_options(options):
    if options.temperature != 0:
        options["beam_size"] = 1 
        return options 
    

def sweep(args):
    args = args.__dict__
    audio = args.pop("audio")
    output_dir = args.pop("output_dir")
    print_progress = args.pop("print_progress")
    vad_onset = args.pop("vad_onset")
    vad_offset = args.pop("vad_offset")
    length_penalty = args.pop("length_penalty")
    repetition_penalty = args.pop("repetition_penalty")
    
    do_beam_size = args.pop("beam_size")
    do_temperature = args.pop("temperature")
    if do_beam_size and do_temperature:
        raise ValueError("Choose only one option to sweep. Both the beam size and temperature can not be sweeped at the same time.")

    vad_options = {
        "vad_onset": vad_onset,
        "vad_offset": vad_offset, 
    }
    if do_beam_size:
        beam_sizes = np.arange(5, 21, 5)
        patiences = np.arange(1, 2, 0.2)
        param_combinations = list(product(beam_sizes, patiences))
        y, sr = librosa.load(audio, sr=16000)
        for beam_size, patience in param_combinations:
            asr_options = DEFAULT_ASR_OPTIONS
            asr_options["beam_size"] = beam_size 
            asr_options["patience"] = patience 
            model = load_model("large-v3", device="cuda", compute_type="float16", asr_options=asr_options, vad_options=vad_options)
            result = model.transcribe(y, language="ko", chunk_size=30, print_progress=print_progress)
            printer.pprint(asr_options)
            printer.pprint(vad_options)
            name = f"_{beam_size}_{patience}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            WriteVTT(output_dir=output_dir)(result=result, audio_path=audio, options=VTT_OPTIONS, name=name)
            del model 
            gc.collect() 
            

    if do_temperature:
        temperatures = np.arange(0, 1, 0.2)
        best_ofs = [1, 3, 5, 10]
        param_combinations = list(product(temperatures, best_ofs))
        y, sr = librosa.load(audio, sr=16000)
        for temperature, best_of in param_combinations:
            asr_options = DEFAULT_ASR_OPTIONS
            asr_options["temperatures"] = temperature 
            asr_options["best_of"] = best_of 

            printer.pprint(asr_options)
            printer.pprint(vad_options)
            
            model = load_model("large-v3", device="cuda", compute_type="float16", asr_options=asr_options, vad_options=vad_options)
            result = model.transcribe(y, language="ko", chunk_size=30, print_progress=print_progress)
            name = f"_{temperature}_{best_of}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            WriteVTT(output_dir=output_dir)(result=result, audio_path=audio, options=VTT_OPTIONS, name=name)
            del model 
            gc.collect() 
    

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--print_progress", action='store_true')
    # Common hyperparamters used for both combinations 
    parser.add_argument("--vad_onset", type=float, default=0.5)
    parser.add_argument("--vad_offset", type=float, default=0.363)
    parser.add_argument("--length_penalty", type=float, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1)

    # choose which one to sweep. the range is fixed based on the selected option
    parser.add_argument("--beam_size", action='store_true')
    parser.add_argument("--temperature", action='store_true')
    args = parser.parse_args()
    sweep(args)