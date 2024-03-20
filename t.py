from typing import NamedTuple
import os 
import traceback 

import torch 

from whisperx.asr import load_model 
from whisperx.audio import load_audio 
from whisperx.utils import WriteVTT 

    
DEFAULT_ASR_OPTIONS = {
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1,
    "no_repeat_ngram_size": 0,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "prompt_reset_on_temperature": 0.5,
    "initial_prompt": None,
    "prefix": None,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "without_timestamps": True,
    "max_initial_timestamp": 0.0,
    "word_timestamps": False,
    "prepend_punctuations": "\"'“¿([{-",
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "suppress_numerals": False,
    "max_new_tokens": None,
    "clip_timestamps": None,
    "hallucination_silence_threshold": None,
}

DEFAULT_VAD_OPTIONS = {
    "vad_offset": 0.363,
    "vad_onset": 0.500,
}

DEFAULT_VTT_OPTIONS = {
    "max_line_width": 1000,
    "max_line_count": 100,
    "highlight_words": False,
}

        


def parse_options(options, default: NamedTuple=None):
    for k, v in default.items():
        if not options.get(k):
            options[k] = default[k]
    updated = {k: v for k, v in default.items()}
    updated.update(options)
            
    changed = {}
    change_count = 0
    for (k1, v1), (k2, v2) in zip(default.items(), updated.items()):
        if v1 != v2:
            changed[k2] = v2 
            change_count += 1
    if change_count > 1: 
        print(change_count)
        raise ValueError(f"Changing multiple options is not yet implemented. Only one option should be changed to compare.")
    return updated, changed


def compare(args, chunk_size=30): 
    # 2. load the FW pipeline with the loaded options 
    whisper_arch = args.whisper_arch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    language = args.language 
    asr_options = DEFAULT_ASR_OPTIONS
    vad_options = DEFAULT_VAD_OPTIONS

    vad_options["vad_offset"] = args.vad_offset
    vad_options["vad_onset"] = args.vad_onset
    
    model = load_model(whisper_arch, device=device, compute_type=compute_type, asr_options=asr_options, vad_options=vad_options)
    
    # 3. loop through audio samples 
    write_options = {
        "max_line_width": 1000,
        "max_line_count": 100,
        "highlight_words": False,
    }
    for (root, dirs, files) in os.walk(args.samples):
        if files:
            prefix = os.path.abspath(root)
            for file in files:
                basename, ext = os.path.splitext(file)
                if ext == ".wav" and basename == "music_3":
                    try:
                        audio_path = os.path.join(prefix, file)
                        audio = load_audio(audio_path)
                        result = model.transcribe(audio, language=language, chunk_size=chunk_size, print_progress=args.print_progress)
                        # make output dir based on the audio sample name and current hyperparameter and initiate the writer.
                        output_dir_name = "chunk_size" 
                        output_dir = os.path.join(args.output_dir, basename, output_dir_name) 
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        name = "_"+str(chunk_size)
                        print(os.path.join(output_dir, file))
                        WriteVTT(output_dir=output_dir)(result=result, audio_path=audio_path, options=write_options, name=name)
                    except Exception as e:
                        traceback.print_tb(e.__traceback__)
                        print(e)
                        
                


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    ## model options 
    parser.add_argument("--whisper_arch", default="large-v3", type=str)
    parser.add_argument("--language", default="ko", type=str)
    parser.add_argument("--print_progress", default=True, type=bool)
    ## wav options 
    parser.add_argument("--samples", type=str, help="folder path of sample wavs.")
    
    ## vad options 
    parser.add_argument("--chunk_size", default=False, type=bool)
    parser.add_argument("--vad_offset", default=0.363, type=float)    
    parser.add_argument("--vad_onset", default=0.5, type=float)
    
    ## vtt write options
    parser.add_argument("--output_dir", default="/home/ubuntu/", type=str)
    args = parser.parse_args()
    
    #run 
    if args.chunk_size:
        chunk_sizes = [i for i in range(5, 31, 5)]
        for chunk_size in chunk_sizes:
            compare(args, chunk_size=chunk_size)
