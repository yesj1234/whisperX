## Voice Activity Detection only script. 
## to check if the detected timestamp is correct before preceding to the model generation step. 
import time

import librosa
import torch
import pprint 

import faster_whisper
from whisperx.audio import log_mel_spectrogram
from whisperx.audio import SAMPLE_RATE, N_SAMPLES
from whisperx.asr import WhisperModel, load_model
from whisperx.vad import Binarize, load_vad_model, merge_chunks
from whisperx.utils import WriteVTT
from whisperx.default_options import DEFAULT_ASR_OPTIONS, DEFAULT_VAD_OPTIONS, DEFAULT_VTT_OPTIONS
from whisperx.hallucinations import hallucination_filters

printer = pprint.PrettyPrinter(sort_dicts=False)

def run_single(args):
    printer.pprint(args)
    audio = args.pop("audio")
    language = args.pop("language", None)
    vad_offset = args.pop("vad_offset", DEFAULT_VAD_OPTIONS["vad_offset"])
    vad_onset = args.pop("vad_onset", DEFAULT_VAD_OPTIONS["vad_onset"])
    outer_chunk_size = args.pop("outer_chunk_size", 5)
    inner_chunk_size = args.pop("inner_chunk_size", 15)
    batch_size = args.pop("batch_size", 5)
    no_repeat_ngram_size = args.pop("no_repeat_ngram_size", 1)
    name = args.pop("name", "vad_test")
    output_dir= args.pop("output_dir", None)
    
    
    vad_options = DEFAULT_VAD_OPTIONS
    vad_options.update(
        {"vad_onset": vad_onset,
         "vad_offset": vad_offset})
    
    # 1. load the audio file and preprocess with log_mel_spectrogram 
    y, sr = librosa.load(audio, sr=SAMPLE_RATE)

        
    asr_options = {
        "no_repeat_ngram_size": no_repeat_ngram_size
    }
    
    pipeline = load_model(
        "large-v3", 
        device="cuda", 
        compute_type="float16", 
        language=language,
        asr_options=asr_options,
        vad_options=vad_options,
        task="transcribe")
    
    res = pipeline.transcribe(y, outer_chunk_size=outer_chunk_size, inner_chunk_size=inner_chunk_size, batch_size=batch_size, print_progress=True, language=language)
    # printer.pprint(res)
    cur_filter = hallucination_filters[language]
    try:
        segments = []
        for i, seg in enumerate(res['segments']):
            if not cur_filter(seg['text'].strip()):
                segments.append(seg)
        res['segments'] = segments 
    except Exception as e:
        print(e)
    printer.pprint(res)

    if output_dir:
        vtt_writer = WriteVTT(output_dir=f"{output_dir}")
        vtt_writer(res, audio_path=audio, options=DEFAULT_VTT_OPTIONS, name=name)
    return res
        
if __name__ == "__main__":
    import argparse
    import time 
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--language", required=False)
    parser.add_argument("--vad_onset", type=float, default=0.5)
    parser.add_argument("--vad_offset", type=float, default=0.363)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--outer_chunk_size", type=int, default=5)
    parser.add_argument("--inner_chunk_size", type=int, default=15)
    
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    parser.add_argument("--output_dir", type=str, default="/home/ubuntu")
    parser.add_argument("--name", type=str, default=f"{time.strftime('%H%M%S',time.localtime(time.time()))}")

    args = parser.parse_args()
    args = args.__dict__
    start = time.time() 
    run_single(args) 
    end = time.time()
    print(f"time elapsed: {end - start} seconds.")