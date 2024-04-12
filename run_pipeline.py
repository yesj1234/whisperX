import time 

from whisperx.asr_origin import load_model 
from whisperx.default_options import DEFAULT_VAD_OPTIONS, DEFAULT_ASR_OPTIONS, DEFAULT_VTT_OPTIONS
from whisperx.utils import WriteVTT



def run_pipeline(args):
    audio = args.pop("audio")
    vad_offset = args.pop("vad_offset", DEFAULT_VAD_OPTIONS["vad_offset"])
    vad_onset = args.pop("vad_onset", DEFAULT_VAD_OPTIONS["vad_onset"])
    asr_chunk_size = args.pop("asr_chunk_size", 20)
    no_repeat_ngram_size = args.pop("no_repeat_ngram_size", 1)
    
    language = args.pop("language", "ko")
    
    output_dir = args.pop("output_dir")
    name = args.pop("name")
    
    asr_options = DEFAULT_ASR_OPTIONS
    asr_options.update(
        {
            "no_repeat_ngram_size": no_repeat_ngram_size
        }
    )
    vad_options = DEFAULT_VAD_OPTIONS
    vad_options.update(
        {
            "vad_onset": vad_onset,
            "vad_offset": vad_offset
        }
    )
    pipeline = load_model("large-v3", device="cuda", compute_type="float16", asr_options=asr_options, vad_options=vad_options)
    res = pipeline.transcribe(audio, chunk_size=asr_chunk_size, language=language)
    if output_dir:
        vtt_writer = WriteVTT(output_dir=f"{output_dir}")
        vtt_writer(res, audio_path=audio, options=DEFAULT_VTT_OPTIONS, name=name)
    return res 

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
     
    parser.add_argument("--audio", required=True)
    parser.add_argument("--asr_chunk_size", type=int, default=20)
    parser.add_argument("--vad_onset", type=float, default=0.5)
    parser.add_argument("--vad_offset", type=float, default=0.363)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--language", type=str, default='ko')
    parser.add_argument("--output_dir", type=str, default='/home/ubuntu/')
    parser.add_argument("--name", type=str, default=f"{time.strftime('%H%M%S',time.localtime(time.time()))}")
    
    args = parser.parse_args()
    args = args.__dict__
    run_pipeline(args)
    