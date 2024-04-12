## Voice Activity Detection only script. 
## to check if the detected timestamp is correct before preceding to the model generation step. 
import time
import re 
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

GHOST_PATTERNS = [
        "이 시각 세계였습니다.",
        "MBC 뉴스 김성현입니다.",
        "지금까지 뉴스 스토리였습니다.",
        "시청해주셔서 감사합니다.",
        "날씨였습니다.",
        "자막 제공 배달의민족",
        "제작지원 자막 제작지원",
        "이 노래는 제가 부르는 노래입니다.",
        "아이유의 러브게임"
]
def is_ghost_pattern(text):
    if text in GHOST_PATTERNS:
        return True
    return False

def run_single(args):
    printer.pprint(args)
    audio = args.pop("audio")
    language = args.pop("language", None)
    vad_offset = args.pop("vad_offset", DEFAULT_VAD_OPTIONS["vad_offset"])
    vad_onset = args.pop("vad_onset", DEFAULT_VAD_OPTIONS["vad_onset"])
    duration_chunk_size = args.pop("duration_chunk_size", 20)
    merge_chunk_size = args.pop("merge_chunk_size", 5)
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
    
    y, sr = librosa.load(audio, sr=16000)
    vad_model = load_vad_model(torch.device("cuda"), use_auth_token=None)
    vad_segments = vad_model({
        "waveform": torch.from_numpy(y).unsqueeze(0),
        "sample_rate": SAMPLE_RATE
    })
    active_segments = merge_chunks(vad_segments, duration_chunk_size=5, merge_chunk_size=5)
    
    before_merge = []
    for merged_segment in active_segments:
        segments = merged_segment["segments"]
        before_merge += segments
    merged_output = {
        "language": language,
        "segments": []
        }
    for segment in before_merge:
        # queue_name 관련 코드.
        start, end = segment  # (0.132, 3.232)
        f1 = int(start * SAMPLE_RATE)
        f2 = int(end * SAMPLE_RATE)
        current_section = y[f1: f2]
        # features = log_mel_spectrogram(
        #     current_section,
        #     n_mels=128,
        #     padding=N_SAMPLES - current_section.shape[0]
        # )
        output = pipeline.transcribe(current_section, language=language, duration_chunk_size=20, merge_chunk_size=20)
        printer.pprint(output)
        try:
            merged_output["segments"].append(
                {
                    "start": start,
                    "end": end,
                    "text": output["segments"][0]['text'].strip()
                }
            )
        except Exception as e:
            print(e)
    res = merged_output
        
    # res = pipeline.transcribe(y, duration_chunk_size=duration_chunk_size, merge_chunk_size=merge_chunk_size, batch_size=batch_size, print_progress=True, language=language)
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
    parser.add_argument("--merge_chunk_size", type=int, default=5)
    parser.add_argument("--duration_chunk_size", type=int, default=15)
    
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    parser.add_argument("--output_dir", type=str, default="/home/ubuntu")
    parser.add_argument("--name", type=str, default=f"{time.strftime('%H%M%S',time.localtime(time.time()))}")

    args = parser.parse_args()
    args = args.__dict__
    start = time.time() 
    run_single(args) 
    end = time.time()
    print(f"time elapsed: {end - start} seconds.")