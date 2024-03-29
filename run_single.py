## Voice Activity Detection only script. 
## to check if the detected timestamp is correct before preceding to the model generation step. 

import librosa
import torch
import pprint 

import faster_whisper
from whisperx.audio import log_mel_spectrogram
from whisperx.audio import SAMPLE_RATE, N_SAMPLES
from whisperx.asr import WhisperModel, load_model
from whisperx.vad import Binarize, load_vad_model, merge_chunks
from whisperx.utils import WriteVTT

printer = pprint.PrettyPrinter(sort_dicts=False)

DEFAULT_VAD_OPTIONS = {
    "vad_onset": 0.5,
    "vad_offset": 0.363
}   
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
    "max_new_tokens": None,
    "clip_timestamps": None,
    "hallucination_silence_threshold": None,
}

def vad_only(args):
    audio = args.pop("audio")
    vad_offset = args.pop("vad_offset", DEFAULT_VAD_OPTIONS["vad_offset"])
    vad_onset = args.pop("vad_onset", DEFAULT_VAD_OPTIONS["vad_onset"])
    vad_chunk_size = args.pop("vad_chunk_size", 5)
    asr_chunk_size = args.pop("asr_chunk_size", 20)
    no_repeat_ngram_size = args.pop("no_repeat_ngram_size", 1)
    name = args.pop("name", "vad_test")
    output_dir= args.pop("output_dir", None)
    
    
    vad_options = DEFAULT_VAD_OPTIONS
    vad_options.update(
        {"vad_onset": vad_onset,
         "vad_offset": vad_offset})
    
    # 1. load the audio file and preprocess with log_mel_spectrogram 
    y, sr = librosa.load(audio, sr=SAMPLE_RATE)

    
    vad_model = load_vad_model(torch.device("cpu"), use_auth_token=None, **vad_options)
    vad_segments = vad_model({
        "waveform": torch.from_numpy(y).unsqueeze(0),
        "sample_rate": SAMPLE_RATE
    })
    active_segments = merge_chunks(vad_segments, chunk_size=vad_chunk_size, onset=vad_options['vad_onset'], offset=vad_options["vad_offset"])
    # printer.pprint(active_segments)
    # [(0.1, 2.1), ()]
    #1.1 for segments only, try to transcribe the given audio from the source. 
    before_merge = []
    for merged_segment in active_segments:
        segments = merged_segment["segments"]
        before_merge += segments 

    asr_options = {
        "no_repeat_ngram_size": no_repeat_ngram_size
    }
    
    model = load_model(
        "large-v3", 
        device="cuda", 
        compute_type="float16", 
        language="ko",
        asr_options=asr_options,
        vad_options=vad_options,
        task="transcribe")
    
    merged_output = {
        "language": "ko",
        "segments": []}
    
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

    for segment in before_merge:
        start, end = segment # (0.132, 3.232)
        f1 = int(start * SAMPLE_RATE)
        f2 = int(end * SAMPLE_RATE)
        current_section = y[f1: f2]
        features = log_mel_spectrogram(
            current_section,
            n_mels = 128,
            padding = N_SAMPLES - current_section.shape[0]
        )
        output = model.transcribe(current_section, chunk_size=asr_chunk_size, language='ko')
        try:
            cur_text = output["segments"][0]['text'].strip()
            merged_output["segments"].append(
                {
                    "start": start,
                    "end": end,
                    "text": cur_text
                }
            )
        except Exception as e:
            print(e)
    # printer.pprint(merged_output)
    if output_dir:
        vtt_writer = WriteVTT(output_dir=f"{output_dir}")
        vtt_options = {
                "max_line_width": 1000,
                "max_line_count": 100,
                "highlight_words": False, 
            }
        vtt_writer(merged_output, audio_path=audio, options=vtt_options, name=name)
    return merged_output
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--vad_onset", type=float, default=0.5)
    parser.add_argument("--vad_offset", type=float, default=0.363)
    parser.add_argument("--vad_chunk_size", type=int, default=5)
    parser.add_argument("--asr_chunk_size", type=int, default=20)
    
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--name", type=str, default="vad_test") # default="/home/ubuntu/vad_test"
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    args = args.__dict__
    vad_only(args) 