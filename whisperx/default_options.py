
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

DEFAULT_VTT_OPTIONS = {
    "max_line_width": 1000,
    "max_line_count": 100,
    "highlight_words": False, 
}