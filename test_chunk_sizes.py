import time 
import logging 
import os

import pytest
import librosa  
import torch 

from whisperx.asr import load_model 
from whisperx.utils import WriteVTT
from whisperx.default_options import DEFAULT_VTT_OPTIONS
from whisperx.hallucinations import hallucination_filters


logger = logging.getLogger(__name__)
@pytest.mark.parametrize("chunk_sizes", [(30, 5), (25, 5), (20, 5), (15, 5)])
def test_chunk_sizes(chunk_sizes: int):
    outer_chunk_size, inner_chunk_size = chunk_sizes
    # 0. init time 
    vad_options = {
        "vad_onset": 0.1, 
        "vad_offset": 0.1
    }
    model = load_model("large-v3", device="cuda", vad_options=vad_options)    
    
    RESULTS_OUTPUT_PATH = "/home/ubuntu/pytest_results_ent"
    if not os.path.exists(os.path.join(RESULTS_OUTPUT_PATH)):
        os.makedirs(RESULTS_OUTPUT_PATH, exist_ok=True)
    
    
    audios = []
    root = "/home/ubuntu/whisper-hyper-wav/entertainment"
    for _root, _dirs, files in os.walk(root):
        if files:
            for file in files:
                if os.path.splitext(file)[-1] == ".wav":
                    audios.append(os.path.join(_root, file))

    for audio in audios:
        f_name = os.path.basename(audio).split(".")[0]
        cur_dir = os.path.join(RESULTS_OUTPUT_PATH, f_name)
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)
        start = time.time()
        y, sr = librosa.load(audio, sr=16000)
        result = model.transcribe(y, language="ko", batch_size=5, outer_chunk_size=int(outer_chunk_size), inner_chunk_size=int(inner_chunk_size))
        cur_filter = hallucination_filters["ko"]
        try:
            segments = []
            for i, seg in enumerate(res['segments']):
                if not cur_filter(seg['text'].strip()):
                    segments.append(seg)
            res['segments'] = segments 
        except Exception as e:
            print(e)
        vtt_writer = WriteVTT(output_dir=cur_dir)
        name = f"_{outer_chunk_size}_{inner_chunk_size}_trial1"
        vtt_writer(result, audio_path=audio, options=DEFAULT_VTT_OPTIONS, name=name)
        end= time.time()
        logger.critical(f"time elapsed: {end - start} | file: {audio}")
    
    
    
