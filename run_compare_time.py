import time 
import pprint

from .run_single import run_single 
from .run_pipeline import run_pipeline 

def run_compare_time(args): 
    # for a given audio input, run as single 
    # and run as pipeline 
    # and compare the whole runtime.
    printer = pprint.PrettyPrinter(sort_dicts=False)
    logs = {
        "run_single": 0,
        "run_pipeline": 0
    } 
    run_single_start = time.time()
    run_single_output = run_single(args) 
    run_single_end = time.time()
    printer.pprint(run_single_output)
    logs["run_single"] = run_single_end - run_single_start
    
    run_pipeline_start = time.time()
    run_pipeline_output = run_pipeline(args) 
    run_pipeline_end = time.time()
    printer.pprint(run_pipeline_output) 
    logs["run_pipeline"] = run_pipeline_end - run_pipeline_start
    
    printer.pprint(logs)


if __name__ == "__main__":
    import argparse 
    args = argparse.ArgumentParser()
    # common args 
    parser.add_argument("--audio", required=True)
    parser.add_argument("--asr_chunk_size", type=int, default=20)
    parser.add_argument("--vad_onset", type=float, default=0.5)
    parser.add_argument("--vad_offset", type=float, default=0.363)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    
    # run_single args 
    parser.add_argument("--vad_chunk_size", type=int, default=5)    
    parser.add_argument("--name", type=str, default="vad_test")
    parser.add_argument("--output_dir", type=str)

    # run_pipeline args 
    
    args = parser.parse_args()
    args = args.__dict__
    
    run_compare_time(args)
    