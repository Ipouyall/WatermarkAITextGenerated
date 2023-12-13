import json
from tqdm import tqdm
import torch
import gc
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LogitsProcessorList

from watermarker.processor import Processor
from watermarker.detector import Detector


def read_json_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]


def write_file_append(filename, data):
    with open(filename, "a") as f:
        f.write("\n".join(data) + "\n")


def write_json_file(filename, data):
    with open(filename, "w") as f:
        json.dump(data, f)


def run_detector(config):
    data = read_json_file(config.input_file)

    if 'llama' in config.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, torch_dtype=torch.float16)

    vocab_size = 50272 if "opt" in config.model_name else tokenizer.vocab_size

    detector = Detector(
        fraction=config.fraction,
        strength=config.strength,
        gamma=config.gamma,
        vocab_size=vocab_size,
        watermark_key=config.hash_key
    )

    z_score_list = []
    iter = tqdm(enumerate(data), total=len(data), leave=False)
    for idx, cur_data in iter:
        gen_tokens = tokenizer(cur_data['gen_completion'][0], add_special_tokens=False)["input_ids"]
        if len(gen_tokens) >= config.min_sequence_tokens:
            z_score_list.append(detector.detect(gen_tokens))
        else:
            print(f"Error: sequence {idx} is too short to test!")
            print("        skipping the sequence")

    save_dict = {
        'z_score': z_score_list,
        'wm_pred': [1 if z > config.watermark_threshold else 0 for z in z_score_list]
    }

    write_json_file(config.input_file.replace('.jsonl', '_z.jsonl'), save_dict)

    print('Finished!')

    print("Detector's report:")
    print(f"\tWatermark threshold:       {config.watermark_threshold}")
    print(f"\tWatermarked sequences:     {sum(save_dict['wm_pred'])} ({sum(save_dict['wm_pred']) / len(save_dict['wm_pred']) * 100:.3f}%)")
    print(f"\tNon-watermarked sequences: {len(save_dict['wm_pred']) - sum(save_dict['wm_pred'])} ({(len(save_dict['wm_pred']) - sum(save_dict['wm_pred'])) / len(save_dict['wm_pred']) * 100:.3f}%)")
    print(f"\tTotal sequences:           {len(save_dict['wm_pred'])}")



@torch.no_grad()
def run_generator(config):
    if 'llama' in config.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(config.model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map='auto')
    model.eval()

    watermark_processor = LogitsProcessorList([
        Processor(
            fraction=config.fraction,
            strength=config.strength,
            gamma=config.gamma,
            vocab_size=tokenizer.vocab_size,
            watermark_key=config.hash_key
        )
    ])

    data = read_json_file(config.prompt_file)
    num_cur_outputs = len(read_json_file(config.output_file)) if os.path.exists(config.output_file) else 0

    outputs = []

    base_generator_config = {
        'output_scores': True,
        'return_dict_in_generate': True,
        'max_new_tokens': config.max_new_tokens,
    }
    if config.apply_watermarking:
        base_generator_config['logits_processor'] = watermark_processor

    processed = 0

    iter = tqdm(enumerate(data), total=min(len(data), config.number_of_tests), leave=False)
    for idx, cur_data in iter:
        if processed >= config.number_of_tests:
            break
        if idx < num_cur_outputs:
            continue
        processed += 1

        if "gold_completion" in cur_data:
            gold_completion = cur_data['gold_completion']
        elif 'targets' in cur_data:
            gold_completion = cur_data['targets'][0]
        else:
            continue
        prefix = cur_data['prefix']

        batch = tokenizer(prefix, truncation=True, return_tensors="pt").to(model.device)
        num_tokens = len(batch['input_ids'][0])

        with torch.inference_mode():
            generate_args = {
                **batch,
                **base_generator_config,
            }

            if config.beam_size is not None:
                generate_args['num_beams'] = config.beam_size
            else:
                generate_args['do_sample'] = True
                generate_args['top_k'] = config.top_k
                generate_args['top_p'] = config.top_p

            generation = model.generate(**generate_args)
            gen_text = tokenizer.batch_decode(generation['sequences'][:, num_tokens:], skip_special_tokens=True)

        outputs.append(json.dumps({
            "prefix": prefix,
            "gold_completion": gold_completion,
            "gen_completion": gen_text
        }))

        if (idx + 1) % config.checkpoint_frequency == 0:
            write_file_append(config.output_file, outputs)
            outputs = []
            gc.collect()

    write_file_append(config.output_file, outputs)
    print("Finished!")
