from dataclasses import dataclass
import typing as t
import random
import os

from watermarker.utils import run_generator, run_detector


def adjust_long_str_len(text, length):
    if len(text) > length:
        prt = length // 2
        l_prt = prt
        r_prt = length - prt - 5
        return text[:l_prt] + "[...]" + text[-r_prt:]
    return text


@dataclass()
class DetectorConfig:
    input_file: str
    model_name: str = "facebook/opt-350m"

    fraction: float = .5
    strength: float = 2.
    gamma: int = 0
    hash_key: int = 0

    watermark_threshold: float = 6.
    min_sequence_tokens: int = 200

    mode: t.Literal["gamma-penalty", "no-gamma-penalty"] = "no-gamma-penalty"

    def __post_init__(self):
        if not os.path.exists(self.input_file):
            print("Error: unable to find the specified input file")
            print(f"      file {self.input_file} doesn't exists!")
            raise Exception()

        if self.mode == "no-gamma-penalty":
            self.gamma = 0
        elif self.mode == "gamma-penalty" and self.gamma == 0:
            gamma = random.randint(2, 10)
            print("Warning: in 'gamma-penalty' mode, gamma shouldn't set to zero")
            print(f"        set gamma = {gamma}")
            self.gamma = gamma

    def __repr__(self):
        return f"""
DetectorConfig(
    input_file: {adjust_long_str_len(self.input_file, 50)},
    model_name: {self.model_name},
    fraction: {self.fraction},
    strength: {self.strength},
    gamma: {self.gamma},
    hash_key: {self.hash_key},
    watermark_threshold: {self.watermark_threshold},
    min_sequence_tokens: {self.min_sequence_tokens},
    mode: {self.mode},
)
        """


@dataclass()
class GeneratorConfig:
    model_name: str = "facebook/opt-350m"

    output_directory: str = "./data/LFQA/"
    prompt_file: str = "./data/LFQA/inputs.jsonl"
    max_new_tokens: int = 300
    number_of_tests: int = 100
    checkpoint_frequency: int = 20

    apply_watermarking: bool = True

    beam_size: t.Union[int, None] = None
    top_k: t.Union[int, None] = None
    top_p: t.Union[float, None] = 0.8

    fraction: float = .5
    strength: float = 2.
    gamma: int = 0
    hash_key: int = 0

    def __post_init__(self):
        if not os.path.exists(self.prompt_file):
            print("Error: unable to find prompt file")
            print(f"      file {self.prompt_file} doesn't exists!")
            raise Exception()

        if not os.path.exists(self.output_directory):
            print(f"Creating output directory: {self.output_directory}")
            os.mkdir(self.output_directory)

        if not self.apply_watermarking:
            self.strength = 0
            self.fraction = 0
            self.gamma = 0

        self.output_file = f"{self.output_directory}/{self.model_name.replace('/', '-')}_strength_{self.strength}_frac_{self.fraction}_len_{self.max_new_tokens}_num_{self.number_of_tests}.jsonl"

    def __repr__(self):
        return f"""
GeneratorConfig(
    model_name: {self.model_name},
    output_directory: {self.output_directory},
    prompt_file: {adjust_long_str_len(self.prompt_file, 50)},
    max_new_tokens: {self.max_new_tokens},
    number_of_tests: {self.number_of_tests},
    checkpoint_frequency: {self.checkpoint_frequency},
    beam_size: {self.beam_size},
    top_k: {self.top_k},
    top_p: {self.top_p},
    fraction: {self.fraction},
    strength: {self.strength},
    gamma: {self.gamma},
    hash_key: {self.hash_key},
    output_file: {adjust_long_str_len(self.output_file, 50)},
)
"""


class Runner:
    @staticmethod
    def run(config: t.Union[DetectorConfig, GeneratorConfig]):
        if isinstance(config, DetectorConfig):
            print("Starting the Detection task...")
            Runner._tun_detector(config)
        elif isinstance(config, GeneratorConfig):
            print("Starting the Generation task...")
            Runner._run_generator(config)
        else:
            print(f"Unknown config {config.__class__.__name__}")
            return

        print("Runner exiting successfully!")

    @staticmethod
    def _run_generator(config: GeneratorConfig):
        run_generator(config)

    @staticmethod
    def _tun_detector(config: DetectorConfig):
        run_detector(config)
