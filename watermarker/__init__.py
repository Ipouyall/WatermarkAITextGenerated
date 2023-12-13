from dataclasses import dataclass
import typing as t
import random
import os

from watermarker.utils import run_generator, run_detector


@dataclass()
class DetectorConfig:
    input_file: str
    model_name: str

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


@dataclass()
class GeneratorConfig:
    model_name: str

    output_directory: str
    prompt_file: str
    max_new_tokens: int = 300
    number_of_tests: int = 100
    checkpoint_frequency: int = 20

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

        self.output_file = f"{self.output_directory}/{self.model_name.replace('/', '-')}_strength_{self.strength}_frac_{self.fraction}_len_{self.max_new_tokens}_num_{self.number_of_tests}.jsonl"


class Runner:
    @staticmethod
    def run(config: t.Union[DetectorConfig, GeneratorConfig]):
        if isinstance(config, DetectorConfig):
            Runner._tun_detector(config)
        elif isinstance(config, GeneratorConfig):
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
