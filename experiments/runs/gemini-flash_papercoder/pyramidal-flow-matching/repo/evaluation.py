import torch
import torch.nn.functional as F
import os
import json
import logging
import subprocess
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, Union, List, Tuple
from torchvision.io import write_video

# Assuming Config and VideoGenerator are available from other modules.
# To avoid circular imports, these are typically imported in main.py and passed around,
# or accessed via a global config object.
# For standalone testing/linting, we'll use minimal stubs if actual imports fail.
try:
    from config import Config
    from inference import VideoGenerator
    # The VideoGenerator stub will rely on the other stubs if needed.
except ImportError as e:
    print(f"Failed to import project modules for evaluation.py: {e}. Using stub classes.")

    # Minimal stubs for Config and VideoGenerator
    class Config:
        def __init__(self):
            self.inference = self.InferenceConfig()
            self.evaluation = self.EvaluationConfig()

        class InferenceConfig:
            output_duration: int = 5
            output_fps: int = 24
            output_resolution: Tuple[int, int] = (256, 256)
            guidance_scale: float = 7.0

        class EvaluationConfig:
            evaluation_prompts_path: str = "dummy_prompts.txt"
            generated_video_output_dir: str = "generated_videos_stub"
            vbench_path: str = "/dummy/vbench_script.sh"
            evalcrafter_path: str = "/dummy/evalcrafter_script.sh"

    class VideoGenerator:
        def __init__(self, config: Config, *args, **kwargs):
            self.config = config
            self.device = torch.device("cpu")
            self.output_resolution = config.inference.output_resolution
            self.output_fps = config.inference.output_fps
            self.output_duration = config.inference.output_duration

        def generate_video(self, prompt: str, image_cond: Optional[torch.Tensor] = None, guidance_scale: float = 7.0, num_frames: int = 241, output_resolution: Tuple[int, int] = (768, 768)) -> torch.Tensor:
            """
            Stub: Returns a dummy video tensor in pixel space (T, H, W, C) uint8.
            """
            height, width = output_resolution
            # Dummy video: gradient frames (0-255)
            dummy_frames = []
            for i in range(num_frames):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :, 0] = (i * (255 / num_frames)) % 255 # Red gradient
                frame[:, :, 1] = (255 - (i * (255 / num_frames))) % 255 # Green gradient
                frame[:, :, 2] = 128 # Blue constant
                dummy_frames.append(frame)
            
            video_tensor = torch.from_numpy(np.stack(dummy_frames)).to(torch.uint8)
            print(f"Stub Generator: Generated dummy video for '{prompt}' with shape {video_tensor.shape}")
            return video_tensor # (T, H, W, C) uint8


# Setup logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class Evaluator:
    """
    The Evaluator class handles quantitative and qualitative evaluation of generated videos.
    It orchestrates video generation, saves them, and interfaces with external evaluation benchmarks.
    """

    def __init__(self, config: Config, generator: VideoGenerator):
        """
        Initializes the Evaluator.

        Args:
            config (Config): The global configuration object.
            generator (VideoGenerator): An initialized VideoGenerator instance ready for use.
        """
        self.config = config
        self.generator = generator
        logger.info("Evaluator initialized.")

    def _load_evaluation_prompts(self) -> List[str]:
        """
        Loads evaluation prompts from the specified file.
        Assumes one prompt per line.
        """
        prompts_path = Path(self.config.evaluation.evaluation_prompts_path)
        if not prompts_path.exists():
            logger.error(f"Evaluation prompts file not found at: {prompts_path}")
            return []
        
        with open(prompts_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} evaluation prompts from {prompts_path}.")
        return prompts

    def evaluate(self) -> Dict[str, Any]:
        """
        Orchestrates the end-to-end evaluation process.
        Loads prompts, generates videos, saves them, and calls external benchmarks.

        Returns:
            Dict[str, Any]: A dictionary containing aggregated evaluation results.
        """
        prompts = self._load_evaluation_prompts()
        if not prompts:
            logger.warning("No prompts loaded for evaluation. Skipping video generation and evaluation.")
            return {"vbench_results": {}, "evalcrafter_results": {}}

        output_dir = Path(self.config.evaluation.generated_video_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generated videos will be saved to: {output_dir}")

        generated_video_paths: List[str] = []
        
        num_frames = self.config.inference.output_duration * self.config.inference.output_fps
        output_resolution = self.config.inference.output_resolution
        guidance_scale = self.config.inference.guidance_scale

        logger.info(f"Starting video generation for {len(prompts)} prompts...")
        for idx, prompt in enumerate(tqdm(prompts, desc="Generating videos")):
            try:
                # Assuming generate_video returns a torch.Tensor of shape (T, H, W, C) with dtype uint8 [0, 255]
                # as required by torchvision.io.write_video.
                generated_video_tensor = self.generator.generate_video(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_frames=num_frames,
                    output_resolution=output_resolution
                )
                
                # Generate a unique filename
                filename = f"generated_video_{idx:03d}_{prompt[:30].replace(' ', '_').replace('/', '')}.mp4"
                video_path = output_dir / filename
                
                # Save the video using torchvision.io.write_video
                # write_video expects (T, H, W, C) of dtype torch.uint8
                write_video(
                    filename=str(video_path),
                    video_array=generated_video_tensor,
                    fps=self.config.inference.output_fps,
                    video_codec="libx264"
                )
                generated_video_paths.append(str(video_path))
                logger.info(f"Saved video for prompt '{prompt}' to {video_path}")

            except Exception as e:
                logger.error(f"Error generating or saving video for prompt '{prompt}' (index {idx}): {e}")
                continue

        logger.info(f"Finished generating {len(generated_video_paths)} videos.")

        all_results = {}
        if generated_video_paths:
            vbench_results = self._run_vbench(generated_video_paths, prompts)
            evalcrafter_results = self._run_evalcrafter(generated_video_paths, prompts)
            all_results = {'vbench_results': vbench_results, 'evalcrafter_results': evalcrafter_results}
        else:
            logger.warning("No videos were successfully generated. Skipping external evaluations.")

        return all_results

    def _run_vbench(self, generated_video_paths: List[str], prompts: List[str]) -> Dict[str, Any]:
        """
        Executes the VBench evaluation suite on the generated videos.
        This method assumes VBench is installed and callable via a shell script.

        Args:
            generated_video_paths (List[str]): List of file paths to the generated videos.
            prompts (List[str]): List of prompts corresponding to the generated videos.

        Returns:
            Dict[str, Any]: A dictionary of VBench evaluation metrics.
        """
        vbench_script_path = Path(self.config.evaluation.vbench_path)
        if not vbench_script_path.exists():
            logger.warning(f"VBENCH script not found at {vbench_script_path}. Skipping VBENCH evaluation.")
            return {"error": "VBENCH script not found"}

        # VBench typically takes a directory of videos and a config/prompt file
        # Create a temporary directory for VBench inputs/outputs
        temp_vbench_dir = Path(self.config.evaluation.generated_video_output_dir) / "vbench_temp"
        temp_vbench_dir.mkdir(parents=True, exist_ok=True)

        # Create a JSON file mapping video paths to prompts, or just a list of video paths
        # (Exact format depends on VBench CLI, this is a common assumption)
        vbench_input_data = []
        for i, (video_path, prompt) in enumerate(zip(generated_video_paths, prompts)):
            vbench_input_data.append({"video_path": video_path, "prompt": prompt})
        
        temp_input_json_path = temp_vbench_dir / "vbench_input.json"
        with open(temp_input_json_path, 'w', encoding='utf-8') as f:
            json.dump(vbench_input_data, f, indent=4)

        vbench_output_json_path = temp_vbench_dir / "vbench_results.json"

        logger.info(f"Running VBENCH evaluation using script: {vbench_script_path}")
        logger.info(f"  Input JSON: {temp_input_json_path}")
        logger.info(f"  Output JSON: {vbench_output_json_path}")

        try:
            # Example command structure, adjust based on actual VBench CLI
            # `python -m VBench.main --video_path_json <input_json> --output_path <output_json_dir>`
            command = [
                "python",
                str(vbench_script_path), # Assuming vbench_path points to VBench.main or similar executable script
                "--video_path_json", str(temp_input_json_path),
                "--output_path", str(vbench_output_json_path.parent) # Output path usually expects a directory
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info(f"VBENCH stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"VBENCH stderr: {result.stderr}")

            # Parse results
            if vbench_output_json_path.exists():
                with open(vbench_output_json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info("VBENCH evaluation completed successfully.")
                return results
            else:
                logger.error(f"VBENCH results file not found at {vbench_output_json_path}")
                return {"error": "VBENCH output file not found", "stdout": result.stdout, "stderr": result.stderr}

        except subprocess.CalledProcessError as e:
            logger.error(f"VBENCH evaluation failed with error: {e}")
            logger.error(f"VBENCH stdout: {e.stdout}")
            logger.error(f"VBENCH stderr: {e.stderr}")
            return {"error": "VBENCH script execution failed", "stdout": e.stdout, "stderr": e.stderr}
        except FileNotFoundError:
            logger.error(f"Python interpreter or VBENCH script not found. Command: {' '.join(command)}")
            return {"error": "Python interpreter or VBENCH script not found"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse VBENCH results JSON: {e}")
            return {"error": "Failed to parse VBENCH results", "detail": str(e)}
        finally:
            # Clean up temporary files (optional, can be useful for debugging)
            # import shutil
            # if temp_vbench_dir.exists():
            #     shutil.rmtree(temp_vbench_dir)
            logger.info(f"VBENCH temporary directory: {temp_vbench_dir}")


    def _run_evalcrafter(self, generated_video_paths: List[str], prompts: List[str]) -> Dict[str, Any]:
        """
        Executes the EvalCrafter evaluation suite on the generated videos.
        This method assumes EvalCrafter is installed and callable via a shell script.

        Args:
            generated_video_paths (List[str]): List of file paths to the generated videos.
            prompts (List[str]): List of prompts corresponding to the generated videos.

        Returns:
            Dict[str, Any]: A dictionary of EvalCrafter evaluation metrics.
        """
        evalcrafter_script_path = Path(self.config.evaluation.evalcrafter_path)
        if not evalcrafter_script_path.exists():
            logger.warning(f"EvalCrafter script not found at {evalcrafter_script_path}. Skipping EvalCrafter evaluation.")
            return {"error": "EvalCrafter script not found"}

        temp_evalcrafter_dir = Path(self.config.evaluation.generated_video_output_dir) / "evalcrafter_temp"
        temp_evalcrafter_dir.mkdir(parents=True, exist_ok=True)

        # EvalCrafter typically takes a JSON/CSV file with video paths and prompts,
        # and an output directory.
        evalcrafter_input_data = []
        for i, (video_path, prompt) in enumerate(zip(generated_video_paths, prompts)):
            evalcrafter_input_data.append({"video_path": video_path, "prompt": prompt})
        
        temp_input_json_path = temp_evalcrafter_dir / "evalcrafter_input.json"
        with open(temp_input_json_path, 'w', encoding='utf-8') as f:
            json.dump(evalcrafter_input_data, f, indent=4)

        evalcrafter_output_json_path = temp_evalcrafter_dir / "evalcrafter_results.json"

        logger.info(f"Running EvalCrafter evaluation using script: {evalcrafter_script_path}")
        logger.info(f"  Input JSON: {temp_input_json_path}")
        logger.info(f"  Output JSON: {evalcrafter_output_json_path}")

        try:
            # Example command structure, adjust based on actual EvalCrafter CLI
            # `python -m EvalCrafter.main --video_list <input_json> --output_dir <output_json_dir>`
            command = [
                "python",
                str(evalcrafter_script_path), # Assuming evalcrafter_path points to EvalCrafter.main or similar executable script
                "--video_list", str(temp_input_json_path),
                "--output_dir", str(evalcrafter_output_json_path.parent)
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            logger.info(f"EvalCrafter stdout: {result.stdout}")
            if result.stderr:
                logger.warning(f"EvalCrafter stderr: {result.stderr}")

            # Parse results
            if evalcrafter_output_json_path.exists():
                with open(evalcrafter_output_json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info("EvalCrafter evaluation completed successfully.")
                return results
            else:
                logger.error(f"EvalCrafter results file not found at {evalcrafter_output_json_path}")
                return {"error": "EvalCrafter output file not found", "stdout": result.stdout, "stderr": result.stderr}

        except subprocess.CalledProcessError as e:
            logger.error(f"EvalCrafter evaluation failed with error: {e}")
            logger.error(f"EvalCrafter stdout: {e.stdout}")
            logger.error(f"EvalCrafter stderr: {e.stderr}")
            return {"error": "EvalCrafter script execution failed", "stdout": e.stdout, "stderr": e.stderr}
        except FileNotFoundError:
            logger.error(f"Python interpreter or EvalCrafter script not found. Command: {' '.join(command)}")
            return {"error": "Python interpreter or EvalCrafter script not found"}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse EvalCrafter results JSON: {e}")
            return {"error": "Failed to parse EvalCrafter results", "detail": str(e)}
        finally:
            # Clean up temporary files (optional)
            # import shutil
            # if temp_evalcrafter_dir.exists():
            #     shutil.rmtree(temp_evalcrafter_dir)
            logger.info(f"EvalCrafter temporary directory: {temp_evalcrafter_dir}")

    def _run_user_study(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Placeholder for potential user study setup.
        As per the design, this method is not intended to execute an interactive user study,
        but rather to prepare data for one.
        """
        logger.info("User study function called. This is a placeholder for preparing user study data.")
        # In a real scenario, this might save metadata, generate web pages, etc.
        return {"user_study_status": "data_prepared_for_external_user_study"}


if __name__ == "__main__":
    print("--- Testing evaluation.py ---")

    # Create a dummy config for testing
    dummy_config = Config()
    # Ensure dummy_prompts.txt exists for testing
    dummy_prompts_path = Path(dummy_config.evaluation.evaluation_prompts_path)
    dummy_prompts_path.write_text("A dog running in a field.\nA cat sleeping on a couch.\nA car driving on a road.")
    
    # Create a dummy VideoGenerator
    dummy_generator = VideoGenerator(dummy_config)

    # Instantiate Evaluator
    evaluator = Evaluator(dummy_config, dummy_generator)

    # Run evaluation
    print("\nStarting dummy evaluation...")
    evaluation_results = evaluator.evaluate()
    print("\nDummy Evaluation Results:")
    print(json.dumps(evaluation_results, indent=2))

    # Clean up dummy prompt file
    dummy_prompts_path.unlink()

    # Clean up dummy generated videos directory
    generated_videos_dir = Path(dummy_config.evaluation.generated_video_output_dir)
    if generated_videos_dir.exists():
        import shutil
        shutil.rmtree(generated_videos_dir)
        print(f"Cleaned up dummy generated videos directory: {generated_videos_dir}")

    print("\nAll evaluation.py tests completed.")

