"""SAM 2.1 adapter; uses official video predictor APIs."""

from contextlib import nullcontext
from pathlib import Path
import numpy as np

MODELS = {
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
    "base-plus": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam2.1_hiera_base_plus.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_large.pt"),
}


class Sam2Tracker:
    def __init__(self, checkpoint: Path, model: str, device: str):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Check PyTorch installation or select --device cpu")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. Run tools/download_model.py")
        self.device = device
        self.checkpoint = checkpoint
        self.model = model
        self.predictor = build_sam2_video_predictor(
            MODELS[model][0], str(checkpoint), device=device,
            apply_postprocessing=False,  # Our explicit cleanup is saved separately.
        )

    def predict(self, frames: Path, prompts, *, initial_masks=None, prompt_frames=None):
        torch = self.torch
        precision = (torch.autocast("cuda", dtype=torch.bfloat16)
                     if self.device == "cuda" and torch.cuda.is_bf16_supported()
                     else nullcontext())
        with torch.inference_mode(), precision:
            state = self.predictor.init_state(str(frames), offload_video_to_cpu=True,
                                               offload_state_to_cpu=True)
            try:
                for prompt in prompts:
                    kwargs = {"inference_state": state,
                              "frame_idx": (prompt_frames or {}).get(prompt.object_id, 0),
                              "obj_id": prompt.object_id}
                    if initial_masks is not None:
                        mask = np.asarray(initial_masks[prompt.object_id], dtype=bool)
                        if mask.shape != (state["video_height"], state["video_width"]) or not mask.any():
                            raise ValueError("Continuation requires a nonempty full-frame mask per object")
                        if not 0 <= kwargs["frame_idx"] < state["num_frames"]:
                            raise ValueError("Continuation frame is outside the window")
                        self.predictor.add_new_mask(**kwargs, mask=mask)
                    else:
                        if prompt.box is not None:
                            kwargs["box"] = np.array(prompt.box, dtype=np.float32)
                        if prompt.points:
                            kwargs["points"] = np.array(prompt.points, dtype=np.float32)
                            kwargs["labels"] = np.array(prompt.labels, dtype=np.int32)
                        self.predictor.add_new_points_or_box(**kwargs)
                for frame_index, ids, logits in self.predictor.propagate_in_video(state, start_frame_idx=0):
                    binary = (logits > 0).cpu().numpy()
                    yield int(frame_index), {int(object_id): binary[index, 0]
                                             for index, object_id in enumerate(ids)}
            finally:
                self.predictor.reset_state(state)
