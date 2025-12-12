import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch.export import Dim

from transformers import AutoImageProcessor, AutoModelForObjectDetection, BatchFeature

# Ensure project root is on sys.path so `e2e_implem` can be imported when run directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import after path setup
from e2e_implem.e2e_utils import E2EModel


@dataclass
class BenchmarkResult:
    config_name: str
    fps: float
    num_frames: int
    total_time: float
    notes: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for running a benchmark"""

    cap: cv2.VideoCapture
    out_path: str
    model_id: str
    device: str
    threshold: float
    frames_limit: int
    warmup: int
    use_fast: Optional[bool] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark object detection pipelines on a video")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model_id", default="facebook/detr-resnet-50", help="HuggingFace model id")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--out_dir", default="./benchmark_outputs_updated", help="Directory to save outputs")
    parser.add_argument("--frames_limit", type=int, default=0, help="Limit number of frames (0 means all)")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup frames before timing")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for drawing boxes")
    parser.add_argument("--video_fps", type=float, default=0.0, help="Override output video FPS (0 keeps input FPS)")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def draw_detections(
    frame_bgr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    id2label: dict[int, str],
    threshold: float,
) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    for score, label_id, box in zip(scores, labels, boxes):
        score_f = float(score)
        if score_f < threshold:
            continue
        label = int(label_id)
        x0, y0, x1, y1 = [int(round(v)) for v in box.tolist()]
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
        text = f"{id2label.get(label, str(label))}: {score_f:.2f}"
        cv2.putText(frame_bgr, text, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame_bgr


def time_block() -> tuple[callable, callable, list[float]]:
    start = [0.0]
    total = [0.0]

    def begin():
        start[0] = time.perf_counter()

    def end():
        total[0] += time.perf_counter() - start[0]

    return begin, end, total


def get_video_writer(in_cap: cv2.VideoCapture, out_path: str, override_fps: float = 0.0) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = override_fps if override_fps > 0 else max(1e-3, in_cap.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter(out_path, fourcc, fps, (width, height))


def benchmark_loop(
    cap: cv2.VideoCapture,
    frames_limit: int,
    warmup: int,
    process_frame_fn,
) -> tuple[int, float, float]:
    """Generic per-frame loop. The callback must handle timing_begin/end and writing frames.
    Returns (num_frames, inference_time_seconds, total_time_seconds)."""
    timing_begin, timing_end, inference_total = time_block()

    # Track total execution time after warmup
    total_execution_start = None
    total_execution_time = 0.0

    num_frames = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frames_limit and num_frames >= frames_limit:
            break

        # Start total timing after warmup
        if num_frames == warmup:
            total_execution_start = time.perf_counter()

        process_frame_fn(num_frames, frame_bgr, timing_begin, timing_end)
        num_frames += 1

    # Calculate total execution time (after warmup)
    if total_execution_start is not None:
        total_execution_time = time.perf_counter() - total_execution_start

    return num_frames, inference_total[0], total_execution_time


def calculate_fps(num_frames: int, warmup: int, total_time: float) -> float:
    """Calculate FPS from timing data"""
    proc_frames = max(0, num_frames - warmup)
    return (proc_frames / total_time) if total_time > 0 and proc_frames > 0 else 0.0


def prepare_first_frame(cap: cv2.VideoCapture, config_name: str) -> tuple[bool, Optional[np.ndarray]]:
    """Read and reset to prepare for processing. Returns (success, frame_bgr)"""
    ret, frame_bgr = cap.read()
    if not ret:
        return False, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return True, frame_bgr


def create_dynamic_shapes_config() -> dict:
    """Create standard dynamic shapes configuration for export"""
    height_dim = Dim("height", min=32, max=4096)
    width_dim = Dim("width", min=32, max=4096)
    return {
        "images": {2: height_dim, 3: width_dim},
        "post_process_kwargs": {
            "target_sizes": None,
            "threshold": None,  # Must match input structure
        },
    }


def get_processor(model_id: str, use_fast: Optional[bool]) -> AutoImageProcessor:
    """Get image processor with optional use_fast parameter"""
    if use_fast is not None:
        return AutoImageProcessor.from_pretrained(model_id, use_fast=use_fast)
    return AutoImageProcessor.from_pretrained(model_id)


def run_normal(config: BenchmarkConfig) -> BenchmarkResult:
    config_name = f"normal(use_fast={config.use_fast})"
    model = AutoModelForObjectDetection.from_pretrained(config.model_id).to(config.device)
    processor = get_processor(config.model_id, config.use_fast)
    writer = get_video_writer(config.cap, config.out_path)

    def process_frame(num_frames: int, frame_bgr: np.ndarray, timing_begin, timing_end):
        pil_img = bgr_to_pil(frame_bgr)
        if num_frames >= config.warmup:
            timing_begin()
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {"pixel_values": inputs["pixel_values"].to(config.device)}
        with torch.no_grad():
            outputs = model(**inputs)
        # Post-process (included in timing)
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[frame_bgr.shape[0], frame_bgr.shape[1]]]),
            threshold=config.threshold,
        )
        if num_frames >= config.warmup:
            timing_end()
        if len(results) > 0:
            res0 = results[0]
            frame_out = draw_detections(
                frame_bgr,
                res0["boxes"].cpu().numpy(),
                res0["scores"].cpu().numpy(),
                res0["labels"].cpu().numpy(),
                model.config.id2label,
                config.threshold,
            )
        else:
            frame_out = frame_bgr
        writer.write(frame_out)

    num_frames, inference_time, total_time = benchmark_loop(
        config.cap, config.frames_limit, config.warmup, process_frame
    )
    writer.release()
    fps = calculate_fps(num_frames, config.warmup, inference_time)
    return BenchmarkResult(config_name=config_name, fps=fps, num_frames=num_frames, total_time=total_time)


def run_e2e_torch_export(config: BenchmarkConfig) -> BenchmarkResult:
    config_name = f"torch.export+E2E(use_fast={config.use_fast})"
    model = E2EModel(
        config.model_id, AutoModelForObjectDetection, AutoImageProcessor, "post_process_object_detection"
    ).to(config.device)
    processor = model.processor

    # Prepare first frame for export
    success, frame_bgr = prepare_first_frame(config.cap, config_name)
    if not success:
        return BenchmarkResult(config_name=config_name, fps=0.0, num_frames=0, notes="empty video")

    images = model.get_tensors_inputs(bgr_to_pil(frame_bgr), device=config.device)
    example_input = {
        "images": images.to(config.device),
        "post_process_kwargs": {
            "target_sizes": torch.tensor([[frame_bgr.shape[0], frame_bgr.shape[1]]], device=config.device),
            "threshold": config.threshold,
        },
    }

    export_dynamic_shapes = create_dynamic_shapes_config()
    exported_program = torch.export.export(
        model,
        args=(),
        kwargs=example_input,
        dynamic_shapes=export_dynamic_shapes,
        strict=False,
    )
    exported = exported_program.module()
    writer = get_video_writer(config.cap, config.out_path)

    def process_frame(num_frames: int, frame_bgr: np.ndarray, timing_begin, timing_end):
        pil_img = bgr_to_pil(frame_bgr)
        if num_frames >= config.warmup:
            timing_begin()
        images = model.get_tensors_inputs(pil_img, device=config.device)
        target_sizes = torch.tensor([[frame_bgr.shape[0], frame_bgr.shape[1]]], device=config.device)
        with torch.no_grad():
            outputs = exported(
                **{
                    "images": images,
                    "post_process_kwargs": {"target_sizes": target_sizes, "threshold": config.threshold},
                }
            )
        if num_frames >= config.warmup:
            timing_end()

        boxes = outputs[0]["boxes"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()
        frame_out = draw_detections(
            frame_bgr,
            boxes,
            scores,
            labels,
            processor.image_processor.config.id2label
            if hasattr(processor, "image_processor")
            else model.model.config.id2label,
            config.threshold,
        )
        writer.write(frame_out)

    num_frames, inference_time, total_time = benchmark_loop(
        config.cap, config.frames_limit, config.warmup, process_frame
    )
    writer.release()
    fps = calculate_fps(num_frames, config.warmup, inference_time)
    return BenchmarkResult(config_name=config_name, fps=fps, num_frames=num_frames, total_time=total_time)


def run_torch_export_model_only(config: BenchmarkConfig) -> BenchmarkResult:
    """Export only the model with torch.export and keep preprocessing/postprocessing in Python.
    Measures FPS for preprocessing + model forward pass only (excludes I/O and drawing)."""
    config_name = f"torch.export(model_only, use_fast={config.use_fast})"
    model = AutoModelForObjectDetection.from_pretrained(config.model_id).to(config.device)
    processor = AutoImageProcessor.from_pretrained(config.model_id, use_fast=config.use_fast)

    # Prepare first frame for export
    success, frame_bgr = prepare_first_frame(config.cap, config_name)
    if not success:
        return BenchmarkResult(config_name=config_name, fps=0.0, num_frames=0, notes="empty video")

    pil_img = bgr_to_pil(frame_bgr)
    inputs_pt = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs_pt["pixel_values"].to(config.device)

    dynamic_shapes = {"pixel_values": None}
    exported_program = torch.export.export(
        model,
        args=(),
        kwargs={"pixel_values": pixel_values},
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    exported_model = exported_program.module()
    writer = get_video_writer(config.cap, config.out_path)

    def process_frame(num_frames: int, frame_bgr: np.ndarray, timing_begin, timing_end):
        pil_img = bgr_to_pil(frame_bgr)
        if num_frames >= config.warmup:
            timing_begin()
        inputs_pt = processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs_pt["pixel_values"].to(config.device)
        with torch.no_grad():
            model_out = exported_model(pixel_values=pixel_values)

        target_sizes = torch.tensor([[frame_bgr.shape[0], frame_bgr.shape[1]]])
        results = processor.post_process_object_detection(
            model_out, target_sizes=target_sizes, threshold=config.threshold
        )
        if num_frames >= config.warmup:
            timing_end()

        if len(results) > 0:
            res0 = results[0]
            frame_out = draw_detections(
                frame_bgr,
                res0["boxes"].detach().cpu().numpy(),
                res0["scores"].detach().cpu().numpy(),
                res0["labels"].detach().cpu().numpy(),
                model.config.id2label,
                config.threshold,
            )
        else:
            frame_out = frame_bgr
        writer.write(frame_out)

    num_frames, inference_time, total_time = benchmark_loop(
        config.cap, config.frames_limit, config.warmup, process_frame
    )
    writer.release()
    fps = calculate_fps(num_frames, config.warmup, inference_time)
    return BenchmarkResult(config_name=config_name, fps=fps, num_frames=num_frames, total_time=total_time)


def run_e2e_onnx_runtime(config: BenchmarkConfig) -> BenchmarkResult:
    config_name = f"onnxruntime(e2e=True, use_fast={config.use_fast})"

    # Build E2E model and export to ONNX
    model = E2EModel(
        config.model_id, AutoModelForObjectDetection, AutoImageProcessor, "post_process_object_detection"
    ).to(config.device)

    # Prepare first frame for export
    success, frame_bgr = prepare_first_frame(config.cap, config_name)
    if not success:
        return BenchmarkResult(config_name=config_name, fps=0.0, num_frames=0, notes="empty video")

    images = model.get_tensors_inputs(bgr_to_pil(frame_bgr), device=config.device)
    example_input = {
        "images": images,
        "post_process_kwargs": {
            "target_sizes": torch.tensor([[frame_bgr.shape[0], frame_bgr.shape[1]]]),
            "threshold": config.threshold,
        },
    }

    export_dynamic_shapes = create_dynamic_shapes_config()
    safe_model_name = config.model_id.replace("/", "_")
    onnx_path = f"{safe_model_name}_e2e_dynamo_dynamic.onnx"

    onnx_program = torch.onnx.export(
        model,
        args=(),
        kwargs=example_input,
        f=onnx_path,
        input_names=["images", "post_process_kwargs"],
        output_names=["output"],
        dynamo=True,
        dynamic_shapes=export_dynamic_shapes,
    )
    onnx_program.optimize()
    onnx_program.save(onnx_path)

    providers = ["CUDAExecutionProvider"] if config.device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    onnx_input_names = [i.name for i in sess.get_inputs()]
    writer = get_video_writer(config.cap, config.out_path)

    def process_frame(num_frames: int, frame_bgr: np.ndarray, timing_begin, timing_end):
        images = bgr_to_pil(frame_bgr)
        if num_frames >= config.warmup:
            timing_begin()
        target_sizes = np.array([[frame_bgr.shape[0], frame_bgr.shape[1]]], dtype=np.int64)
        image_array = np.array(images).transpose(2, 0, 1)[np.newaxis, ...]
        inputs = {onnx_input_names[0]: image_array, onnx_input_names[1]: target_sizes}
        outputs = sess.run(None, inputs)
        if num_frames >= config.warmup:
            timing_end()
        # Expect outputs as list; assume [scores, labels, boxes]
        scores, labels, boxes = outputs[0], outputs[1], outputs[2]
        frame_out = draw_detections(frame_bgr, boxes, scores, labels, model.model.config.id2label, config.threshold)
        writer.write(frame_out)

    num_frames, inference_time, total_time = benchmark_loop(
        config.cap, config.frames_limit, config.warmup, process_frame
    )
    writer.release()
    fps = calculate_fps(num_frames, config.warmup, inference_time)
    return BenchmarkResult(config_name=config_name, fps=fps, num_frames=num_frames, total_time=total_time)


def run_onnx_model_only(config: BenchmarkConfig) -> BenchmarkResult:
    """Export only the model to ONNX (ORT) and keep preprocessing/postprocessing in Python.
    Measures FPS for preprocessing + ORT forward only (excludes I/O and drawing)."""
    config_name = f"onnxruntime(model_only, use_fast={config.use_fast})"
    processor = AutoImageProcessor.from_pretrained(config.model_id, use_fast=config.use_fast)
    torch_model = AutoModelForObjectDetection.from_pretrained(config.model_id)

    success, frame_bgr = prepare_first_frame(config.cap, config_name)
    if not success:
        return BenchmarkResult(config_name=config_name, fps=0.0, num_frames=0, notes="empty video")

    pil_img = bgr_to_pil(frame_bgr)
    inputs_pt = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs_pt["pixel_values"]

    dynamic_shapes = {"pixel_values": None}
    safe_model = config.model_id.replace("/", "_")
    onnx_path = os.path.join(os.path.dirname(config.out_path), f"{safe_model}_model_only.onnx")

    onnx_program = torch.onnx.export(
        torch_model,
        args=(),
        kwargs={"pixel_values": pixel_values},
        f=onnx_path,
        input_names=["pixel_values"],
        output_names=["output"],
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
    )
    onnx_program.optimize()
    onnx_program.save(onnx_path)

    providers = ["CUDAExecutionProvider"] if config.device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    onnx_input_name = sess.get_inputs()[0].name
    writer = get_video_writer(config.cap, config.out_path)

    def process_frame(num_frames: int, frame_bgr: np.ndarray, timing_begin, timing_end):
        pil_img = bgr_to_pil(frame_bgr)
        if num_frames >= config.warmup:
            timing_begin()
        if config.use_fast:
            inputs_pt = processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs_pt["pixel_values"].numpy(force=True)
        else:
            inputs_pt = processor(images=pil_img, return_tensors="np")
            pixel_values = inputs_pt["pixel_values"]
        outputs = sess.run(None, {onnx_input_name: pixel_values})
        pt_outputs = {"logits": torch.from_numpy(outputs[0]), "pred_boxes": torch.from_numpy(outputs[1])}
        pt_outputs_bf = BatchFeature(data=pt_outputs, tensor_type="pt")
        results = processor.post_process_object_detection(
            pt_outputs_bf,
            target_sizes=torch.tensor([[frame_bgr.shape[0], frame_bgr.shape[1]]]),
            threshold=config.threshold,
        )
        if num_frames >= config.warmup:
            timing_end()

        if len(results) > 0:
            res0 = results[0]
            frame_out = draw_detections(
                frame_bgr,
                res0["boxes"].numpy(),
                res0["scores"].numpy(),
                res0["labels"].numpy(),
                torch_model.config.id2label,
                config.threshold,
            )
        else:
            frame_out = frame_bgr
        writer.write(frame_out)

    num_frames, inference_time, total_time = benchmark_loop(
        config.cap, config.frames_limit, config.warmup, process_frame
    )
    writer.release()
    fps = calculate_fps(num_frames, config.warmup, inference_time)
    return BenchmarkResult(config_name=config_name, fps=fps, num_frames=num_frames, total_time=total_time)


def write_results_csv(out_dir: str, results: list[BenchmarkResult]) -> str:
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "benchmark_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "fps", "num_frames", "total_time_sec", "notes"])
        for r in results:
            writer.writerow([r.config_name, f"{r.fps:.4f}", r.num_frames, f"{r.total_time:.4f}", r.notes])
    return csv_path


def run_benchmark(config: BenchmarkConfig, benchmark_type: str, use_fast: Optional[bool]) -> BenchmarkResult:
    """Run a specific benchmark type with the given configuration"""
    config.use_fast = use_fast
    config.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    benchmark_functions = {
        "normal": run_normal,
        "torch_export_model_only": run_torch_export_model_only,
        "torch_export_e2e": run_e2e_torch_export,
        "onnx_model_only": run_onnx_model_only,
        "onnx_e2e": run_e2e_onnx_runtime,
    }

    return benchmark_functions[benchmark_type](config)


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    base_name = os.path.splitext(os.path.basename(args.video))[0]
    safe_model = args.model_id.replace("/", "_")

    # Create model-specific output directory
    model_output_dir = os.path.join(args.out_dir, safe_model)
    ensure_dir(model_output_dir)

    # Create base configuration
    base_config = BenchmarkConfig(
        cap=cap,
        out_path="",  # Will be set per benchmark
        model_id=args.model_id,
        device=args.device,
        threshold=args.threshold,
        frames_limit=args.frames_limit,
        warmup=args.warmup,
    )

    results: list[BenchmarkResult] = []

    # Define all benchmarks to run
    benchmarks = [
        ("normal", None, "normal"),
        ("torch_export_model_only", False, "torch_export_modelonly_fastFalse"),
        ("torch_export_model_only", True, "torch_export_modelonly_fastTrue"),
        ("torch_export_e2e", True, "torch_export_e2e"),
        ("onnx_model_only", False, "onnx_modelonly_fastFalse"),
        ("onnx_model_only", True, "onnx_modelonly_fastTrue"),
        ("onnx_e2e", True, "onnx_e2e"),
    ]

    for benchmark_type, use_fast, suffix in benchmarks:
        base_config.out_path = os.path.join(model_output_dir, f"{base_name}_{safe_model}_{suffix}.mp4")
        result = run_benchmark(base_config, benchmark_type, use_fast)
        results.append(result)

    cap.release()

    csv_path = write_results_csv(model_output_dir, results)
    print("\nBenchmark results:")
    for r in results:
        proc_frames = max(0, r.num_frames - base_config.warmup)
        total_fps = proc_frames / r.total_time if r.total_time > 0 else 0.0
        print(
            f"- {r.config_name}: {r.fps:.3f} FPS (inference) | {total_fps:.3f} FPS (total) | "
            f"{r.total_time:.3f}s total | {r.num_frames} frames {('(' + r.notes + ')') if r.notes else ''}"
        )
    print(f"\nSummary saved to: {csv_path}")
    print(f"All outputs saved to: {model_output_dir}")


if __name__ == "__main__":
    # tested checkpoints (working):
    # facebook/detr-resnet-50
    # PekingU/rtdetr_v2_r101vd
    # PekingU/rtdetr_r50vd_coco_o365
    # ustc-community/dfine-large-obj365
    main()
