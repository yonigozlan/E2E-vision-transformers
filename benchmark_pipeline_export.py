"""
Benchmark object detection pipelines using the new Transformers export API.

This script tests multiple models with various export configurations:
- Normal pipeline (baseline)
- torch.export model-only (fast processor) - model export with Python pre/post
- torch.export model-only (slow processor) - model export with Python pre/post
- torch.export E2E - full pipeline export (preprocessing + model + postprocessing)
- ONNX model-only (fast processor) - model export with Python pre/post
- ONNX model-only (slow processor) - model export with Python pre/post
- ONNX E2E - full pipeline export (preprocessing + model + postprocessing)

Supported models:
- facebook/detr-resnet-50
- PekingU/rtdetr_v2_r101vd
- PekingU/rtdetr_r50vd_coco_o365
- ustc-community/dfine-large-obj365
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from transformers import AutoImageProcessor, AutoModelForObjectDetection, BatchFeature, pipeline


@dataclass
class BenchmarkResult:
    model_id: str
    config_name: str
    fps: float
    num_frames: int
    total_time: float
    inference_time: float
    notes: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark object detection pipelines with export")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "facebook/detr-resnet-50",
            "PekingU/rtdetr_v2_r101vd",
            "PekingU/rtdetr_r50vd_coco_o365",
            "ustc-community/dfine-large-obj365",
        ],
        help="List of model IDs to benchmark",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("--out_dir", default="./benchmark_pipeline_outputs", help="Directory to save outputs")
    parser.add_argument("--frames_limit", type=int, default=100, help="Limit number of frames (0 means all)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames before timing")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score threshold for detections")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def draw_detections(
    frame_bgr: np.ndarray,
    results: list[dict],
    threshold: float,
) -> np.ndarray:
    """Draw bounding boxes on frame from pipeline output"""
    if not results or len(results) == 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    for detection in results:
        score = detection["score"]
        if score < threshold:
            continue

        box = detection["box"]
        label = detection["label"]

        x0 = int(box["xmin"])
        y0 = int(box["ymin"])
        x1 = int(box["xmax"])
        y1 = int(box["ymax"])

        # Clamp to frame bounds
        x0 = max(0, min(w - 1, x0))
        y0 = max(0, min(h - 1, y0))
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))

        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (0, 0, 255), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(frame_bgr, text, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame_bgr


def time_block():
    """Create timing context"""
    start = [0.0]
    total = [0.0]

    def begin():
        start[0] = time.perf_counter()

    def end():
        total[0] += time.perf_counter() - start[0]

    return begin, end, total


def get_video_writer(in_cap: cv2.VideoCapture, out_path: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width = int(in_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(in_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = max(1e-3, in_cap.get(cv2.CAP_PROP_FPS))
    return cv2.VideoWriter(out_path, fourcc, fps, (width, height))


def benchmark_normal_pipeline(
    model_id: str,
    cap: cv2.VideoCapture,
    device: str,
    threshold: float,
    frames_limit: int,
    warmup: int,
    out_path: str,
) -> BenchmarkResult:
    """Benchmark normal pipeline without export"""
    print("\n  Testing normal pipeline...")

    pipe = pipeline("object-detection", model=model_id, device=device)
    writer = get_video_writer(cap, out_path)

    timing_begin, timing_end, inference_total = time_block()
    total_execution_start = None
    total_execution_time = 0.0
    num_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frames_limit and num_frames >= frames_limit:
            break

        if num_frames == warmup:
            total_execution_start = time.perf_counter()

        pil_img = bgr_to_pil(frame_bgr)

        if num_frames >= warmup:
            timing_begin()

        results = pipe(pil_img, threshold=threshold)

        if num_frames >= warmup:
            timing_end()

        frame_out = draw_detections(frame_bgr, results, threshold)
        writer.write(frame_out)
        num_frames += 1

    if total_execution_start is not None:
        total_execution_time = time.perf_counter() - total_execution_start

    writer.release()

    proc_frames = max(0, num_frames - warmup)
    fps = (proc_frames / inference_total[0]) if inference_total[0] > 0 and proc_frames > 0 else 0.0

    print(f"    ✓ Normal pipeline: {fps:.2f} FPS (inference) | {num_frames} frames")

    return BenchmarkResult(
        model_id=model_id,
        config_name="normal",
        fps=fps,
        num_frames=num_frames,
        total_time=total_execution_time,
        inference_time=inference_total[0],
    )


def benchmark_torch_export_model_only(
    model_id: str,
    cap: cv2.VideoCapture,
    device: str,
    threshold: float,
    frames_limit: int,
    warmup: int,
    out_path: str,
    use_fast: bool = True,
) -> BenchmarkResult:
    """Benchmark torch.export model-only (preprocessing/postprocessing in Python)"""
    config_name = f"torch_export_model_only(fast={use_fast})"
    print(f"\n  Testing torch.export model-only (use_fast={use_fast})...")

    model = AutoModelForObjectDetection.from_pretrained(model_id).to(device)
    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=use_fast)
    id2label = model.config.id2label

    # Get first frame for export
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_bgr = cap.read()
    if not ret:
        return BenchmarkResult(
            model_id=model_id,
            config_name=config_name,
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes="empty video",
        )

    # Export model only
    pil_img = bgr_to_pil(frame_bgr)
    inputs_pt = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs_pt["pixel_values"].to(device)

    try:
        dynamic_shapes = {"pixel_values": None}
        exported_program = torch.export.export(
            model,
            args=(),
            kwargs={"pixel_values": pixel_values},
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
        exported_model = exported_program.module()
    except Exception as e:
        print(f"    ✗ Export failed: {e}")
        return BenchmarkResult(
            model_id=model_id,
            config_name=config_name,
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes=f"export_failed: {str(e)[:50]}",
        )

    writer = get_video_writer(cap, out_path)
    timing_begin, timing_end, inference_total = time_block()
    total_execution_start = None
    total_execution_time = 0.0
    num_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frames_limit and num_frames >= frames_limit:
            break

        if num_frames == warmup:
            total_execution_start = time.perf_counter()

        pil_img = bgr_to_pil(frame_bgr)

        if num_frames >= warmup:
            timing_begin()

        # Preprocess in Python
        inputs_pt = processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs_pt["pixel_values"].to(device)

        # Model inference
        with torch.no_grad():
            model_out = exported_model(pixel_values=pixel_values)

        # Postprocess in Python
        target_sizes = torch.tensor([[frame_bgr.shape[0], frame_bgr.shape[1]]])
        results = processor.post_process_object_detection(model_out, target_sizes=target_sizes, threshold=threshold)

        if num_frames >= warmup:
            timing_end()

        # Draw detections
        if len(results) > 0:
            res0 = results[0]
            boxes = res0["boxes"].detach().cpu().numpy()
            scores = res0["scores"].detach().cpu().numpy()
            labels = res0["labels"].detach().cpu().numpy()

            # Convert to pipeline format
            results_formatted = []
            for box, score, label_id in zip(boxes, scores, labels):
                label_id_int = int(label_id)
                label_name = id2label.get(label_id_int, str(label_id_int))
                results_formatted.append(
                    {
                        "score": float(score),
                        "label": label_name,
                        "box": {
                            "xmin": float(box[0]),
                            "ymin": float(box[1]),
                            "xmax": float(box[2]),
                            "ymax": float(box[3]),
                        },
                    }
                )

            frame_out = draw_detections(frame_bgr, results_formatted, threshold)
        else:
            frame_out = frame_bgr

        writer.write(frame_out)
        num_frames += 1

    if total_execution_start is not None:
        total_execution_time = time.perf_counter() - total_execution_start

    writer.release()

    proc_frames = max(0, num_frames - warmup)
    fps = (proc_frames / inference_total[0]) if inference_total[0] > 0 and proc_frames > 0 else 0.0

    print(f"    ✓ torch.export model-only: {fps:.2f} FPS (inference) | {num_frames} frames")

    return BenchmarkResult(
        model_id=model_id,
        config_name=config_name,
        fps=fps,
        num_frames=num_frames,
        total_time=total_execution_time,
        inference_time=inference_total[0],
    )


def benchmark_onnx_model_only(
    model_id: str,
    cap: cv2.VideoCapture,
    device: str,
    threshold: float,
    frames_limit: int,
    warmup: int,
    out_path: str,
    onnx_path: str,
    use_fast: bool = True,
) -> BenchmarkResult:
    """Benchmark ONNX model-only export (preprocessing/postprocessing in Python)"""
    config_name = f"onnx_model_only(fast={use_fast})"
    print(f"\n  Testing ONNX model-only (use_fast={use_fast})...")

    processor = AutoImageProcessor.from_pretrained(model_id, use_fast=use_fast)
    torch_model = AutoModelForObjectDetection.from_pretrained(model_id)
    id2label = torch_model.config.id2label

    # Get first frame for export
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_bgr = cap.read()
    if not ret:
        return BenchmarkResult(
            model_id=model_id,
            config_name=config_name,
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes="empty video",
        )

    # Export model only to ONNX
    pil_img = bgr_to_pil(frame_bgr)
    inputs_pt = processor(images=pil_img, return_tensors="pt")
    pixel_values = inputs_pt["pixel_values"]

    try:
        dynamic_shapes = {"pixel_values": None}
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
    except Exception as e:
        print(f"    ✗ ONNX export failed: {e}")
        return BenchmarkResult(
            model_id=model_id,
            config_name=config_name,
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes=f"export_failed: {str(e)[:50]}",
        )

    # Load ONNX model
    try:
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_path, providers=providers)
        onnx_input_name = sess.get_inputs()[0].name
    except Exception as e:
        print(f"    ✗ ONNX Runtime loading failed: {e}")
        return BenchmarkResult(
            model_id=model_id,
            config_name=config_name,
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes=f"runtime_failed: {str(e)[:50]}",
        )

    writer = get_video_writer(cap, out_path)
    timing_begin, timing_end, inference_total = time_block()
    total_execution_start = None
    total_execution_time = 0.0
    num_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frames_limit and num_frames >= frames_limit:
            break

        if num_frames == warmup:
            total_execution_start = time.perf_counter()

        pil_img = bgr_to_pil(frame_bgr)

        if num_frames >= warmup:
            timing_begin()

        # Preprocess in Python
        if use_fast:
            inputs_pt = processor(images=pil_img, return_tensors="pt")
            pixel_values = inputs_pt["pixel_values"].numpy(force=True)
        else:
            inputs_pt = processor(images=pil_img, return_tensors="np")
            pixel_values = inputs_pt["pixel_values"]

        # ONNX model inference
        onnx_outputs = sess.run(None, {onnx_input_name: pixel_values})

        # Convert ONNX outputs back to PyTorch format for postprocessing
        pt_outputs = {"logits": torch.from_numpy(onnx_outputs[0]), "pred_boxes": torch.from_numpy(onnx_outputs[1])}
        pt_outputs_bf = BatchFeature(data=pt_outputs, tensor_type="pt")

        # Postprocess in Python
        results = processor.post_process_object_detection(
            pt_outputs_bf,
            target_sizes=torch.tensor([[frame_bgr.shape[0], frame_bgr.shape[1]]]),
            threshold=threshold,
        )

        if num_frames >= warmup:
            timing_end()

        # Draw detections
        if len(results) > 0:
            res0 = results[0]
            boxes = res0["boxes"].numpy()
            scores = res0["scores"].numpy()
            labels = res0["labels"].numpy()

            # Convert to pipeline format
            results_formatted = []
            for box, score, label_id in zip(boxes, scores, labels):
                label_id_int = int(label_id)
                label_name = id2label.get(label_id_int, str(label_id_int))
                results_formatted.append(
                    {
                        "score": float(score),
                        "label": label_name,
                        "box": {
                            "xmin": float(box[0]),
                            "ymin": float(box[1]),
                            "xmax": float(box[2]),
                            "ymax": float(box[3]),
                        },
                    }
                )

            frame_out = draw_detections(frame_bgr, results_formatted, threshold)
        else:
            frame_out = frame_bgr

        writer.write(frame_out)
        num_frames += 1

    if total_execution_start is not None:
        total_execution_time = time.perf_counter() - total_execution_start

    writer.release()

    proc_frames = max(0, num_frames - warmup)
    fps = (proc_frames / inference_total[0]) if inference_total[0] > 0 and proc_frames > 0 else 0.0

    print(f"    ✓ ONNX model-only: {fps:.2f} FPS (inference) | {num_frames} frames")

    return BenchmarkResult(
        model_id=model_id,
        config_name=config_name,
        fps=fps,
        num_frames=num_frames,
        total_time=total_execution_time,
        inference_time=inference_total[0],
    )


def benchmark_torch_export(
    model_id: str,
    cap: cv2.VideoCapture,
    device: str,
    threshold: float,
    frames_limit: int,
    warmup: int,
    out_path: str,
) -> BenchmarkResult:
    """Benchmark torch.export E2E with dynamic shapes"""
    print("\n  Testing torch.export E2E...")

    pipe = pipeline("object-detection", model=model_id, device=device)
    id2label = pipe.model.config.id2label  # Save id2label for drawing

    # Get first frame for export
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_bgr = cap.read()
    if not ret:
        return BenchmarkResult(
            model_id=model_id,
            config_name="torch_export",
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes="empty video",
        )

    pil_img = bgr_to_pil(frame_bgr)

    # Export with dynamic shapes
    try:
        exported = pipe.export(
            example_image=pil_img,
            format="torch",
            dynamic_shapes=True,
            threshold=threshold,  # Use the same threshold as inference
        )
        exported_module = exported.module()
    except Exception as e:
        print(f"    ✗ Export failed: {e}")
        return BenchmarkResult(
            model_id=model_id,
            config_name="torch_export",
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes=f"export_failed: {str(e)[:50]}",
        )

    # Get exportable module for preprocessing
    exportable = pipe.get_exportable_module()

    writer = get_video_writer(cap, out_path)
    timing_begin, timing_end, inference_total = time_block()
    total_execution_start = None
    total_execution_time = 0.0
    num_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frames_limit and num_frames >= frames_limit:
            break

        if num_frames == warmup:
            total_execution_start = time.perf_counter()

        pil_img = bgr_to_pil(frame_bgr)

        if num_frames >= warmup:
            timing_begin()

        # Prepare inputs
        images = exportable.get_tensors_inputs(pil_img)
        target_sizes = torch.tensor([[frame_bgr.shape[0], frame_bgr.shape[1]]], device=device)
        inputs = {
            "images": images.to(device),
            "post_process_kwargs": {"target_sizes": target_sizes, "threshold": threshold},
        }

        with torch.no_grad():
            outputs = exported_module(**inputs)

        if num_frames >= warmup:
            timing_end()

        # Convert output to pipeline format
        results = []
        if outputs and len(outputs) > 0:
            output = outputs[0]
            for box, score, label_id in zip(output["boxes"], output["scores"], output["labels"]):
                box_cpu = box.cpu().numpy()
                label_id_int = int(label_id.cpu().numpy())
                label_name = id2label.get(label_id_int, str(label_id_int))
                results.append(
                    {
                        "score": float(score.cpu().numpy()),
                        "label": label_name,
                        "box": {
                            "xmin": float(box_cpu[0]),
                            "ymin": float(box_cpu[1]),
                            "xmax": float(box_cpu[2]),
                            "ymax": float(box_cpu[3]),
                        },
                    }
                )

        frame_out = draw_detections(frame_bgr, results, threshold)
        writer.write(frame_out)
        num_frames += 1

    if total_execution_start is not None:
        total_execution_time = time.perf_counter() - total_execution_start

    writer.release()

    proc_frames = max(0, num_frames - warmup)
    fps = (proc_frames / inference_total[0]) if inference_total[0] > 0 and proc_frames > 0 else 0.0

    print(f"    ✓ torch.export E2E: {fps:.2f} FPS (inference) | {num_frames} frames")

    return BenchmarkResult(
        model_id=model_id,
        config_name="torch_export_e2e",
        fps=fps,
        num_frames=num_frames,
        total_time=total_execution_time,
        inference_time=inference_total[0],
    )


def benchmark_onnx_export(
    model_id: str,
    cap: cv2.VideoCapture,
    device: str,
    threshold: float,
    frames_limit: int,
    warmup: int,
    out_path: str,
    onnx_path: str,
) -> BenchmarkResult:
    """Benchmark ONNX E2E export with dynamic shapes"""
    print("\n  Testing ONNX E2E export...")

    pipe = pipeline("object-detection", model=model_id, device="cpu")  # ONNX export on CPU
    id2label = pipe.model.config.id2label  # Save id2label for drawing

    # Get first frame for export
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_bgr = cap.read()
    if not ret:
        return BenchmarkResult(
            model_id=model_id,
            config_name="onnx",
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes="empty video",
        )

    pil_img = bgr_to_pil(frame_bgr)

    # Export to ONNX
    try:
        pipe.export(
            example_image=pil_img,
            format="onnx",
            save_path=onnx_path,
            dynamic_shapes=True,
            threshold=threshold,  # Use the same threshold as inference
        )
    except Exception as e:
        print(f"    ✗ ONNX export failed: {e}")
        return BenchmarkResult(
            model_id=model_id,
            config_name="onnx",
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes=f"export_failed: {str(e)[:50]}",
        )

    # Load ONNX model with ONNX Runtime
    try:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print(f"    ✗ ONNX Runtime loading failed: {e}")
        return BenchmarkResult(
            model_id=model_id,
            config_name="onnx",
            fps=0.0,
            num_frames=0,
            total_time=0.0,
            inference_time=0.0,
            notes=f"runtime_failed: {str(e)[:50]}",
        )

    writer = get_video_writer(cap, out_path)
    timing_begin, timing_end, inference_total = time_block()
    total_execution_start = None
    total_execution_time = 0.0
    num_frames = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Note: ONNX export includes preprocessing, so we just need raw images
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frames_limit and num_frames >= frames_limit:
            break

        if num_frames == warmup:
            total_execution_start = time.perf_counter()

        if num_frames >= warmup:
            timing_begin()

        # Prepare ONNX inputs
        # The E2E exported model expects raw uint8 PIL images (for preprocessing)
        pil_img = bgr_to_pil(frame_bgr)
        image_array = np.array(pil_img).transpose(2, 0, 1)[np.newaxis, ...].astype(np.uint8)
        target_sizes = np.array([[frame_bgr.shape[0], frame_bgr.shape[1]]], dtype=np.int64)

        onnx_inputs = {
            sess.get_inputs()[0].name: image_array,
            sess.get_inputs()[1].name: target_sizes,
        }

        try:
            onnx_outputs = sess.run(None, onnx_inputs)
        except Exception as e:
            print(f"    ✗ ONNX inference failed: {e}")
            break

        if num_frames >= warmup:
            timing_end()

        # Parse ONNX outputs: [scores, labels, boxes]
        if len(onnx_outputs) == 3:
            scores_onnx = onnx_outputs[0]
            labels_onnx = onnx_outputs[1]
            boxes_onnx = onnx_outputs[2]

            # Handle batch dimension if present
            if len(scores_onnx.shape) > 1 and scores_onnx.shape[0] == 1:
                scores_onnx = scores_onnx[0]
                labels_onnx = labels_onnx[0]
                boxes_onnx = boxes_onnx[0]

            # Convert to pipeline format for drawing
            results = []
            for score, label_id, box in zip(scores_onnx, labels_onnx, boxes_onnx):
                label_id_int = int(label_id)
                label_name = id2label.get(label_id_int, str(label_id_int))
                results.append(
                    {
                        "score": float(score),
                        "label": label_name,
                        "box": {
                            "xmin": float(box[0]),
                            "ymin": float(box[1]),
                            "xmax": float(box[2]),
                            "ymax": float(box[3]),
                        },
                    }
                )

            frame_out = draw_detections(frame_bgr, results, threshold)
        else:
            # Unknown format, just write frame without detections
            frame_out = frame_bgr

        writer.write(frame_out)
        num_frames += 1

    if total_execution_start is not None:
        total_execution_time = time.perf_counter() - total_execution_start

    writer.release()

    proc_frames = max(0, num_frames - warmup)
    fps = (proc_frames / inference_total[0]) if inference_total[0] > 0 and proc_frames > 0 else 0.0

    print(f"    ✓ ONNX E2E: {fps:.2f} FPS (inference) | {num_frames} frames")

    return BenchmarkResult(
        model_id=model_id,
        config_name="onnx_e2e",
        fps=fps,
        num_frames=num_frames,
        total_time=total_execution_time,
        inference_time=inference_total[0],
    )


def benchmark_model(
    model_id: str,
    video_path: str,
    device: str,
    threshold: float,
    frames_limit: int,
    warmup: int,
    out_dir: str,
) -> list[BenchmarkResult]:
    """Run all benchmarks for a single model"""
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {model_id}")
    print(f"{'=' * 70}")

    safe_model = model_id.replace("/", "_")
    model_out_dir = os.path.join(out_dir, safe_model)
    ensure_dir(model_out_dir)

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    results = []

    # Normal pipeline
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ✗ Cannot open video: {video_path}")
        return results

    out_path = os.path.join(model_out_dir, f"{base_name}_normal.mp4")
    result = benchmark_normal_pipeline(model_id, cap, device, threshold, frames_limit, warmup, out_path)
    results.append(result)
    cap.release()

    # torch.export model-only (fast processor)
    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join(model_out_dir, f"{base_name}_torch_export_model_only_fast.mp4")
    result = benchmark_torch_export_model_only(
        model_id, cap, device, threshold, frames_limit, warmup, out_path, use_fast=True
    )
    results.append(result)
    cap.release()

    # torch.export model-only (slow processor)
    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join(model_out_dir, f"{base_name}_torch_export_model_only_slow.mp4")
    result = benchmark_torch_export_model_only(
        model_id, cap, device, threshold, frames_limit, warmup, out_path, use_fast=False
    )
    results.append(result)
    cap.release()

    # torch.export E2E
    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join(model_out_dir, f"{base_name}_torch_export_e2e.mp4")
    result = benchmark_torch_export(model_id, cap, device, threshold, frames_limit, warmup, out_path)
    results.append(result)
    cap.release()

    # ONNX model-only (fast processor)
    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join(model_out_dir, f"{base_name}_onnx_model_only_fast.mp4")
    onnx_path = os.path.join(model_out_dir, f"{safe_model}_model_only_fast.onnx")
    result = benchmark_onnx_model_only(
        model_id, cap, device, threshold, frames_limit, warmup, out_path, onnx_path, use_fast=True
    )
    results.append(result)
    cap.release()

    # ONNX model-only (slow processor)
    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join(model_out_dir, f"{base_name}_onnx_model_only_slow.mp4")
    onnx_path = os.path.join(model_out_dir, f"{safe_model}_model_only_slow.onnx")
    result = benchmark_onnx_model_only(
        model_id, cap, device, threshold, frames_limit, warmup, out_path, onnx_path, use_fast=False
    )
    results.append(result)
    cap.release()

    # ONNX E2E export
    cap = cv2.VideoCapture(video_path)
    out_path = os.path.join(model_out_dir, f"{base_name}_onnx_e2e.mp4")
    onnx_path = os.path.join(model_out_dir, f"{safe_model}_e2e.onnx")
    result = benchmark_onnx_export(model_id, cap, device, threshold, frames_limit, warmup, out_path, onnx_path)
    results.append(result)
    cap.release()

    return results


def write_results_csv(out_dir: str, all_results: list[BenchmarkResult]) -> str:
    """Write consolidated results to CSV"""
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "benchmark_summary.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model_id",
                "config",
                "fps_inference",
                "fps_total",
                "num_frames",
                "inference_time_sec",
                "total_time_sec",
                "notes",
            ]
        )

        for r in all_results:
            proc_frames = max(0, r.num_frames - 10)  # Assuming warmup of 10
            total_fps = proc_frames / r.total_time if r.total_time > 0 else 0.0

            writer.writerow(
                [
                    r.model_id,
                    r.config_name,
                    f"{r.fps:.4f}",
                    f"{total_fps:.4f}",
                    r.num_frames,
                    f"{r.inference_time:.4f}",
                    f"{r.total_time:.4f}",
                    r.notes,
                ]
            )

    return csv_path


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return

    ensure_dir(args.out_dir)

    all_results = []

    for model_id in args.models:
        try:
            results = benchmark_model(
                model_id=model_id,
                video_path=args.video,
                device=args.device,
                threshold=args.threshold,
                frames_limit=args.frames_limit,
                warmup=args.warmup,
                out_dir=args.out_dir,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"\n✗ Failed to benchmark {model_id}: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Write summary
    if all_results:
        csv_path = write_results_csv(args.out_dir, all_results)
        print(f"\n{'=' * 70}")
        print("BENCHMARK SUMMARY")
        print(f"{'=' * 70}")
        for r in all_results:
            proc_frames = max(0, r.num_frames - args.warmup)
            total_fps = proc_frames / r.total_time if r.total_time > 0 else 0.0
            status = "✓" if r.fps > 0 else "✗"
            print(
                f"{status} {r.model_id:45s} | {r.config_name:15s} | "
                f"{r.fps:7.2f} FPS (inf) | {total_fps:7.2f} FPS (total) | "
                f"{r.notes if r.notes else 'OK'}"
            )
        print(f"\nResults saved to: {csv_path}")
        print(f"All outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
