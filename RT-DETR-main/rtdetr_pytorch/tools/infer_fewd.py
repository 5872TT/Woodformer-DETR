"""Run FEWD-DETR on one image or every supported image in a folder."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core import YAMLConfig


DEFAULT_CONFIG = os.path.join(ROOT, "configs", "rtdetr", "fewd_rtdetr_r18_ouludefect.yml")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_weights(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "ema" in checkpoint:
        ema = checkpoint["ema"]
        if isinstance(ema, dict) and "module" in ema:
            return ema["module"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def preprocess(image, side):
    resampling = getattr(Image, "Resampling", Image)
    resized = image.resize((side, side), resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.unsqueeze(0)


def draw_result(image, result, class_names, threshold):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", max(14, image.width // 80))
    except OSError:
        font = ImageFont.load_default()
    labels = result["labels"].detach().cpu().tolist()
    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    for label, box, score in zip(labels, boxes, scores):
        if score < threshold:
            continue
        x1, y1, x2, y2 = box
        # The standard postprocessor maps labels back to COCO category IDs (1-based)
        # when remap_mscoco_category is enabled in the reference configuration.
        class_index = label - 1 if 1 <= label <= len(class_names) else label
        name = class_names[class_index] if 0 <= class_index < len(class_names) else str(label)
        text = f"{name} {score:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(40, 230, 90), width=max(2, image.width // 500))
        bounds = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle(bounds, fill=(30, 170, 70))
        draw.text((x1, y1), text, fill="white", font=font)
    return image


def image_files(source):
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"source image or directory does not exist: {source}")


def main():
    parser = argparse.ArgumentParser(description="FEWD-DETR image inference")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="FEWD-DETR YAML configuration")
    parser.add_argument("-r", "--resume", required=True, help="trained FEWD-DETR checkpoint")
    parser.add_argument("-f", "--source", required=True, help="image file or directory")
    parser.add_argument("-o", "--output", default="fewd_results", help="output image or output directory")
    parser.add_argument("-d", "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence for drawing boxes")
    parser.add_argument("--classes", default="dry_knot,sound_knot,edge_knot",
                        help="comma-separated class names in model label order")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        parser.error(f"configuration not found: {args.config}")
    if not os.path.isfile(args.resume):
        parser.error(f"checkpoint not found: {args.resume}")
    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")

    device = torch.device(args.device)
    cfg = YAMLConfig(args.config)
    cfg.model.load_state_dict(load_weights(args.resume), strict=True)
    model = cfg.model.to(device).eval().deploy()
    postprocessor = cfg.postprocessor.to(device).eval()
    side = int(cfg.yaml_cfg["RTDETRTransformer"].get("eval_spatial_size", [640, 640])[0])
    names = [part.strip() for part in args.classes.split(",") if part.strip()]
    paths = image_files(args.source)
    if not paths:
        parser.error(f"no supported images found under: {args.source}")

    output = Path(args.output)
    single_input = len(paths) == 1 and Path(args.source).is_file()
    if single_input:
        output.parent.mkdir(parents=True, exist_ok=True)
        destinations = [output]
    else:
        output.mkdir(parents=True, exist_ok=True)
        destinations = [output / path.name for path in paths]

    for path, destination in zip(paths, destinations):
        image = Image.open(path).convert("RGB")
        original_size = torch.tensor([[image.width, image.height]], device=device)
        inputs = preprocess(image, side).to(device)
        with torch.inference_mode():
            predictions = model(inputs)
            result = postprocessor(predictions, original_size)[0]
        rendered = draw_result(image.copy(), result, names, args.conf)
        rendered.save(destination)
        print(f"{path} -> {destination}")


if __name__ == "__main__":
    main()
