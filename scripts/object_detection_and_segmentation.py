import os
import torch
import json
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
from sam3.model.box_ops import box_xywh_to_cxcywh
import matplotlib.pyplot as plt
import argparse
from google import genai
from typing import List, Tuple
import numpy as np


def _extract_json(text: str):
    """Extract JSON string from model response."""
    if not text:
        raise ValueError("Empty response from model.")

    if "```json" in text:
        try:
            return text.split("```json")[1].split("```")[0].strip()
        except IndexError:
            pass

    return text.strip()


def _convert_box_to_xywh(box, img_w, img_h):
    """
    Convert [ymin, xmin, ymax, xmax] (0–1000 normalized)
    to [x, y, w, h] in pixel coordinates.
    """
    if len(box) != 4:
        raise ValueError(f"Invalid box format: {box}")

    ymin, xmin, ymax, xmax = box

    x1 = int(xmin / 1000 * img_w)
    y1 = int(ymin / 1000 * img_h)
    x2 = int(xmax / 1000 * img_w)
    y2 = int(ymax / 1000 * img_h)

    return [x1, y1, x2 - x1, y2 - y1]


def generate_gemini_mask(
    img: Image.Image, client: genai.Client, object_names: List[str]
) -> Tuple[List[List[int]], List[str]]:
    img_w, img_h = img.size

    prompt = f"""
    Segment all objects on the table.
    Exclude robots and the table itself.

    Output a JSON list where each entry contains:
    - "box_2d": [ymin, xmin, ymax, xmax] (normalized 0–1000)
    - "label": object name from: {object_names}
    Ensure correct capitalization.
    """

    print("Sending request to Gemini...")

    # Call model
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[img, prompt],
    )

    # Extract JSON
    try:
        json_str = _extract_json(response.text)
        mask_data = json.loads(json_str)
    except Exception as e:
        raise RuntimeError(f"Failed to parse model output:\n{response.text}") from e

    bounding_boxes = []
    labels = []

    for entry in mask_data:
        label = entry["label"]
        box = entry["box_2d"]

        bbox = _convert_box_to_xywh(box, img_w, img_h)

        bounding_boxes.append(bbox)
        labels.append(label)

    return bounding_boxes, labels


def scan_objects(
    img: Image.Image, masks_folder: str
) -> Tuple[List[str], List[List[int]]]:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    object_library = [
        "A_shape_video",
        "B_shape_video",
        "D_shape_video",
        "E_shape_video",
        "I_shape_video",
        "R_shape_video",
        "S_shape_video",
        "3_shape_video",
        "baby_toy",
        "book",
        "tape",
    ]

    # Load SAM3 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model)
    width, height = img.size
    inference_state = processor.set_image(img)

    # Use Gemini to generate bounding boxes and labels
    box_input_xywh, object_names = generate_gemini_mask(img, client, object_library)
    print(
        f"Identified {len(box_input_xywh)} objects with bounding boxes: {box_input_xywh}"
    )
    print(f"Identified {len(object_names)} objects with names: {object_names}")

    box_input_cxcywh = box_xywh_to_cxcywh(torch.tensor(box_input_xywh).view(-1, 4))
    norm_boxes_cxcywh = normalize_bbox(box_input_cxcywh, width, height).tolist()

    for i, object_name in enumerate(object_names):
        # Create negative and positive boxes for sam3's box prompt
        # Negative boxes are boxes that are not the object of interest
        box_labels = [False] * len(object_names)
        box_labels[i] = True

        processor.reset_all_prompts(inference_state)

        for box, label in zip(norm_boxes_cxcywh, box_labels):
            inference_state = processor.add_geometric_prompt(
                state=inference_state, box=box, label=label
            )

        assert (
            inference_state["masks"].shape[0] == 1
        ), "Only one mask should be generated"

        mask = inference_state["masks"][0][0].detach().cpu().numpy()
        mask_img = Image.fromarray(mask.astype("uint8") * 255, mode="L")
        mask_img.save(f"{masks_folder}/mask_{object_name}.png")

    boxes_xywh = [[int(v) for v in row] for row in box_input_xywh]
    return object_names, boxes_xywh


if __name__ == "__main__":
    img = Image.open("/Users/hienbui/git/sam3/scripts/realsense_capture.jpg")
    folder = "/Users/hienbui/Downloads/"
    names, boxes = scan_objects(img, folder)
    print(names, boxes)
