import argparse, os
from utils.image_utils import load_image_cv2, numpy_to_pil_rgb
from utils.detection_utils import run_yolo_inference, generate_heatmap_from_boxes, overlay_heatmap_on_image

def infer(weights, source, save=True, out_dir="outputs"):
    img = load_image_cv2(source)

    # Run YOLOv12 inference
    boxes, annotated = run_yolo_inference(weights, img)

    # Generate heatmap
    heatmap = generate_heatmap_from_boxes(img, boxes)
    overlay = overlay_heatmap_on_image(img, heatmap)

    # Save results
    if save:
        os.makedirs(out_dir, exist_ok=True)
        numpy_to_pil_rgb(annotated).save(os.path.join(out_dir, "annotated.png"))
        numpy_to_pil_rgb(overlay).save(os.path.join(out_dir, "heatmap_overlay.png"))
        print(f"✅ Results saved in {out_dir}/")

    print("🔍 Detections:", boxes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    infer(args.weights, args.source)
