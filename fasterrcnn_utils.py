import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tarfile
import io
import os

# ===== PASCAL VOC class names =====
VOC_CLASSES = [
    "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# ==================================


def run_fasterrcnn_gradcam(args):
    """
    Run Faster R-CNN with Grad-CAM visualization.
    Supports both PASCAL VOC 2007 dataset and single image input.
    """
    print("Running Faster-RCNN Grad-CAM...")

    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    from fasterrcnn_gradcam import FasterRCNNGradCAM
    from torchvision import transforms
    from PIL import Image

    from utils import create_combined_image
    from fasterrcnn_utils import draw_fasterrcnn_boxes_voc, draw_fasterrcnn_boxes, combine_three_images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Load pretrained Faster-RCNN =====
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    model.eval()

    # Get model's class names (correct 91 categories)
    model_classes = weights.meta["categories"]

    # ===== Init Grad-CAM wrapper =====
    gradcam = FasterRCNNGradCAM(model)

    # ===== Image transform =====
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # ===== Process PASCAL VOC 2007 dataset =====
    if args.det_img == "PASCAL VOC 2007":
        tar_path = r'.\data\VOCtest_06-Nov-2007.tar'
        output_dir = "fasterrcnn_gradcam_voc2007_results_{target_layer}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Processing PASCAL VOC 2007 dataset from: {tar_path}")
        
        with tarfile.open(tar_path, 'r') as tar:
            # Find all JPEG images in the tar file
            image_members = [m for m in tar.getmembers() 
                           if m.name.endswith('.jpg') and 'JPEGImages' in m.name]
            
            # Limit to 200 images
            image_members = image_members[:200]
            total_images = len(image_members)
            
            print(f"Found {total_images} images to process.")
            
            for idx, member in enumerate(image_members):
                # Read image from tar
                f = tar.extractfile(member)
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
                img_tensor = transform(img).to(device)
                
                # Run detection
                with torch.no_grad():
                    outputs = model([img_tensor])[0]

                if len(outputs["boxes"]) == 0:
                    print(f"[{idx+1}/{total_images}] No object detected: {os.path.basename(member.name)}")
                    continue

                boxes = outputs["boxes"]
                labels = outputs["labels"]
                scores = outputs["scores"]

                # Filter detections to keep only VOC-compatible classes
                boxes, scores, voc_class_names = filter_to_voc_classes(
                    boxes, labels, scores, model_classes
                )
                
                if len(boxes) == 0:
                    print(f"[{idx+1}/{total_images}] No VOC-compatible objects: {os.path.basename(member.name)}")
                    continue

                # Generate Grad-CAM
                cam = gradcam.generate(img_tensor)

                if cam is None:
                    print(f"[{idx+1}/{total_images}] Grad-CAM failed: {os.path.basename(member.name)}")
                    continue

                # Prepare original image
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

                # Grad-CAM overlay
                cam_overlay = overlay_gradcam_on_image(img_tensor, cam)

                # Detection visualization with VOC class names
                det_result = draw_fasterrcnn_boxes_voc(
                    img_tensor,
                    boxes,
                    voc_class_names,
                    scores
                )

                # Combine three images
                final_img = combine_three_images(img_np, cam_overlay, det_result)

                # Save result
                save_name = os.path.basename(member.name).replace('.jpg', '_result.png')
                save_path = os.path.join(output_dir, save_name)
                plt.imsave(save_path, final_img)

                print(f"[{idx+1}/{total_images}] Saved: {save_path}")
        
        print(f"PASCAL VOC 2007 processing complete. Results saved to: {output_dir}")
        return
    
    # ===== Process single image =====
    else:
        img = Image.open(args.det_img).convert("RGB")
        img_tensor = transform(img).to(device)

        # Run detection
        with torch.no_grad():
            outputs = model([img_tensor])[0]

        if len(outputs["boxes"]) == 0:
            print("No object detected.")
            return

        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]

        # Generate Grad-CAM
        cam = gradcam.generate(img_tensor)

        if cam is None:
            print("Grad-CAM failed.")
            return

        # Prepare original image
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

        # Grad-CAM overlay
        cam_overlay = overlay_gradcam_on_image(img_tensor, cam)

        # Detection visualization
        # [FIX] Use model_classes instead of COCO_CLASSES
        det_img = draw_fasterrcnn_boxes(
            img_tensor,
            boxes,
            labels,
            scores,
            model_classes  # <-- FIXED: was COCO_CLASSES
        )

        # Combine three images
        final_img = combine_three_images(img_np, cam_overlay, det_img)

        # Save result
        os.makedirs("fasterrcnn_gradcam_results", exist_ok=True)
        save_path = os.path.join("fasterrcnn_gradcam_results", "result.png")
        plt.imsave(save_path, final_img)

        print(f"Faster-RCNN Grad-CAM result saved to: {save_path}")


def map_to_voc_class(class_name):
    """
    Map a class name to its corresponding PASCAL VOC class name.
    Returns None if the class doesn't exist in VOC.
    """
    # Mapping for classes with different names
    name_mapping = {
        "airplane": "aeroplane",
        "motorcycle": "motorbike",
        "couch": "sofa",
        "potted plant": "pottedplant",
        "dining table": "diningtable",
        "tv": "tvmonitor",
    }
    
    # Apply mapping if exists
    mapped_name = name_mapping.get(class_name, class_name)
    
    # Check if the mapped name exists in VOC classes
    voc_classes_lower = [c.lower() for c in VOC_CLASSES]
    if mapped_name.lower() in voc_classes_lower:
        # Return the properly capitalized VOC class name
        idx = voc_classes_lower.index(mapped_name.lower())
        return VOC_CLASSES[idx]
    
    return None


def filter_to_voc_classes(boxes, labels, scores, model_classes):
    """
    Filter detections to keep only classes that exist in PASCAL VOC.
    
    Args:
        boxes: Tensor of bounding boxes
        labels: Tensor of class labels (indices)
        scores: Tensor of confidence scores
        model_classes: List of class names from the model
    
    Returns:
        filtered_boxes: Tensor of filtered bounding boxes
        filtered_scores: Tensor of filtered scores
        voc_class_names: List of VOC class name strings
    """
    filtered_boxes = []
    filtered_scores = []
    voc_class_names = []
    
    for box, label, score in zip(boxes, labels, scores):
        model_class_name = model_classes[int(label)]
        voc_name = map_to_voc_class(model_class_name)
        
        if voc_name is not None:
            filtered_boxes.append(box)
            filtered_scores.append(score)
            voc_class_names.append(voc_name)
    
    if len(filtered_boxes) > 0:
        filtered_boxes = torch.stack(filtered_boxes)
        filtered_scores = torch.stack(filtered_scores)
    else:
        filtered_boxes = torch.tensor([])
        filtered_scores = torch.tensor([])
    
    return filtered_boxes, filtered_scores, voc_class_names


def draw_fasterrcnn_boxes_voc(image, boxes, voc_class_names, scores, score_thresh=0.5):
    """
    Draw bounding boxes with VOC class names on the image.
    
    Args:
        image: torch.Tensor [3, H, W] (0~1)
        boxes: Tensor of bounding boxes
        voc_class_names: List of VOC class name strings
        scores: Tensor of confidence scores
        score_thresh: Minimum score threshold for display
    
    Returns:
        np.ndarray [H, W, 3] with drawn boxes
    """
    # Tensor to numpy image
    img = image.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    W, H = pil_img.size

    for box, cls_name, score in zip(boxes, voc_class_names, scores):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = box.int().tolist()
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

        # Draw label text with VOC class name
        text = f"{cls_name} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Background box for text
        draw.rectangle(
            [x1, y1 - text_h - 6, x1 + text_w + 6, y1],
            fill="darkgreen"
        )
        draw.text((x1 + 3, y1 - text_h - 3), text, fill="white", font=font)

    return np.array(pil_img).astype(np.float32) / 255.0


def draw_fasterrcnn_boxes(image, boxes, labels, scores, class_names, score_thresh=0.5):
    """
    Draw bounding boxes with class names on the image.
    
    Args:
        image: torch.Tensor [3, H, W] (0~1)
        boxes: Tensor of bounding boxes
        labels: Tensor of class label indices
        scores: Tensor of confidence scores
        class_names: List of class names
        score_thresh: Minimum score threshold for display
    
    Returns:
        np.ndarray [H, W, 3] with drawn boxes
    """
    # Tensor to numpy image
    img = image.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    W, H = pil_img.size

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue

        x1, y1, x2, y2 = box.int().tolist()
        x1 = max(0, min(W - 1, x1))
        x2 = max(0, min(W - 1, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(0, min(H - 1, y2))

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        # Draw label text
        cls_name = class_names[int(label)]
        text = f"{cls_name} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Background box for text
        draw.rectangle(
            [x1, y1 - text_h - 6, x1 + text_w + 6, y1],
            fill="black"
        )
        draw.text((x1 + 3, y1 - text_h - 3), text, fill="yellow", font=font)

    return np.array(pil_img).astype(np.float32) / 255.0


def combine_three_images(img1, img2, img3):
    """
    Combine three images horizontally.
    
    Args:
        img1, img2, img3: [H, W, 3] float32 arrays
    
    Returns:
        [H, 3W, 3] combined image
    """
    h = min(img1.shape[0], img2.shape[0], img3.shape[0])
    w = min(img1.shape[1], img2.shape[1], img3.shape[1])

    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    img3 = img3[:h, :w]

    return np.hstack([img1, img2, img3])


def overlay_gradcam_on_image(image, cam, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        image: torch.Tensor [3, H, W] (0~1)
        cam: torch.Tensor [H, W] (0~1)
        alpha: Blending factor for overlay
    
    Returns:
        np.ndarray [H, W, 3] with Grad-CAM overlay
    """
    img = image.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 1)

    # Convert heatmap to RGB using colormap
    heatmap_np = cam.detach().cpu().numpy()
    heatmap_rgb = plt.get_cmap('jet')(heatmap_np)[..., :3]

    # Overlay heatmap on original image
    overlay = (1 - alpha) * img + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    return overlay
