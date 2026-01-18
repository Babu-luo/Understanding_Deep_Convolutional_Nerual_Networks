import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def get_cam_CIFAR10(model, image, class_idx=None):
    """
    Compute Class Activation Map (CAM) for a given input image.
    
    :param model(nn.Module): A CAM-compatible model (e.g., ResNet with `feature_maps` attribute).
    :param image(torch.Tensor): A single input image, shape [C, H, W](e.g., [3, 32, 32]).
    :param class_idx(int, optional): Target class index. If None, uses the predicted class. 

    returns:
        cam (torch.Tensor): Normalized CAM heatmap, shape [H, W], values in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        output = model(image.unsqueeze(0))

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        feature_maps = model.feature_maps[0]
        weights = model.fc.weight[class_idx]

        cam = torch.zeros(feature_maps.shape[1:], dtype=torch.float32, device=feature_maps.device)
        for k, w in enumerate(weights):
            cam += w * feature_maps[k]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(32, 32),
            mode='bilinear',
            align_corners=False
        ).squeeze()
        return cam
    
def get_gradcam_CIFAR10(model, image, class_idx=None):
    """
    Compute Grad-CAM for a given input image (CIFAR-10).

    :param model(nn.Module): A CNN model with attributes:
        - feature_maps: feature maps from the last convolution layer
        - gradients: gradients w.r.t. the feature maps
    :param image(torch.Tensor): A single input image, shape [C, H, W]
    :param class_idx(int, optional): Target class index. If None, uses the predicted class.

    returns:
        grad_cam (torch.Tensor): Normalized Grad-CAM heatmap, shape [H, W], values in [0, 1].
    """

    model.eval()

    # Grad-CAM requires gradient computation
    output = model(image.unsqueeze(0))

    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    # Clear existing gradients
    model.zero_grad()

    # Backpropagate the score of the target class
    score = output[0, class_idx]
    score.backward(retain_graph=True)

    # Retrieve gradients and feature maps
    gradients = model.gradients[0]        # [C, H, W]
    feature_maps = model.feature_maps[0]  # [C, H, W]

    # Compute channel-wise weights by global average pooling
    weights = gradients.mean(dim=(1, 2))  # [C]

    # Weighted combination of feature maps
    grad_cam = torch.zeros(
        feature_maps.shape[1:],
        dtype=torch.float32,
        device=feature_maps.device
    )

    for k, w in enumerate(weights):
        grad_cam += w * feature_maps[k]

    # Apply ReLU to keep only positive contributions
    grad_cam = F.relu(grad_cam)

    # Normalize to [0, 1]
    grad_cam = grad_cam - grad_cam.min()
    grad_cam = grad_cam / (grad_cam.max() + 1e-8)

    # ===== [MODIFICATION] =====
    # Upsample Grad-CAM to input size (32x32 for CIFAR-10)
    grad_cam = F.interpolate(
        grad_cam.unsqueeze(0).unsqueeze(0),  # [1, 1, H, W]
        size=(32, 32),
        mode='bilinear',
        align_corners=False
    ).squeeze()  # [32, 32]
    # ==========================

    return grad_cam

def create_combined_image(original_img, heatmap, mean=[0.4914, 0.4822, 0.4465],
                        std=[0.2470, 0.2435, 0.2616], alpha=0.4):
    """
    Creates a combined image showing original image and heatmap side-by-side.
    
    :param original_img (torch.Tensor): Normalized image tensor [C, H, W]
    :param heatmap (torch.Tensor): Heatmap tensor [H, W]
    :param mean: Normalization mean used
    :param std: Normalization std used
    :param alpha: Heatmap transparency (0~1)
        
    Returns:
        np.ndarray: Combined image [H, 2*W, 3] in RGB format
    """
    # Denormalize image
    mean_tensor = torch.tensor(mean).view(3, 1, 1).to(original_img.device)
    std_tensor = torch.tensor(std).view(3, 1, 1).to(original_img.device)
    img_denorm = original_img * std_tensor + mean_tensor
    img_denorm = img_denorm.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    img_denorm = np.clip(img_denorm, 0, 1)
    
    # Convert heatmap to RGB using colormap
    heatmap_np = heatmap.detach().cpu().numpy()
    heatmap_rgb = plt.get_cmap('jet')(heatmap_np)[..., :3]  # [H, W, 3]

    # Normalize heatmap to [0,1]
    heatmap_rgb = heatmap_rgb[..., :3]
    
    # Overlay heatmap on original image (transparent)
    overlay = (1 - alpha) * img_denorm + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)

    # Concatenate: original | overlay
    combined = np.hstack((img_denorm, overlay))
    return combined



def compute_deletion_curve(model, image, label, heatmap, num_steps=100, hidden_strategy='gray'):
    """
    Compute the deletion metric for a given image
    
    :param model: trained model 
    :param image: single image tensor, shape: [C, H, W]
    :param label: gt label
    :param num_steps: number of deletion steps
    :param use_gradcam: whether to use Grad-CAM
    """
    model.eval()
    device = image.device
    
    h, w = heatmap.shape
    flat_heatmap = heatmap.view(-1)
    sorted_indices = torch.argsort(flat_heatmap, descending=True)
    total_pixels = h * w
    pixels_per_step = (total_pixels + num_steps - 1) // num_steps

    current_image = image.clone()
    scores = []

    # Initial score (0% masked)
    with torch.no_grad():
        init_output = model(current_image.unsqueeze(0))
        init_prob = F.softmax(init_output, dim=1)
        init_score = init_prob[0, label].item()
        scores.append(init_score)

    # Progressive deletion over steps
    for step in range(num_steps):
        start = step * pixels_per_step
        if start >= total_pixels:
            break
        end = min(start + pixels_per_step, total_pixels)
        indices_to_mask = sorted_indices[start:end]

        # ===== [MODIFICATION] =====
        # Handle hidden strategy
        if hidden_strategy == 'gray':
            hidden_val = 0.5
        elif hidden_strategy == 'mean':
            hidden_val = current_image.view(3, -1).mean(dim=1).unsqueeze(1)
        else:
            hidden_val = 0.0
        # ==========================
        current_image.view(3, -1)[:, indices_to_mask] = hidden_val

        # Compute score AFTER masking all pixels in this step
        with torch.no_grad():
            output = model(current_image.unsqueeze(0))
            probs = F.softmax(output, dim=1)
            score = probs[0, label].item()
            scores.append(score)

    return scores

def compute_insertion_curve(model, image, label, heatmap, num_steps=100, hidden_strategy='gray'):
    model.eval()
    device = image.device

    h, w = heatmap.shape
    flat_heatmap = heatmap.view(-1)
    sorted_indices = torch.argsort(flat_heatmap, descending=True)
    total_pixels = h * w
    pixels_per_step = (total_pixels + num_steps - 1) // num_steps

    if hidden_strategy == 'gray':
        current_image = torch.full_like(image, 0.5)
    elif hidden_strategy == 'zero':
        current_image = torch.zeros_like(image)
    else:
        mean_val = image.view(3, -1).mean(dim=1).view(3, 1, 1)
        current_image = mean_val.expand_as(image).clone()
    scores = []

    with torch.no_grad():
        init_output = model(current_image.unsqueeze(0))
        scores.append(F.softmax(init_output, dim=1)[0, label].item())

    image_flat = image.view(3, -1)
    current_image_flat = current_image.view(3, -1)
    for step in range(num_steps):
        start = step * pixels_per_step
        if start >= total_pixels:
            break
        end = min(start + pixels_per_step, total_pixels)
        indices_to_unmask = sorted_indices[start:end]
        current_image_flat[:, indices_to_unmask] = image_flat[:, indices_to_unmask]

        with torch.no_grad():
            output = model(current_image.unsqueeze(0))
            score = F.softmax(output, dim=1)[0, label].item()
            scores.append(score)

    return scores


# ===== [ADDED FOR FASTER-RCNN GRAD-CAM FUNCTION] =====
def run_fasterrcnn_gradcam(args):
    print("Running Faster-RCNN Grad-CAM...")

    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from fasterrcnn_gradcam import FasterRCNNGradCAM
    from torchvision import transforms
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained Faster-RCNN
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights).to(device)
    # model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    gradcam = FasterRCNNGradCAM(model)

    # Load image
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img = Image.open(args.det_img).convert("RGB")
    img_tensor = transform(img)

    cam = gradcam.generate(img_tensor)

    if cam is None:
        print("No object detected.")
        return

    # Visualization (no cv2)
    img_np = np.array(img) / 255.0
    cam_np = cam.numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img_np)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Faster-RCNN Grad-CAM")
    plt.imshow(img_np)
    plt.imshow(cam_np, cmap='jet', alpha=0.4)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("fasterrcnn_gradcam.png", dpi=300)
    print("Saved fasterrcnn_gradcam.png")
# ====================================================
