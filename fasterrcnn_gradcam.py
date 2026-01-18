# fasterrcnn_gradcam.py
import torch
import torch.nn.functional as F


class FasterRCNNGradCAM:
    """
    Grad-CAM for torchvision Faster R-CNN
    Target layer: model.backbone.body.layer4
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

        self.feature_maps = None
        self.gradients = None

        # ===== [ADDED] Register hooks =====
        target_layer = self.model.backbone.body.layer4

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
        # =================================

    # ===== [ADDED] =====
    def _forward_hook(self, module, input, output):
        self.feature_maps = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    # ===================

    def generate(self, image, target_class=None):
        """
        image: Tensor [3, H, W]  (0~1)
        return: heatmap Tensor [H, W] (0~1)
        """

        device = next(self.model.parameters()).device
        image = image.to(device)

        # Detection model expects list input
        outputs = self.model([image])

        scores = outputs[0]["scores"]
        labels = outputs[0]["labels"]

        if len(scores) == 0:
            return None

        # Select target detection
        if target_class is None:
            idx = scores.argmax()
        else:
            match = (labels == target_class).nonzero()
            if len(match) == 0:
                idx = scores.argmax()
            else:
                idx = match[0].item()

        score = scores[idx]

        # Backward target score
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # ===== Grad-CAM computation =====
        gradients = self.gradients[0]      # [C, H, W]
        feature_maps = self.feature_maps[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(feature_maps.shape[1:], device=device)

        for k, w in enumerate(weights):
            cam += w * feature_maps[k]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        # Upsample to image size (no cv2)
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(image.shape[1], image.shape[2]),
            mode="bilinear",
            align_corners=False
        ).squeeze()

        return cam.detach().cpu()
