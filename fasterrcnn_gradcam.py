import torch
import torch.nn.functional as F


class FasterRCNNGradCAM:
    """
    Grad-CAM for torchvision Faster R-CNN
    Target layer: model.backbone.body.layer2/layer3/layer4 (selectable)
    """

    def __init__(self, model, target_layer="layer4"):
        """
        Initialize Grad-CAM wrapper.
        
        Args:
            model: Faster R-CNN model
            target_layer: Which layer to hook ("layer2", "layer3", or "layer4")
        """
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer

        self.feature_maps = None
        self.gradients = None
        
        # Store hook handles for cleanup
        self.forward_hook_handle = None
        self.backward_hook_handle = None

        # Register hooks on the selected layer
        self._register_hooks(target_layer)

    def _get_target_layer(self, layer_name):
        """Get the target layer module based on layer name"""
        if layer_name == "layer2":
            return self.model.backbone.body.layer2
        elif layer_name == "layer3":
            return self.model.backbone.body.layer3
        elif layer_name == "layer4":
            return self.model.backbone.body.layer4
        else:
            raise ValueError(f"Invalid layer name: {layer_name}. Choose from 'layer2', 'layer3', 'layer4'")

    def _register_hooks(self, layer_name):
        """Register forward and backward hooks on the target layer"""
        # Remove existing hooks if any
        self._remove_hooks()
        
        target_layer = self._get_target_layer(layer_name)
        
        self.forward_hook_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook_handle = target_layer.register_full_backward_hook(self._backward_hook)
        
    def _remove_hooks(self):
        """Remove existing hooks"""
        if self.forward_hook_handle is not None:
            self.forward_hook_handle.remove()
            self.forward_hook_handle = None
        if self.backward_hook_handle is not None:
            self.backward_hook_handle.remove()
            self.backward_hook_handle = None

    def _forward_hook(self, module, input, output):
        self.feature_maps = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def set_target_layer(self, layer_name):
        """
        Change the target layer for Grad-CAM.
        
        Args:
            layer_name: "layer2", "layer3", or "layer4"
        """
        self.target_layer_name = layer_name
        self._register_hooks(layer_name)

    def generate(self, image, target_class=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: Tensor [3, H, W] (0~1)
            target_class: Target class index (optional, uses highest score if None)
            
        Returns:
            heatmap: Tensor [H, W] (0~1)
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

        # Grad-CAM computation
        if self.gradients is None or self.feature_maps is None:
            return None
            
        gradients = self.gradients[0]      # [C, H, W]
        feature_maps = self.feature_maps[0]

        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(feature_maps.shape[1:], device=device)

        for k, w in enumerate(weights):
            cam += w * feature_maps[k]

        cam = F.relu(cam)
        cam -= cam.min()
        cam /= cam.max() + 1e-8

        # Upsample to image size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(image.shape[1], image.shape[2]),
            mode="bilinear",
            align_corners=False
        ).squeeze()

        return cam.detach().cpu()
    
    def __del__(self):
        """Cleanup hooks when object is destroyed"""
        self._remove_hooks()