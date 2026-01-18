import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
import os


class FasterRCNNGradCAMApp:
    """
    GUI Application for Faster R-CNN Grad-CAM Visualization
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Faster R-CNN Grad-CAM Visualization")
        self.root.geometry("1400x800")
        self.root.resizable(True, True)
        
        # Variables
        self.image_path = tk.StringVar()
        self.selected_layer = tk.StringVar(value="layer4")
        self.result_image = None
        
        # Initialize model (lazy loading)
        self.model = None
        self.gradcam = None
        self.device = None
        self.model_classes = None
        
        # Build UI
        self._create_widgets()
        
    def _create_widgets(self):
        """Create all UI widgets"""
        
        # ===== Top Frame: Controls =====
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X, side=tk.TOP)
        
        # Image selection
        ttk.Label(control_frame, text="Image:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        image_entry = ttk.Entry(control_frame, textvariable=self.image_path, width=60)
        image_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        browse_btn = ttk.Button(control_frame, text="Browse...", command=self._browse_image)
        browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Layer selection
        ttk.Label(control_frame, text="Hook Layer:").grid(row=0, column=3, padx=20, pady=5, sticky=tk.W)
        
        layer_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.selected_layer,
            values=["layer2", "layer3", "layer4"],
            state="readonly",
            width=10
        )
        layer_combo.grid(row=0, column=4, padx=5, pady=5)
        layer_combo.set("layer4")
        
        # Run button
        run_btn = ttk.Button(
            control_frame, 
            text="Run Grad-CAM", 
            command=self._run_gradcam,
            style="Accent.TButton"
        )
        run_btn.grid(row=0, column=5, padx=20, pady=5)
        
        # Save button
        save_btn = ttk.Button(
            control_frame, 
            text="Save Result", 
            command=self._save_result
        )
        save_btn.grid(row=0, column=6, padx=5, pady=5)
        
        # ===== Status Bar =====
        self.status_var = tk.StringVar(value="Ready. Please select an image.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
        
        # ===== Progress Bar =====
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # ===== Main Frame: Image Display =====
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas with scrollbars for large images
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            bg='gray20',
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)
        
        # Label for instructions
        self.instruction_label = ttk.Label(
            main_frame,
            text="Result will be displayed here\n(Original Image | Grad-CAM Heatmap | Detection Result)",
            font=("Arial", 12),
            foreground="gray"
        )
        self.instruction_label.pack(pady=20)
        
    def _browse_image(self):
        """Open file dialog to select an image"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=filetypes
        )
        
        if filepath:
            self.image_path.set(filepath)
            self.status_var.set(f"Selected: {os.path.basename(filepath)}")
            
    def _load_model(self):
        """Load the Faster R-CNN model (lazy loading)"""
        if self.model is not None:
            return
            
        self.status_var.set("Loading Faster R-CNN model...")
        self.root.update()
        
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()
        
        self.model_classes = weights.meta["categories"]
        
        self.status_var.set("Model loaded successfully.")
        
    def _run_gradcam(self):
        """Run Grad-CAM on the selected image"""
        
        # Validate input
        image_path = self.image_path.get()
        if not image_path:
            messagebox.showwarning("Warning", "Please select an image first.")
            return
            
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Image file not found: {image_path}")
            return
        
        # Start progress
        self.progress.start()
        self.status_var.set("Processing...")
        self.root.update()
        
        try:
            # Load model if not loaded
            self._load_model()
            
            # Get selected layer
            layer_name = self.selected_layer.get()
            
            # Initialize Grad-CAM with selected layer
            from fasterrcnn_gradcam import FasterRCNNGradCAM
            self.gradcam = FasterRCNNGradCAM(self.model, target_layer=layer_name)
            
            # Run processing
            result_img = self._process_image(image_path, layer_name)
            
            if result_img is not None:
                self._display_result(result_img)
                self.status_var.set(f"Completed! Layer: {layer_name}")
            else:
                self.status_var.set("No objects detected or Grad-CAM failed.")
                
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            
        finally:
            self.progress.stop()
            
    def _process_image(self, image_path, layer_name):
        """Process the image and return the combined result"""
        from torchvision import transforms
        from fasterrcnn_utils import (
            overlay_gradcam_on_image,
            draw_fasterrcnn_boxes,
            combine_three_images
        )
        
        # Load and transform image
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).to(self.device)
        
        # Run detection
        with torch.no_grad():
            outputs = self.model([img_tensor])[0]
            
        if len(outputs["boxes"]) == 0:
            return None
            
        boxes = outputs["boxes"]
        labels = outputs["labels"]
        scores = outputs["scores"]
        
        # Generate Grad-CAM
        cam = self.gradcam.generate(img_tensor)
        
        if cam is None:
            return None
            
        # Prepare original image
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Grad-CAM overlay
        cam_overlay = overlay_gradcam_on_image(img_tensor, cam)
        
        # Detection visualization
        det_img = draw_fasterrcnn_boxes(
            img_tensor,
            boxes,
            labels,
            scores,
            self.model_classes
        )
        
        # Combine three images
        final_img = combine_three_images(img_np, cam_overlay, det_img)
        
        # Store for saving
        self.result_image = final_img
        
        return final_img
        
    def _display_result(self, result_img):
        """Display the result image on the canvas"""
        
        # Hide instruction label
        self.instruction_label.pack_forget()
        
        # Convert numpy array to PIL Image
        result_img_uint8 = (np.clip(result_img, 0, 1) * 255).astype(np.uint8)
        pil_img = Image.fromarray(result_img_uint8)
        
        # Resize if too large for display
        max_width = self.canvas.winfo_width() - 20
        max_height = self.canvas.winfo_height() - 20
        
        if max_width < 100:
            max_width = 1200
        if max_height < 100:
            max_height = 600
            
        # Calculate scaling
        img_width, img_height = pil_img.size
        scale = min(max_width / img_width, max_height / img_height, 1.0)
        
        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            pil_img_display = pil_img.resize((new_width, new_height), Image.LANCZOS)
        else:
            pil_img_display = pil_img
            
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(pil_img_display)
        
        # Clear canvas and display
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width() // 2,
            self.canvas.winfo_height() // 2,
            anchor=tk.CENTER,
            image=self.photo
        )
        
        # Update scroll region
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
    def _save_result(self):
        """Save the result image to file"""
        
        if self.result_image is None:
            messagebox.showwarning("Warning", "No result to save. Run Grad-CAM first.")
            return
            
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.asksaveasfilename(
            title="Save Result Image",
            defaultextension=".png",
            filetypes=filetypes
        )
        
        if filepath:
            import matplotlib.pyplot as plt
            plt.imsave(filepath, self.result_image)
            self.status_var.set(f"Result saved to: {filepath}")
            messagebox.showinfo("Success", f"Result saved to:\n{filepath}")


def run_gui():
    """Launch the GUI application"""
    root = tk.Tk()
    
    # Style configuration
    style = ttk.Style()
    style.theme_use('clam')  # Use 'clam', 'alt', 'default', or 'classic'
    
    app = FasterRCNNGradCAMApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()