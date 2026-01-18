# Final_Project Understanding Deep Convolutional Neural Networks

We choose ResNet as our basic model, adding a GAP layer and a fully connected layer to modify it as CAM

## main.py

~~~py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import VGG, ResNet, ResNext
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import get_cam_CIFER10 as get_cam
from utils import get_gradcam_CIFER10 as get_gradcam
from utils import compute_deletion_curve, create_combined_image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# # ===== [MODIFICATION] =====
# # CAM computation function (ResNet only)
# def compute_cam(model, class_idx):
#     """
#     Compute CAM for a given class index.
#     """
#     # feature_maps: [B, C, H, W]
#     feature_maps = model.feature_maps[0]      # [C, H, W]
#     weights = model.fc.weight[class_idx]      # [C]

#     cam = torch.zeros(feature_maps.shape[1:], dtype=torch.float32)

#     for k, w in enumerate(weights):
#         cam += w * feature_maps[k]

#     cam = F.relu(cam)

#     # normalize to [0, 1]
#     cam -= cam.min()
#     cam /= cam.max() + 1e-8

#     return cam
# # ==========================

def train(model, args):
    '''
    Model training function
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    # create dataset, data augmentation
    device = torch.device("cpu")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                saturation=0.2, hue=0.02),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616])
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=test_transform)

    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # create optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    # create scheduler 
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )
    best_acc = 0.0

    # ctreat summary writer
    writer = SummaryWriter(log_dir=args.log_dir)
    #writer = SummaryWriter()

    # train
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0

        for step, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # optimize
            optimizer.step()
            running_loss += loss.item()
            # write summary
            if step % 100 == 0:
                writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + step)
                writer.flush()

        # scheduler adjusts learning rate
        scheduler.step()

        # test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        writer.add_scalar('test/accuracy', acc, epoch)
        writer.flush()
        print(f"Epoch [{epoch+1}/{args.num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Test Acc: {acc:.2f}%")

    # save checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'acc': best_acc,
            }, args.save_path)
            print(f"Saved model with acc = {best_acc:.2f}%")

    writer.flush()
    writer.close()
    print("Training finished, best test accuracy: {:.2f}%".format(best_acc))

def test(model, args):
    '''
    input: 
        model: linear classifier or full-connected neural network classifier
        args: configuration
    '''
    device = torch.device("cpu")
    # load checkpoint (Tutorial: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()        
    # create testing dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                            std=[0.2470, 0.2435, 0.2616])
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=test_transform)
    # create dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    # test
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            # forward
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    acc = 100.0 * correct / total
    print("Test Accuracy: {:.2f}%".format(acc))






    # ===== [ADDED FOR CAM AND GRAD-CAM] =====
    os.makedirs("cam_results", exist_ok=True)
    cam_saved = 0
    max_cam_images = args.vis_num   # save only first vis_num CAMs

    os.makedirs("gradcam_results", exist_ok=True)
    gradcam_saved = 0
    max_gradcam_images = args.vis_num
    # ==========================

    # ===== [ADDED FOR CAM AND GRDA-CAM] =====
    if args.vis != 'none':
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                img = inputs[i]
                pred = predicted[i].item()
                gt = labels[i].item()

                if args.vis == 'cam' and cam_saved < max_cam_images:
                    cam = get_cam(model, img, pred)
                    combined_img = create_combined_image(img, cam)
                    plt.imsave(
                        f"cam_results/cam_{cam_saved}_pred_{pred}_gt_{gt}.png",
                        # cam.cpu().numpy(),
                        # cmap='jet'
                        combined_img
                    )
                    cam_saved += 1

                if args.vis == 'gradcam' and gradcam_saved < max_gradcam_images:
                    with torch.enable_grad():
                        gradcam = get_gradcam(model, img, pred)
                    combined_img = create_combined_image(img, gradcam)
                    plt.imsave(
                        f"gradcam_results/gradcam_{gradcam_saved}_pred_{pred}_gt_{gt}.png",
                        # gradcam.detach().cpu().numpy(),
                        # cmap='jet'
                        combined_img
                    )
                    gradcam_saved += 1

                if cam_saved >= max_cam_images or gradcam_saved >= max_gradcam_images:
                    return acc
    # ========================================





    # # ===== [DELETION VISUALIZATION] =====
    # deletion_curves = []
    # num_deletion_samples = 5  # 只评估前5张图，避免太慢
    # count = 0

    # with torch.no_grad():
    #     for data in test_loader:
    #         if count >= num_deletion_samples:
    #             break
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         outputs = model(inputs)
    #         _, predicted = torch.max(outputs, 1)

    #         for i in range(inputs.size(0)):
    #             if count >= num_deletion_samples:
    #                 break
    #             if predicted[i].item() != labels[i].item():
    #                 continue

    #             image = inputs[i]
    #             label = labels[i].item()
    #             _ = model(image.unsqueeze(0))

    #             try:
    #                 scores, auc = deletion_metric(model, image, label, num_steps=20, use_gradcam=False)
    #                 deletion_curves.append(scores)
    #                 print(f"Sample {count}: Deletion Curve = {scores}")
    #                 count += 1
    #             except Exception as e:
    #                 print(f"Error on sample {count}: {e}")
    #                 continue

    # if deletion_curves:
    #     avg_scores = np.mean(deletion_curves, axis=0)
    #     steps = np.linspace(0, 1, len(avg_scores))

    #     plt.figure(figsize=(6,4))
    #     plt.plot(steps, avg_scores, 'b-', linewidth=2, label='CAM Deletion Curve')
    #     plt.fill_between(steps, np.min(deletion_curves, axis=0), np.max(deletion_curves, axis=0), color='b',alpha=0.2)
    #     plt.xlabel('Fraction of Pixels Deleted')
    #     plt.ylabel('Target class probability')
    #     plt.title('Deletion Metric (CAM)')
    #     plt.legend()
    #     plt.grid(True, linestyle='--', alpha=0.6)
    #     plt.savefig('deletion_curve_cam.png', dpi=300, bbox_inches='tight')
    #     plt.show()
    # # ==========================
    # ===== [DELETION VISUALIZATION] =====
    deletion_curves = []
    num_eval_samples = 5  # 可视化前5个正确预测的样本
    eval_count = 0

    with torch.no_grad():
        for data in test_loader:
            if eval_count >= num_eval_samples:
                break
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                if eval_count >= num_eval_samples:
                    break
                if predicted[i] != labels[i]:  # 只分析正确预测
                    continue

                image = inputs[i]
                target_class = labels[i].item()

                # 必须先做一次前向传播，以设置 model.feature_maps
                _ = model(image.unsqueeze(0))

                try:
                    scores, auc = compute_deletion_curve(model, image, target_class, num_steps=20)
                    deletion_curves.append(scores)
                    eval_count += 1
                    print(f"Sample {eval_count}: Deletion AUC = {auc:.4f}")
                except Exception as e:
                    print(f"Error on sample {eval_count}: {e}")

    # 平均曲线
    if deletion_curves:
        avg_scores = np.mean(deletion_curves, axis=0)
        steps = np.linspace(0, 1, len(avg_scores))

        plt.figure(figsize=(6, 4))
        plt.plot(steps, avg_scores, 'b-', linewidth=2, label='CAM')
        plt.fill_between(steps, np.min(deletion_curves, axis=0), np.max(deletion_curves, axis=0), color='b', alpha=0.2)
        plt.xlabel('Fraction of pixels masked')
        plt.ylabel('Target class probability')
        plt.title('Deletion Metric (CAM)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig('deletion_curve_cam.png', dpi=300, bbox_inches='tight')
        plt.show()
    # ==========================
    
    return acc














if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--save_path', type=str, default='./checkpoint.pth')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
    # ===== [ADDED FOR CAM / GRAD-CAM CONTROL] =====
    parser.add_argument(
        '--vis',
        type=str,
        choices=['none', 'cam', 'gradcam'],
        default='none',
        help='Visualization type: none | cam | gradcam'
    )
    parser.add_argument(
        '--vis_num',
        type=int,
        default=20,
        help='Number of visualization heatmaps to save'
    )
    # =================================================

    args = parser.parse_args()

    model = ResNet()
    # train / test
    if args.mode == 'train':
        train(model, args)
    else:
        test(model, args)




# python main.py --run=train --model=cnn --optimizer=sgd --scheduler=cosine
# python main.py --run=train --model=fcnn --optimizer=sgd --scheduler=step
# python -m tensorboard.main --logdir=./logs_for_VGG
# python main.py --mode train --batch_size 64 --lr 0.01 --num_epochs 25 --save_path ./checkpoints/vgg_best.pth --log_dir ./logs
# python main.py --mode test --batch_size 64 --lr 0.005 --num_epochs 20 --save_path ./checkpoints/vgg_best_new.pth --log_dir ./logs_for_VGG
# python main.py --mode train --batch_size 64 --lr 0.005 --num_epochs 20 --save_path ./checkpoints/resnet_best.pth --log_dir ./logs_for_ResNet
# python main.py --mode train --batch_size 64 --lr 0.005 --num_epochs 20 --save_path ./checkpoints/resnext_best.pth --log_dir ./logs_for_ResNext
# python main.py --mode test --batch_size 64 --lr 0.005 --num_epochs 20 --save_path ./checkpoints/resnet_best.pth --vis cam --vis_num 10
~~~

## utils.py

~~~py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def get_cam_CIFER10(model, image, class_idx=None):
    """
    Compute Class Activeation Map (CAM) for a given input image.
    
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
    
def get_gradcam_CIFER10(model, image, class_idx=None):
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

def create_combined_image(original_img, heatmap, mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]):
    """
    Creates a combined image showing original image and heatmap side-by-side.
    
    :param original_img (torch.Tensor): Normalized image tensor [C, H, W]
    :param heatmap (torch.Tensor): Heatmap tensor [H, W]
    :param mean: Normalization mean used
    :param std: Normalization std used
        
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
    heatmap_rgb = plt.get_cmap('jet')(heatmap_np)  # [H, W, 4]
    heatmap_rgb = heatmap_rgb[..., :3]  # Remove alpha channel -> [H, W, 3]
    
    # Concatenate horizontally [H, W, 3] + [H, W, 3] -> [H, 2*W, 3]
    combined = np.hstack((img_denorm, heatmap_rgb))
    return combined

# def deletion_metric(model, image, label, num_steps=10, use_gradcam=False):
#     """
#     Compute the deletion metric for a given image
    
#     :param model: trained model
#     :param image: singele image tensor, shape: [C, H, W]
#     :param label: gt label
#     :param num_steps: number of deletion steps
#     :param use_cam: whether to use CAM
#     """
#     model.eval()
#     device = image.device

#     if use_gradcam:
#         heatmap = get_gradcam_CIFER10(model, image, label)
#     else:
#         heatmap = get_cam_CIFER10(model, image, label)
    
#     h, w = heatmap.shape
#     flat_heatmap = heatmap.view(-1)
#     sorted_indices = torch.argsort(flat_heatmap, descending=True)
#     total_pixels = h * w
#     pixels_per_step = (total_pixels + num_steps - 1) // num_steps

#     current_image = image.clone()
#     scores = []

#     with torch.no_grad():
#         init_output = model(current_image.unsqueeze(0))
#         init_prob = F.softmax(init_output, dim=1)
#         init_score = init_prob[0, label].item()
#         scores.append(init_score)

#     for step in range(num_steps):
#         start = step * pixels_per_step
#         if start >= total_pixels:
#             break
#         end = min(start + pixels_per_step, total_pixels)
#         indices_to_mask = sorted_indices[start:end]

#         for idx in indices_to_mask:
#             y = idx // w
#             x = idx % w
#             current_image[:, y, x] = 0.5
#             with torch.no_grad():
#                 output = model(current_image.unsqueeze(0))
#                 probs = F.softmax(output, dim=1)
#                 score = probs[0, label].item()
#                 scores.append(score)

#     auc = np.trapz(scores, dx=1.0 / num_steps)
#     return scores, auc

def compute_deletion_curve(model, image, label, num_steps=10, use_gradcam=False):
    """
    Compute the deletion metric for a given image
    
    :param model: trained model
    :param image: single image tensor, shape: [C, H, W]
    :param label: gt label
    :param num_steps: number of deletion steps
    :param use_gradcam: whether to use Grad-CAM (not CAM)
    """
    model.eval()
    device = image.device

    if use_gradcam:
        heatmap = get_gradcam_CIFER10(model, image, label)
    else:
        heatmap = get_cam_CIFER10(model, image, label)
    
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

        # Mask all pixels for this step
        for idx in indices_to_mask:
            y = idx // w
            x = idx % w
            current_image[:, y, x] = 0.5

        # Compute score AFTER masking all pixels in this step
        with torch.no_grad():
            output = model(current_image.unsqueeze(0))
            probs = F.softmax(output, dim=1)
            score = probs[0, label].item()
            scores.append(score)

    # ✅ Only compute AUC and return AFTER all steps
    auc = np.trapz(scores, dx=1.0 / num_steps)
    return scores, auc
~~~
