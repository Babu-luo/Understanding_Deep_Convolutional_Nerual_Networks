import torch
import torch.nn as nn
import torch.nn.functional as F
from models import VGG, ResNet, ResNext
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import get_cam_CIFAR10 as get_cam
from utils import get_gradcam_CIFAR10 as get_gradcam
from utils import compute_deletion_curve, compute_insertion_curve, create_combined_image
from fasterrcnn_utils import run_fasterrcnn_gradcam
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
# Close oneDNN optimization
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# ===== [ADDED FOR FASTER-RCNN] =====
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from fasterrcnn_gradcam import FasterRCNNGradCAM
# ================================= 

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
    checkpoint = torch.load(args.save_path, map_location=device,weights_only=True)
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
    
    acc = 1.0
    if args.vis == 'none':
        test
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
        return acc

    if args.vis != 'none' and args.hidden_strategy == 'none':
        # ===== [ADDED FOR CAM AND GRAD-CAM] =====
        global_sample_idx = 0 # control how many heatmaps have been generated

        os.makedirs("cam_results", exist_ok=True)
        cam_saved = 0
        max_cam_images = args.vis_num   # save only first vis_num CAMs

        os.makedirs("gradcam_results", exist_ok=True)
        gradcam_saved = 0
        max_gradcam_images = args.vis_num

        if args.vis != 'none':
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                for i in range(inputs.size(0)):
                    img = inputs[i]
                    pred = predicted[i].item()
                    gt = labels[i].item()

                    current_index = global_sample_idx
                    global_sample_idx += 1

                    if args.vis == 'cam' and cam_saved < max_cam_images and current_index in {24,52,33,97}:
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


    # ===== [QUANTITATIVE EVALUATION VISUALIZATION] =====
    deletion_curves = []
    insertion_curves = []
    num_eval_samples = 100
    eval_count = 0
    use_gradcam = False
    if args.vis == 'gradcam':
        use_gradcam = True
    hidden_strategy = args.hidden_strategy
    heatmap = None

    for data in test_loader:
        if eval_count >= num_eval_samples:
            break
        inputs, labels = data[0].to(device), data[1].to(device)

        for i in range(inputs.size(0)):
            if eval_count >= num_eval_samples:
                break
            
            with torch.no_grad():
                single_input = inputs[i].unsqueeze(0)
                output = model(single_input)
                _, predicted = torch.max(output, 1)

            if predicted.item() != labels[i].item():
                continue

            image = inputs[i]
            target_class = labels[i].item()

            model.eval()
            # _ = model(image.unsqueeze(0))
            if use_gradcam:
                heatmap = get_gradcam(model, image, target_class)
            else:
                heatmap = get_cam(model, image, target_class)

            try:
                scores = compute_deletion_curve(model, image, target_class, heatmap, num_steps=100, hidden_strategy=hidden_strategy)
                deletion_curves.append(scores)
                scores = compute_insertion_curve(model, image, target_class, heatmap, num_steps=100, hidden_strategy=hidden_strategy)
                insertion_curves.append(scores)
                eval_count += 1
            except Exception as e:
                print(f"Error on sample {eval_count}: {e}")
                continue
    all_scores = np.array(deletion_curves)
    mean_curve = np.mean(all_scores, axis=0)
    avg_auc = np.trapezoid(mean_curve, np.linspace(0, 1, len(mean_curve)))
    print(f"Average Deletion AUC: {avg_auc:.4f}")

    all_scores = np.array(insertion_curves)
    mean_curve = np.mean(all_scores, axis=0)
    avg_auc = np.trapezoid(mean_curve, np.linspace(0, 1, len(mean_curve)))
    print(f"Average Insertion AUC: {avg_auc:.4f}")

    # 平均曲线
    if deletion_curves and insertion_curves:
        d_all = np.array(deletion_curves)
        i_all = np.array(insertion_curves)
        d_avg = np.mean(d_all, axis=0)
        d_min, d_max = np.min(d_all, axis=0), np.max(d_all, axis=0)
        i_avg = np.mean(i_all, axis=0)
        i_min, i_max = np.min(i_all, axis=0), np.max(i_all, axis=0)

        steps = np.linspace(0, 1, len(d_avg))
        d_auc = np.trapezoid(d_avg, steps)
        i_auc = np.trapezoid(i_avg, steps)

        method_name = 'Grad-CAM' if use_gradcam else 'CAM'
        color = 'red' if use_gradcam else 'blue'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        plt.subplots_adjust(wspace=0.3)

        # Deletion Curve
        ax1.plot(steps, d_avg, color=color, linewidth=2, label=f'{method_name} (AUC: {d_auc:.4f})')
        ax1.fill_between(steps, d_min, d_max, color=color, alpha=0.15)
        ax1.set_xlabel('Fraction of Pixels Masked')
        ax1.set_ylabel('Target Class Probability')
        ax1.set_title(f'Deletion Metric - {method_name}')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Insertion Curve
        ax2.plot(steps, i_avg, color=color, linewidth=2, label=f'{method_name} (AUC: {i_auc:.4f})')
        ax2.fill_between(steps, i_min, i_max, color=color, alpha=0.15)
        ax2.set_xlabel('Fraction of Pixels Inserted')
        ax2.set_ylabel('Target Class Probability')
        ax2.set_title(f'Insertion Metric - {method_name}')
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.6)

        filename = f'results_{method_name.lower()}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figures saved as {filename}")
        # plt.show()

    # ==========================
    
    return acc







if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='The configs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--save_path', type=str, default='./checkpoints/resnet_best.pth')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default='test')
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
    parser.add_argument(
        '--hidden_strategy',
        type=str,
        choices=['zero', 'gray', 'mean', 'none'],
        default='none',
        help='Hidden strategy for insertion curve: zero | gray | mean | none'
    )
    # =================================================
    # ===== [ADDED FOR FASTER-RCNN] =====
    parser.add_argument(
        '--det_vis',
        action='store_true',
        help='Use FasterRCNN Grad-CAM visualization'
    )
    parser.add_argument(
        '--det_img',
        type=str,
        default='PASCAL VOC 2007',
        help='Input image for FasterRCNN Grad-CAM'
    )
    # =================================


    args = parser.parse_args()

    model = ResNet()
    # train / test
    if args.mode == 'train':
        train(model, args)
    else:
        import time
        start_time = time.perf_counter()
        test(model, args)
        end_time = time.perf_counter()
        print(f"Total time: {end_time - start_time}")

    # ===== [ADDED FOR FASTER-RCNN GRAD-CAM ENTRY] =====
    if args.det_vis:
        run_fasterrcnn_gradcam(args)
    # ===============================================


# D:\Anaconda_Envs\torch_env\python.exe main.py --mode test --save_path ./checkpoints/resnet_best.pth --vis cam --hidden_strategy zero
# D:\Anaconda_Envs\torch_env\python.exe main.py --mode test --save_path ./checkpoints/resnet_best.pth --vis cam --hidden_strategy gray
# D:\Anaconda_Envs\torch_env\python.exe main.py --mode test --save_path ./checkpoints/resnet_best.pth --vis cam --hidden_strategy mean


# D:\Anaconda_Envs\torch_env\python.exe main.py --run=train --model=cnn --optimizer=sgd --scheduler=cosine
# D:\Anaconda_Envs\torch_env\python.exe main.py --run=train --model=fcnn --optimizer=sgd --scheduler=step
# D:\Anaconda_Envs\torch_env\python.exe -m tensorboard.main --logdir=./logs_for_VGG
# D:\Anaconda_Envs\torch_env\python.exe main.py --mode train --batch_size 64 --lr 0.01 --num_epochs 25 --save_path ./checkpoints/vgg_best.pth --log_dir ./logs
# D:\Anaconda_Envs\torch_env\python.exe main.py --mode test --batch_size 64 --lr 0.005 --num_epochs 20 --save_path ./checkpoints/vgg_best_new.pth --log_dir ./logs_for_VGG
# D:\Anaconda_Envs\torch_env\python.exe main.py --mode train --batch_size 64 --lr 0.005 --num_epochs 20 --save_path ./checkpoints/resnet_best.pth --log_dir ./logs_for_ResNet
# D:\Anaconda_Envs\torch_env\python.exe main.py --mode train --batch_size 64 --lr 0.005 --num_epochs 20 --save_path ./checkpoints/resnext_best.pth --log_dir ./logs_for_ResNext

# D:\Anaconda_Envs\torch_env\python.exe main.py --mode test --save_path ./checkpoints/resnet_best.pth --vis cam --vis_num 10

# test faster-rcnn
# D:\Anaconda_Envs\torch_env\python.exe main.py --mode test --det_vis --det_img test.png
