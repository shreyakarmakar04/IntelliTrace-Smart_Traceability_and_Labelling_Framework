#updated main code with pics of dataset using resnet

import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report, accuracy_score
import glob
from tqdm import tqdm
import joblib
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Define paths - UPDATE THESE PATHS FOR YOUR SYSTEM
DATASET_PATH = "/content/drive/MyDrive/bottle_images"  # Update this path
CATEGORY = ""  # Empty since we're using the full path above
FEATURE_EXTRACTOR = "resnet18"  # Options: resnet18, resnet50, wide_resnet50_2
METHOD = "pca"  # Options: pca, ocsvm (One-Class SVM)

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None, dataset_type='auto'):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Auto-detect dataset structure
        if dataset_type == 'auto':
            dataset_type = self._detect_dataset_structure(folder_path)

        if dataset_type == 'mvtec':
            self._load_mvtec_structure(folder_path)
        elif dataset_type == 'flat':
            self._load_flat_structure(folder_path)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        print(f"Dataset type: {dataset_type}")
        print(f"Found {len(self.image_paths)} images:")
        if len(self.image_paths) > 0:
            print(f"  Normal: {sum(1 for l in self.labels if l == 0)}")
            print(f"  Anomaly: {sum(1 for l in self.labels if l == 1)}")
        else:
            print("  No images found!")

    def _detect_dataset_structure(self, folder_path):
        """Auto-detect if this is MVTec structure or flat structure"""
        # Check for MVTec structure (train/test folders)
        if (os.path.exists(os.path.join(folder_path, 'train')) and
            os.path.exists(os.path.join(folder_path, 'test'))):
            return 'mvtec'
        # Check for flat structure with images directly in folder
        else:
            return 'flat'

    def _get_image_files(self, folder_path):
        """Get all image files from a folder"""
        if not os.path.exists(folder_path):
            return []

        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        return sorted(image_paths)

    def _load_mvtec_structure(self, folder_path):
        """Load MVTec-style dataset structure"""
        # Load training images (all good)
        train_good_path = os.path.join(folder_path, 'train', 'good')
        if os.path.exists(train_good_path):
            train_images = self._get_image_files(train_good_path)
            self.image_paths.extend(train_images)
            self.labels.extend([0] * len(train_images))
            print(f"  Found {len(train_images)} training images")

        # Load test images
        test_path = os.path.join(folder_path, 'test')
        if os.path.exists(test_path):
            # Good test images
            test_good_path = os.path.join(test_path, 'good')
            if os.path.exists(test_good_path):
                test_good_images = self._get_image_files(test_good_path)
                self.image_paths.extend(test_good_images)
                self.labels.extend([0] * len(test_good_images))
                print(f"  Found {len(test_good_images)} test good images")

            # Defect test images
            defect_count = 0
            for item in os.listdir(test_path):
                defect_path = os.path.join(test_path, item)
                if os.path.isdir(defect_path) and item != 'good':
                    defect_images = self._get_image_files(defect_path)
                    if defect_images:
                        self.image_paths.extend(defect_images)
                        self.labels.extend([1] * len(defect_images))
                        defect_count += len(defect_images)
                        print(f"  Found defect type: {item} ({len(defect_images)} images)")

            if defect_count == 0:
                print("  No defect images found in test folder")

    def _load_flat_structure(self, folder_path):
        """Load flat structure with filename-based labeling"""
        image_paths = self._get_image_files(folder_path)

        if not image_paths:
            print(f"  No images found in {folder_path}")
            return

        for path in image_paths:
            filename = os.path.basename(path).lower()
            folder_name = os.path.basename(os.path.dirname(path)).lower()

            # Check folder name first, then filename
            if any(keyword in folder_name for keyword in ['good', 'normal', 'ok', 'train']):
                label = 0
            elif any(keyword in folder_name for keyword in ['bad', 'defect', 'anomaly', 'abnormal', 'test']):
                label = 1
            elif any(keyword in filename for keyword in ['good', 'normal', 'ok']):
                label = 0
            elif any(keyword in filename for keyword in ['bad', 'defect', 'anomaly', 'abnormal']):
                label = 1
            else:
                # Default assumption: assume normal if unclear
                # This is safer than assuming anomaly
                label = 0
                #print(f"  Warning: Unclear label for {filename}, assuming normal")

            self.image_paths.append(path)
            self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            # Verify image can be loaded properly
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError("Empty image")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        label = self.labels[idx]

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error transforming image {img_path}: {e}")
                # Create a tensor of zeros if transform fails
                image = torch.zeros(3, 224, 224)

        return image, label, img_path

class FeatureExtractor:
    def __init__(self, model_name='resnet18', device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load pre-trained model with proper weights parameter
        if model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model_name == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Model {model_name} not supported")

        # Remove the final classification layer
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, dataloader):
        features, labels, paths = [], [], []

        with torch.no_grad():
            for batch_idx, (imgs, lbls, pths) in enumerate(tqdm(dataloader, desc="Extracting features")):
                try:
                    imgs = imgs.to(self.device)
                    feats = self.model(imgs)

                    # Flatten features
                    feats = feats.view(feats.size(0), -1)

                    features.append(feats.cpu().numpy())
                    labels.extend(lbls.numpy())
                    paths.extend(pths)

                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {e}")
                    continue

        if not features:
            raise ValueError("No features extracted! Check your data loading.")

        return np.vstack(features), np.array(labels), paths

def train_pca_detector(features, n_components=0.95):
    """Train PCA-based anomaly detector"""
    print(f"Training PCA with {len(features)} normal samples...")

    if len(features) == 0:
        raise ValueError("No features provided for training!")

    pca = PCA(n_components=n_components)
    pca.fit(features)

    # Calculate reconstruction error for normal samples
    proj = pca.transform(features)
    recon = pca.inverse_transform(proj)
    errors = np.mean((features - recon)**2, axis=1)

    # Set threshold based on statistical analysis of normal samples
    threshold = np.percentile(errors, 95)  # 95th percentile

    print(f"PCA components: {pca.n_components_}")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    print(f"Threshold set to: {threshold:.6f}")

    return pca, threshold

def train_ocsvm_detector(features):
    """Train One-Class SVM anomaly detector"""
    print(f"Training One-Class SVM with {len(features)} normal samples...")

    if len(features) == 0:
        raise ValueError("No features provided for training!")

    ocsvm = OneClassSVM(gamma='scale', nu=0.05)
    ocsvm.fit(features)

    return ocsvm

def detect_anomalies_pca(pca, features):
    """Detect anomalies using PCA reconstruction error"""
    proj = pca.transform(features)
    recon = pca.inverse_transform(proj)
    return np.mean((features - recon)**2, axis=1)

def detect_anomalies_ocsvm(ocsvm, features):
    """Detect anomalies using One-Class SVM"""
    # Returns negative values for anomalies, positive for normal
    scores = ocsvm.decision_function(features)
    # Convert to anomaly scores (higher = more anomalous)
    return -scores

def save_results_to_excel(paths, scores, labels, predictions, threshold, filename="results.xlsx"):
    """Save detailed per-image results to Excel file"""
    print(f"\nSaving results to {filename}...")

    # Prepare data for DataFrame
    results_data = []

    for i, (path, score, true_label, pred_label) in enumerate(zip(paths, scores, labels, predictions)):
        # Get image filename and directory
        img_filename = os.path.basename(path)
        img_directory = os.path.dirname(path)

        # Convert labels to readable format
        true_class = "Good" if true_label == 0 else "Defective"
        predicted_class = "Good" if pred_label == 0 else "Defective"

        # Determine if prediction is correct
        is_correct = true_label == pred_label
        prediction_status = "Correct" if is_correct else "Incorrect"

        # Calculate confidence (distance from threshold)
        confidence = abs(score - threshold)

        results_data.append({
            'Image_ID': i + 1,
            'Image_Filename': img_filename,
            'Image_Path': path,
            'Directory': img_directory,
            'Anomaly_Score': round(score, 6),
            'Threshold': round(threshold, 6),
            'True_Label': true_class,
            'Predicted_Label': predicted_class,
            'Prediction_Status': prediction_status,
            'Confidence': round(confidence, 6),
            'Above_Threshold': score > threshold
        })

    # Create DataFrame and save to Excel
    df = pd.DataFrame(results_data)

    # Create summary statistics
    summary_data = {
        'Metric': [
            'Total Images',
            'Good Images (True)',
            'Defective Images (True)',
            'Good Images (Predicted)',
            'Defective Images (Predicted)',
            'Correct Predictions',
            'Incorrect Predictions',
            'Accuracy',
            'Threshold Used',
            'Average Anomaly Score (Good)',
            'Average Anomaly Score (Defective)'
        ],
        'Value': [
            len(paths),
            sum(1 for l in labels if l == 0),
            sum(1 for l in labels if l == 1),
            sum(1 for p in predictions if p == 0),
            sum(1 for p in predictions if p == 1),
            sum(1 for t, p in zip(labels, predictions) if t == p),
            sum(1 for t, p in zip(labels, predictions) if t != p),
            f"{accuracy_score(labels, predictions):.4f}",
            f"{threshold:.6f}",
            f"{np.mean([s for s, l in zip(scores, labels) if l == 0]):.6f}" if any(l == 0 for l in labels) else "N/A",
            f"{np.mean([s for s, l in zip(scores, labels) if l == 1]):.6f}" if any(l == 1 for l in labels) else "N/A"
        ]
    }
    summary_df = pd.DataFrame(summary_data)

    # Save to Excel with multiple sheets
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Main results sheet
        df.to_excel(writer, sheet_name='Image_Results', index=False)

        # Summary sheet
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Separate sheets for good and defective predictions
        good_predictions = df[df['Predicted_Label'] == 'Good']
        defective_predictions = df[df['Predicted_Label'] == 'Defective']

        if not good_predictions.empty:
            good_predictions.to_excel(writer, sheet_name='Predicted_Good', index=False)

        if not defective_predictions.empty:
            defective_predictions.to_excel(writer, sheet_name='Predicted_Defective', index=False)

    print(f"Results saved successfully!")
    print(f"  - Total images processed: {len(paths)}")
    print(f"  - Results saved to: {os.path.abspath(filename)}")
    print(f"  - Sheets created: Image_Results, Summary, Predicted_Good, Predicted_Defective")

    return df

def visualize_results(paths, scores, labels, threshold, num_images=5):
    """Visualize detection results"""
    normal_indices = np.where(labels == 0)[0]
    anomaly_indices = np.where(labels == 1)[0]

    if len(normal_indices) == 0:
        print("Warning: No normal samples found for visualization")
        return
    if len(anomaly_indices) == 0:
        print("Warning: No anomaly samples found for visualization")
        return

    np.random.seed(42)
    show_normal = np.random.choice(normal_indices, min(num_images, len(normal_indices)), replace=False)
    show_anomaly = np.random.choice(anomaly_indices, min(num_images, len(anomaly_indices)), replace=False)

    fig, axs = plt.subplots(2, num_images, figsize=(num_images*3, 6))
    if num_images == 1:
        axs = axs.reshape(2, 1)

    # Show normal samples
    for i, idx in enumerate(show_normal):
        try:
            img = Image.open(paths[idx])
            axs[0, i].imshow(img)
            prediction = "CORRECT" if scores[idx] < threshold else "WRONG"
            color = 'green' if scores[idx] < threshold else 'red'
            axs[0, i].set_title(f"Normal ({prediction})\nScore: {scores[idx]:.4f}", color=color, fontsize=8)
            axs[0, i].axis('off')
        except Exception as e:
            axs[0, i].text(0.5, 0.5, f"Error loading\n{os.path.basename(paths[idx])}",
                          ha='center', va='center', transform=axs[0, i].transAxes)
            axs[0, i].axis('off')

    # Show anomaly samples
    for i, idx in enumerate(show_anomaly):
        try:
            img = Image.open(paths[idx])
            axs[1, i].imshow(img)
            prediction = "CORRECT" if scores[idx] > threshold else "WRONG"
            color = 'green' if scores[idx] > threshold else 'red'
            axs[1, i].set_title(f"Anomaly ({prediction})\nScore: {scores[idx]:.4f}", color=color, fontsize=8)
            axs[1, i].axis('off')
        except Exception as e:
            axs[1, i].text(0.5, 0.5, f"Error loading\n{os.path.basename(paths[idx])}",
                          ha='center', va='center', transform=axs[1, i].transAxes)
            axs[1, i].axis('off')

    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(labels, scores):
    """Plot Precision-Recall curve instead of ROC curve"""
    try:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        ap_score = average_precision_score(labels, scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'Precision-Recall curve (AP = {ap_score:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        # Add baseline (random classifier)
        baseline = sum(labels) / len(labels)
        plt.axhline(y=baseline, color='navy', linestyle='--',
                   label=f'Random classifier (AP = {baseline:.4f})')
        plt.legend(loc="lower left")

        plt.show()
    except Exception as e:
        print(f"Error plotting Precision-Recall curve: {e}")

def main():
    # Handle direct path or path + category
    if CATEGORY:
        dataset_full_path = os.path.join(DATASET_PATH, CATEGORY)
    else:
        dataset_full_path = DATASET_PATH

    if not os.path.exists(dataset_full_path):
        print(f"Error: Dataset path does not exist: {dataset_full_path}")
        print("Please check your DATASET_PATH and CATEGORY settings.")
        print("\nTo fix this issue:")
        print("1. Verify the path exists on your system")
        print("2. Update DATASET_PATH to point to the correct folder")
        print("3. If using a category subfolder, update CATEGORY")
        print("\nCurrent settings:")
        print(f"  DATASET_PATH: {DATASET_PATH}")
        print(f"  CATEGORY: {CATEGORY}")
        print(f"  Full path: {dataset_full_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset path: {dataset_full_path}")
    print(f"Feature extractor: {FEATURE_EXTRACTOR}")
    print(f"Method: {METHOD}")
    print("-" * 50)

    try:
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(FEATURE_EXTRACTOR, device)

        # Create dataset and dataloader
        dataset = CustomDataset(dataset_full_path, feature_extractor.transform)

        if len(dataset) == 0:
            print("Error: No images found in the dataset!")
            print("Please check that your dataset folder contains images.")
            return

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

        # Extract features
        print("\nExtracting features...")
        features, labels, paths = feature_extractor.extract_features(dataloader)
        print(f"Extracted features shape: {features.shape}")

        # Get normal samples for training
        normal_mask = labels == 0
        normal_features = features[normal_mask]

        if len(normal_features) == 0:
            print("Error: No normal samples found for training!")
            print("Please check your image naming convention or folder structure.")
            print("For MVTec format: ensure you have train/good or test/good folders")
            print("For flat format: ensure some images contain keywords like 'good', 'normal', 'ok'")
            return

        if len(normal_features) < 5:
            print(f"Warning: Only {len(normal_features)} normal samples found. Results may be unreliable.")

        print(f"Using {len(normal_features)} normal samples for training")

        # Train detector
        if METHOD == "pca":
            detector, threshold = train_pca_detector(normal_features)
            scores = detect_anomalies_pca(detector, features)
        elif METHOD == "ocsvm":
            detector = train_ocsvm_detector(normal_features)
            scores = detect_anomalies_ocsvm(detector, features)
            # For OCSVM, we need to determine threshold from the scores
            normal_scores = scores[normal_mask]
            threshold = np.percentile(normal_scores, 95) if len(normal_scores) > 0 else 0
        else:
            raise ValueError(f"Method {METHOD} not supported")

        # Evaluation
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:  # We have both normal and anomaly samples
            try:
                auc = roc_auc_score(labels, scores)
                ap_score = average_precision_score(labels, scores)
                print(f"\nAUROC: {auc:.4f}")
                print(f"Average Precision: {ap_score:.4f}")

                # Find optimal threshold using precision-recall curve
                precision, recall, thresholds = precision_recall_curve(labels, scores)
                f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
                best_idx = np.argmax(f1_scores)
                optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else threshold
                print(f"Optimal threshold (best F1): {optimal_threshold:.6f}")

                # Use optimal threshold for predictions
                predictions = (scores > optimal_threshold).astype(int)

                print(f"\nUsing threshold: {optimal_threshold:.6f}")
                print("\nClassification Report:")
                print(classification_report(labels, predictions, target_names=["Normal", "Anomaly"]))
                print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")

                # Save results to Excel
                results_df = save_results_to_excel(paths, scores, labels, predictions, optimal_threshold)

                # Visualizations
                plot_precision_recall_curve(labels, scores)
                visualize_results(paths, scores, labels, optimal_threshold)

            except Exception as e:
                print(f"Error during evaluation: {e}")
                # Still save results even if evaluation fails
                predictions = (scores > threshold).astype(int) if 'threshold' in locals() else np.zeros_like(labels)
                save_results_to_excel(paths, scores, labels, predictions, threshold if 'threshold' in locals() else 0)

        else:
            print("Warning: Only one class found in dataset. Cannot compute metrics.")
            print("Showing anomaly scores for all samples:")
            for i, (path, score) in enumerate(zip(paths[:10], scores[:10])):  # Show first 10
                print(f"{os.path.basename(path)}: {score:.6f}")
            if len(paths) > 10:
                print(f"... and {len(paths) - 10} more samples")

            # Still save results
            predictions = (scores > threshold).astype(int) if 'threshold' in locals() else np.zeros_like(labels)
            save_results_to_excel(paths, scores, labels, predictions, threshold if 'threshold' in locals() else 0)

        # Save the trained model
        model_filename = f"{FEATURE_EXTRACTOR}_{METHOD}_detector.pkl"
        try:
            joblib.dump({
                'detector': detector,
                'threshold': threshold if 'threshold' in locals() else None,
                'method': METHOD,
                'feature_extractor': FEATURE_EXTRACTOR
            }, model_filename)
            print(f"\nModel saved as: {model_filename}")
        except Exception as e:
            print(f"Error saving model: {e}")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
