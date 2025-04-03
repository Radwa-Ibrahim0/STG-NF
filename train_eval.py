import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs, init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params, calc_num_of_params
from utils.scoring_utils import score_dataset

def main():
    parser = init_parser()
    args = parser.parse_args()

    if args.seed == 999:
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    num_of_params = calc_num_of_params(model)
    trainer = Trainer(args, model, loader['train'], loader['test'],
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        trainer.load_checkpoint(pretrained)
    else:
        writer = SummaryWriter()
        trainer.train(log_writer=writer)
        dump_args(args, args.ckpt_dir)

    normality_scores = trainer.test()
    auc, scores_np = score_dataset(normality_scores, dataset["test"].metadata, args=args)

    # Convert scores to binary predictions
    gt_np = np.concatenate([np.array(meta["label"]) for meta in dataset["test"].metadata])
    
    # Print Precision-Recall Thresholds
    precision, recall, thresholds_pr = precision_recall_curve(gt_np, scores_np)
    print("\nPrecision-Recall Thresholds:")
    for i in range(0, len(thresholds_pr), max(1, len(thresholds_pr)//10)):
        print(f"Threshold: {thresholds_pr[i]:.4f}, Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}")
    
    # Find Best Threshold Using F1-Score
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_idx]
    print(f"\nBest Threshold (Highest F1-Score): {best_threshold:.4f}")
    
    # Find Best Threshold Using ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(gt_np, scores_np)
    gmeans = np.sqrt(tpr * (1 - fpr))
    best_roc_idx = np.argmax(gmeans)
    best_roc_threshold = thresholds_roc[best_roc_idx]
    print(f"\nBest Threshold (ROC-based): {best_roc_threshold:.4f}")

    # Print Classification Report
    y_pred = (scores_np >= best_threshold).astype(int)
    print("\nClassification Report:")
    print(classification_report(gt_np, y_pred, digits=4))

    # Plot ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(args.ckpt_dir + "/roc_curve.png")
    plt.show()

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='red')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(args.ckpt_dir + "/pr_curve.png")
    plt.show()

    # Visualize Anomaly Scores over Time
    plt.figure(figsize=(12, 5))
    plt.plot(scores_np, label="Anomaly Score", color='purple')
    plt.plot(gt_np, label="Ground Truth", linestyle='dashed', color='black')
    plt.xlabel("Frames")
    plt.ylabel("Score")
    plt.title("Anomaly Scores Over Time")
    plt.legend()
    plt.savefig(args.ckpt_dir + "/anomaly_scores.png")
    plt.show()

    # Logging results
    print("\n-------------------------------------------------------")
    print(f"\033[92m Done with {auc * 100:.2f}% AuC for {scores_np.shape[0]} samples\033[0m")
    print("-------------------------------------------------------\n\n")

if __name__ == '__main__':
    main()
