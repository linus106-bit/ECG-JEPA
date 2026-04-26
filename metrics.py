import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def safe_ratio(numerator, denominator):
  return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=np.float64), where=denominator > 0)


def compute_macro_sensitivity_specificity(y_true, y_pred):
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  if y_true.ndim == 1:
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    tp = np.diag(cm).astype(np.float64)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)
  else:
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn = mcm[:, 0, 0].astype(np.float64)
    fp = mcm[:, 0, 1].astype(np.float64)
    fn = mcm[:, 1, 0].astype(np.float64)
    tp = mcm[:, 1, 1].astype(np.float64)
  sensitivity_per_class = safe_ratio(tp, tp + fn)
  specificity_per_class = safe_ratio(tn, tn + fp)
  positive_per_class = tp + fn
  negative_per_class = tn + fp
  return {
    'sensitivity_per_class': sensitivity_per_class,
    'specificity_per_class': specificity_per_class,
    'positive_per_class': positive_per_class,
    'negative_per_class': negative_per_class,
    'tp_per_class': tp,
    'tn_per_class': tn,
    'fp_per_class': fp,
    'fn_per_class': fn,
    'sensitivity_macro': float(np.nanmean(sensitivity_per_class)),
    'specificity_macro': float(np.nanmean(specificity_per_class)),
  }


def safe_macro_ovr_auroc(y_true, y_score):
  classes = np.arange(y_score.shape[1])
  class_aurocs = []
  skipped_classes = []
  for cls in classes:
    y_true_binary = (y_true == cls).astype(np.int32)
    if y_true_binary.min() == y_true_binary.max():
      skipped_classes.append(int(cls))
      continue
    class_aurocs.append(roc_auc_score(y_true_binary, y_score[:, cls]))
  if len(class_aurocs) == 0:
    return float('nan'), skipped_classes
  return float(np.mean(class_aurocs)), skipped_classes


def _compute_optimal_thresholds(y_true, y_score, ovr):
  classes = np.arange(y_score.shape[1])
  threshold_rows = []
  threshold_curve_rows = []
  per_class_auroc = []
  for cls in classes:
    y_true_binary = (y_true == cls).astype(np.int32) if ovr else y_true[:, cls].astype(np.int32)
    if y_true_binary.min() == y_true_binary.max():
      continue
    fpr, tpr, thresholds = roc_curve(y_true_binary, y_score[:, cls])
    opt_idx = int(np.argmax(tpr - fpr))
    class_auroc = roc_auc_score(y_true_binary, y_score[:, cls])
    per_class_auroc.append(float(class_auroc))
    for i, th in enumerate(thresholds):
      row = {
        'class_index': int(cls),
        'threshold': round(float(th), 6),
        'sensitivity': float(tpr[i]),
        'specificity': float(1.0 - fpr[i]),
        'youden_j': float(tpr[i] - fpr[i]),
      }
      threshold_rows.append(row)
      threshold_curve_rows.append(row)
    threshold_rows.append({
      'class_index': int(cls),
      'auroc': float(class_auroc),
      'opt_youden_j': float(tpr[opt_idx] - fpr[opt_idx]),
      'opt_threshold': float(thresholds[opt_idx]),
      'opt_sens': float(tpr[opt_idx]),
      'opt_spec': float(1.0 - fpr[opt_idx]),
    })

  if not threshold_rows:
    return {
      'threshold_rows': [],
      'threshold_curve_rows': [],
      'macro_auroc_from_thresholds': float('nan'),
      'opt_youden_j_macro': float('nan'),
      'opt_threshold_macro': float('nan'),
      'opt_sens_macro': float('nan'),
      'opt_spec_macro': float('nan'),
    }

  opt_rows = [row for row in threshold_rows if 'opt_youden_j' in row]
  return {
    'threshold_rows': threshold_rows,
    'threshold_curve_rows': threshold_curve_rows,
    'macro_auroc_from_thresholds': float(np.mean(per_class_auroc)) if per_class_auroc else float('nan'),
    'opt_youden_j_macro': float(np.mean([row['opt_youden_j'] for row in opt_rows])) if opt_rows else float('nan'),
    'opt_threshold_macro': float(np.mean([row['opt_threshold'] for row in opt_rows])) if opt_rows else float('nan'),
    'opt_sens_macro': float(np.mean([row['opt_sens'] for row in opt_rows])) if opt_rows else float('nan'),
    'opt_spec_macro': float(np.mean([row['opt_spec'] for row in opt_rows])) if opt_rows else float('nan'),
  }


def compute_optimal_ovr_thresholds(y_true, y_score):
  return _compute_optimal_thresholds(y_true, y_score, ovr=True)


def compute_optimal_multilabel_thresholds(y_true, y_score):
  return _compute_optimal_thresholds(y_true, y_score, ovr=False)


def metric_semantics_text(single_label):
  if single_label:
    return ('Sensitivity=TPR(recall), positive=the class itself in one-vs-rest; '
            'Specificity=TNR, negative=all other classes in one-vs-rest.')
  return ('Sensitivity=TPR(recall), positive=label 1; '
          'Specificity=TNR, negative=label 0.')


def compute_per_label_metrics_from_confusion(metric_stats, per_label_auroc_map):
  tp = np.asarray(metric_stats['tp_per_class'], dtype=np.float64)
  tn = np.asarray(metric_stats['tn_per_class'], dtype=np.float64)
  fp = np.asarray(metric_stats['fp_per_class'], dtype=np.float64)
  fn = np.asarray(metric_stats['fn_per_class'], dtype=np.float64)
  sensitivity = np.asarray(metric_stats['sensitivity_per_class'], dtype=np.float64)
  specificity = np.asarray(metric_stats['specificity_per_class'], dtype=np.float64)
  total = tp + tn + fp + fn
  accuracy = safe_ratio(tp + tn, total)
  precision = safe_ratio(tp, tp + fp)
  f1 = safe_ratio(2.0 * precision * sensitivity, precision + sensitivity)
  per_label_metrics = []
  for cls in range(len(tp)):
    per_label_metrics.append({
      'class_index': int(cls),
      'auroc': float(per_label_auroc_map.get(cls, float('nan'))),
      'f1': float(f1[cls]),
      'accuracy': float(accuracy[cls]),
      'sensitivity': float(sensitivity[cls]),
      'specificity': float(specificity[cls]),
    })
  return per_label_metrics


def extract_per_label_auroc_map(threshold_rows):
  return {
    int(row['class_index']): float(row['auroc'])
    for row in threshold_rows
    if 'auroc' in row
  }


def compute_single_label_metrics(targets, logits, warn_fn=None):
  probs = torch.softmax(logits, dim=1).cpu().numpy()
  preds = logits.argmax(dim=1).cpu().numpy()
  f1 = f1_score(y_true=targets, y_pred=preds, average='macro')
  acc = accuracy_score(y_true=targets, y_pred=preds)
  metric_stats = compute_macro_sensitivity_specificity(targets, preds)
  auroc, skipped_classes = safe_macro_ovr_auroc(targets, probs)
  threshold_stats = compute_optimal_ovr_thresholds(targets, probs)
  metric_stats.update({
    'macro_auroc_from_thresholds': threshold_stats['macro_auroc_from_thresholds'],
    'opt_youden_j_macro': threshold_stats['opt_youden_j_macro'],
    'opt_threshold_macro': threshold_stats['opt_threshold_macro'],
    'opt_sens_macro': threshold_stats['opt_sens_macro'],
    'opt_spec_macro': threshold_stats['opt_spec_macro'],
    'threshold_rows': threshold_stats['threshold_rows'],
    'threshold_curve_rows': threshold_stats['threshold_curve_rows'],
  })
  if warn_fn and skipped_classes:
    warn_fn(f'AUROC skipped classes without both positive/negative samples: {skipped_classes}')
  return preds, probs, f1, acc, auroc, metric_stats


def compute_multi_label_metrics(targets, logits):
  probs = torch.sigmoid(logits).cpu().numpy()
  preds = (probs >= 0.5).astype(np.int32)
  f1 = f1_score(y_true=targets, y_pred=preds, average='macro')
  acc = accuracy_score(y_true=targets, y_pred=preds)
  auroc = roc_auc_score(y_true=targets, y_score=probs, average='macro')
  metric_stats = compute_macro_sensitivity_specificity(targets, preds)
  threshold_stats = compute_optimal_multilabel_thresholds(targets, probs)
  metric_stats.update({
    'macro_auroc_from_thresholds': threshold_stats['macro_auroc_from_thresholds'],
    'opt_youden_j_macro': threshold_stats['opt_youden_j_macro'],
    'opt_threshold_macro': threshold_stats['opt_threshold_macro'],
    'opt_sens_macro': threshold_stats['opt_sens_macro'],
    'opt_spec_macro': threshold_stats['opt_spec_macro'],
    'threshold_rows': threshold_stats['threshold_rows'],
    'threshold_curve_rows': threshold_stats['threshold_curve_rows'],
  })
  return preds, probs, f1, acc, auroc, metric_stats
