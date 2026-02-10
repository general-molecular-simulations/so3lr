import logging

from typing import Dict


def stop_by_lr(x: float, lr_min: float) -> bool:
    if x <= lr_min:
        logging.info('Current learning rate {} is <= {}. Stopping the training.'.format(x, lr_min))
        return True
    else:
        return False


def stop_by_metric(current_metrics: Dict[str, float], target_metrics: Dict[str, float]):
    for k, v in target_metrics.items():
        if current_metrics[k] <= v:
            logging.info('Accuracy for {} has value {} which is smaller than its specified threshold {}. '
                         'Stopping the training'.format(k, current_metrics[k], v))
            return True
        else:
            pass

    return False
