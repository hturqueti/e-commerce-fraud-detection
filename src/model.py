# Libraries
import numpy as np
import pandas as pd
import yaml

from pathlib import Path
from typing import Optional, Union

# Constants
project_path = Path('.').resolve().parent

def load_parameters() -> dict[str, str]:
    with open(project_path.joinpath("parameters.yaml"), 'r') as file:
        parameters = yaml.safe_load(file)
    return parameters


import numpy as np

def profit_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Calculates the total profit (or loss) including an opportunity cost for denying legitimate transactions.

    Assumptions:
    - y_pred[i] is the predicted probability of FRAUD (class 1).
      Therefore, if y_pred[i] < threshold, the transaction is approved 
      (because the model indicates a low probability of fraud).
    - Gains and losses are computed as follows:
        1) If a transaction is approved and actually legitimate (y_true = 0): 
           +10% of the transaction amount.
        2) If a transaction is approved and actually fraudulent (y_true = 1): 
           –100% of the transaction amount.
        3) If a transaction is denied and actually legitimate (y_true = 0): 
           –10% of the transaction amount (opportunity cost).
        4) If a transaction is denied and actually fraudulent (y_true = 1): 
           0 (no gain, no loss).

    Parameters:
    - y_true: array of true labels (0 for legitimate, 1 for fraud).
    - y_pred: array of predicted fraud probabilities (class 1).
    - amounts: array of transaction amounts.
    - threshold: cutoff used to approve the transaction (approve if predicted fraud probability < threshold).

    Returns:
    - The total profit (sum of gains and losses for all transactions).
    """

    # Approve if predicted fraud probability < threshold
    approve = (y_pred < threshold)

    # Compute the profit or loss for each transaction
    profit = np.where(
        approve,
        # If approved:
        np.where(y_true == 0, 0.1 * amounts,     # legitimate approved => +10%
                 -1.0 * amounts),               # fraud approved => -100%
        # If denied:
        np.where(y_true == 0, -0.1 * amounts,   # legitimate denied => -10% (opportunity cost)
                 0.0)                           # fraud denied => 0
    )

    return profit.sum()
