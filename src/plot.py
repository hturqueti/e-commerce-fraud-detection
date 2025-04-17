import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re
import sys


from pathlib import Path
from sklearn.metrics import confusion_matrix as c_matrix
from typing import Literal, Optional, Sequence, Tuple, Union

# Project path
project_path = Path('.').resolve().parent

# Include in path
sys.path.append(str(project_path))

# Import auxiliary functions
from src import auxiliary as aux

# Import parameters
prm = aux.load_parameters()

# Set Plotly theme and render
pio.templates.default = prm['plotly']['theme']
pio.renderers.default = prm['plotly']['renderer']


# Functions
# Plot bar chart
def counting_bar_chart(
    df: pd.DataFrame,
    binary_feature: str,
    positive_label: str,
    negative_label: str,
    group_feature: Optional[str] = None,
    group_labels: Optional[dict] = None,
    value_format: Literal['absolute', 'relative'] = 'absolute',
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
) -> go.Figure:

    groupby_columns = [binary_feature]
    if group_feature is not None:
        groupby_columns.append(group_feature)

    df_count = (
        df[groupby_columns]
        .value_counts()
        .reset_index()
    )

    # Rename columns
    df_count.columns = [*groupby_columns, 'count']

    # Calculate percentage
    if group_feature is not None:
        df_total_count = df_count[[group_feature, 'count']].groupby(group_feature).sum().rename(columns={'count': 'total_group_feature'})
        df_count = df_count.merge(df_total_count, on=group_feature)
        df_count['percentage'] = (df_count['count'] / df_count['total_group_feature'])
        df_count.drop(['total_group_feature'], axis=1, inplace=True)
    else:
        df_count['percentage'] = (df_count['count'] / df_count['count'].sum())

    # Change labels of features
    df_count[binary_feature] = df_count[binary_feature].map({1: positive_label, 0: negative_label})
    if group_feature is not None:
        df_count[group_feature] = df_count[group_feature].map(group_labels)

    # Sort dataframe
    df_count = df_count.sort_values(by=group_feature if group_feature is not None else binary_feature)

    # Create bar chart
    fig = px.bar(
        df_count,
        x=group_feature if group_feature is not None else binary_feature,
        y='count' if value_format == 'absolute' else 'percentage',
        text_auto=True,
        barmode='group' if group_feature is not None else 'stack',
        color=binary_feature,
        color_discrete_map={negative_label: '#1D69E0', positive_label: '#FA4549'},
        width=500,
        height=400,
    )

    # Change text format
    if value_format == 'absolute':
        fig.update_traces(texttemplate='%{y:,.0f}')
    elif value_format == 'relative':
        fig.update_traces(texttemplate='%{y:,.1%}')
        fig.update_layout(yaxis=dict(tickformat='.0%'))

   # Set titles and legend
    fig.update_layout(
        title=title,
        title_subtitle_text=subtitle,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True if group_feature is not None else False,
        legend_title_text="",
    )

    # Move legend to top center, if applicable
    if group_feature is not None:
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="center", x=0.5))

    # Hide the grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True)

    return fig

# Plot overlaid histogram
def overlaid_histograms(
    df: pd.DataFrame,
    numeric_feature: str,
    binary_feature: str,
    positive_label: str,
    negative_label: str,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    min_quantile: Optional[float] = None,
    max_quantile: Optional[float] = None,
    nbinsx: int = 30
) -> go.Figure:
    # Filter by quantiles, if applied
    """
    Generate a figure with two overlaid histograms of a numeric feature.
    The first histogram is for negative values of a binary feature, and the second histogram is for positive values of a binary feature.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe which contains the numeric feature and the binary feature.
    numeric_feature : str
        The name of the numeric feature.
    binary_feature : str
        The name of the binary feature.
    positive_label : str
        The label for the positive values of the binary feature.
    negative_label : str
        The label for the negative values of the binary feature.
    title : Optional[str], optional
        The title of the figure, by default None.
    subtitle : Optional[str], optional
        The subtitle of the figure, by default None.
    xaxis_title : Optional[str], optional
        The title of the x-axis, by default None.
    yaxis_title : Optional[str], optional
        The title of the y-axis, by default None.
    min_quantile : Optional[float], optional
        The quantile of the minimum value, by default None.
    max_quantile : Optional[float], optional
        The quantile of the maximum value, by default None.
    nbinsx : int, optional
        The number of bins in the x-axis, by default 30.

    Returns
    -------
    go.Figure
        A figure with two overlaid histograms of a numeric feature.
    """
    if min_quantile is not None or max_quantile is not None:
        df = aux.filter_quantiles(df, numeric_feature, min_quantile, max_quantile)

    # Filter out by binary feature
    df_positive = df.loc[df[binary_feature] == 1, numeric_feature]
    df_negative = df.loc[df[binary_feature] == 0, numeric_feature]

    # Add two histograms
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df_negative,
            name=negative_label,
            nbinsx=nbinsx,
            histnorm='probability',
            marker=dict(color=prm['plotly']['colors']['blue']) 
        )
    )
    fig.add_trace(
        go.Histogram(
            x=df_positive,
            name=positive_label,
            nbinsx=nbinsx,
            histnorm='probability',
            marker=dict(color=prm['plotly']['colors']['red']) 
        )
    )

    # Overlay both histograms
    fig.update_layout(barmode='overlay', width=600, height=550)

    # Move legend to top center
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="center", x=0.5))

    # Reduce opacity to see both histograms
    fig.update_traces(opacity=prm['plotly']['transparency'])

    # Set titles
    fig.update_layout(
        title=title,
        title_subtitle_text=subtitle,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
    )

    # Set comma as thousands separator
    fig.update_xaxes(tickformat=',')
    
    # Set percentage on y-axis
    fig.update_yaxes(tickformat=',.0%')

    # Hide the grid lines
    fig.update_xaxes(showgrid=False)
    # fig.update_yaxes(showgrid=False)

    return fig


#Plot score distribution
def score_distribution(df: pd.DataFrame, score_number: int, nbinsx: int = 30) -> go.Figure:
    # Get min and max quantiles from parameter file
    min_quantile=prm['purchase-database']['quantiles']['min']
    max_quantile=prm['purchase-database']['quantiles']['max']

    # Calculate data_coverage
    data_coverage = max_quantile - min_quantile

    # Filter df by min and max quantiles
    df = aux.filter_quantiles(df, f'score_{score_number}', min_quantile, max_quantile)

    # Get min and max x values
    min_x = df[f'score_{score_number}'].min()
    max_x = df[f'score_{score_number}'].max()

    fig = overlaid_histograms(
        df=df,
        numeric_feature=f'score_{score_number}',
        binary_feature='fraud',
        positive_label='fraud',
        negative_label='not fraud',
        title=f'Score {score_number} distribution',
        subtitle=f'{data_coverage:,.2%} of data, from {min_x:,.0f} to {max_x:,.0f}',
        xaxis_title='Score value',
        yaxis_title='Probability of purchases',
        min_quantile=min_quantile,
        max_quantile=max_quantile,
        nbinsx=nbinsx,
    )

    return fig


def roc_curve(
    fpr: Sequence[float],
    tpr: Sequence[float],
    thresholds: Sequence[float],
    title: Optional[str] = "ROC Curve",
    subtitle: Optional[str] = None
) -> go.Figure:
    """
    Plots an ROC curve using Plotly, given the false positive rates (fpr),
    true positive rates (tpr), and thresholds from scikit-learn's roc_curve.

    Parameters
    ----------
    fpr : array-like
        False Positive Rates from roc_curve.
    tpr : array-like
        True Positive Rates from roc_curve.
    thresholds : array-like
        Thresholds corresponding to fpr and tpr.
    title : str, optional
        Title of the plot, by default "ROC Curve".
    subtitle : str, optional
        Subtitle of the plot, by default None.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure object containing the ROC curve.
    """
    fig = go.Figure()

    # Add the ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name='ROC Curve',
            text=[f"{thr:.4f}" for thr in thresholds],
            hovertemplate=(
                "FPR: %{x:.4f}<br>"
                "TPR: %{y:.4f}<br>"
                "Threshold: %{text}"
            ),
            line=dict(width=2, color='#1D69E0')
        )
    )

    # Add the random guess reference line (y = x), dashed
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='#FA4549'),
            name='Random Guess'
        )
    )

    # Update layout for titles, axis labels, and making the plot square
    fig.update_layout(
        title=title,
        title_subtitle=dict(
            text=subtitle
        ),
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=600,   # Set figure width
        height=600,  # Set figure height (same as width for a square aspect)
        showlegend=False,
    )

    return fig


Label = Union[int, str]
def confusion_matrix(
    y_true: Sequence[Label],
    y_pred: Sequence[Label],
    labels: Tuple[Label, Label] = (0, 1),
    class_names: Tuple[str, str] = ("Not fraud", "Fraud"),
    figure_size: int = 600,
    percent_digits: int = 1,
) -> go.Figure:
    """
    Plot a 2×2 confusion matrix with Plotly, following scikit‑learn’s default:
      • negative label at index 0 (row 0 / col 0)
      • positive label at index 1 (row 1 / col 1)
    The matrix is displayed so that the negative class appears in the top‑left.

    Parameters
    ----------
    y_true : Sequence[Label]
        Ground‐truth labels.
    y_pred : Sequence[Label]
        Predicted labels.
    labels : (neg_label, pos_label), default=(0,1)
        Actual label values for negative and positive classes.
    class_names : (neg_name, pos_name), default=("Not fraud","Fraud")
        Display names for those two classes.
    figure_size : int, default=600
        Width and height of the square figure in pixels.
    percent_digits : int, default=1
        Decimal places for the percentage text.

    Returns
    -------
    go.Figure
        A Plotly heatmap figure of the confusion matrix.
    """
    neg_label, pos_label = labels
    neg_name,  pos_name  = class_names

    # 1) Compute confusion matrix with fixed label order
    cm = c_matrix(y_true, y_pred, labels=[neg_label, pos_label])
    cm_pct = cm / cm.sum() * 100.0

    # 2) Prepare cell annotation: count + percentage
    fmt = f"{{:.{percent_digits}f}}"
    cell_text = [
        [
            f"{cm[i, j]}<br>({fmt.format(cm_pct[i, j])}%)"
            for j in range(2)
        ]
        for i in range(2)
    ]

    # 3) Build the heatmap
    colorscale = [[0.0, "#FFFFFF"], [1.0, "#1D69E0"]]  # white→blue

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[f"{neg_name}", f"{pos_name}"],
            y=[f"{neg_name}", f"{pos_name}"],
            text=cell_text,
            texttemplate="%{text}",
            hovertemplate=(
                "True %{y}<br>"
                "Predicted %{x}<br>"
                "%{text}<extra></extra>"
            ),
            colorscale=colorscale,
            showscale=False,
        )
    )

    # 4) Flip y‑axis so that row 0 appears at the top
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")

    # 5) Layout settings
    fig.update_layout(
        width=figure_size,
        height=figure_size,
        xaxis_title="Predicted label",
        yaxis_title="True label",
    )

    return fig