import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re
import sys

from pathlib import Path
from typing import Literal, Optional

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
    df_total_count = df_count[[group_feature, 'count']].groupby(group_feature).sum().rename(columns={'count': 'total_group_feature'})
    df_count = df_count.merge(df_total_count, on=group_feature)
    df_count['percentage'] = (df_count['count'] / df_count['total_group_feature'])
    df_count.drop(['total_group_feature'], axis=1, inplace=True)

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
        fig.update_traces(texttemplate='%{y:.0f}')
    elif value_format == 'relative':
        fig.update_traces(texttemplate='%{y:.1%}')
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

    # Hide the grid lines
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

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
