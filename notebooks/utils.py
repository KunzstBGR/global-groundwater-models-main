import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

EPSILON = 1e-5

def predictions_to_df(index: pd.DataFrame, predictions: np.ndarray, group_ids, date_range, lead):
    
    predictions_df = index
    for i, f in enumerate(predictions):
        predictions_df[i] = f

    predictions_df = predictions_df.melt(id_vars=group_ids + ['time_idx'], value_vars=list(range(lead)), var_name='horizon', value_name='forecast')
    predictions_df['time_idx'] = predictions_df['time_idx'] + predictions_df['horizon']
    predictions_df = predictions_df.merge(date_range, on=['time_idx'], how='left')
    predictions_df['horizon'] += 1
    predictions_df.drop('time_idx', axis=1, inplace=True)
    predictions_df.set_index(group_ids+['time', 'horizon'], inplace=True)
    return predictions_df

nse = lambda pred, real: 1 - (np.sum((pred - real) ** 2) / np.sum((real - np.mean(real)) ** 2))
nrmse = lambda pred, real: np.sqrt(np.square(pred - real).mean(axis=0)) / np.diff(np.quantile(real, q=[0.25, 0.75]))[0]
rmse = lambda pred, real: np.sqrt(np.square(pred - real).mean(axis=0))
mbe = lambda pred, real: np.mean(pred - real, axis=0)
rmbe = lambda pred, real: mbe(pred, real) / np.std(real)
mae = lambda pred, real: (abs(pred - real)).mean(axis=0)

def interval_score(
        observations,
        alpha,
        q_dict=None,
        q_left=None,
        q_right=None,
        percent=False,
        check_consistency=True,
):
    """
    Compute interval scores (1) for an array of observations and predicted intervals.

    Either a dictionary with the respective (alpha/2) and (1-(alpha/2)) quantiles via q_dict needs to be
    specified or the quantiles need to be specified via q_left and q_right.

    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alpha : numeric
        Alpha level for (1-alpha) interval.
    q_dict : dict, optional
        Dictionary with predicted quantiles for all instances in `observations`.
    q_left : array_like, optional
        Predicted (alpha/2)-quantiles for all instances in `observations`.
    q_right : array_like, optional
        Predicted (1-(alpha/2))-quantiles for all instances in `observations`.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.

    Returns
    -------
    total : array_like
        Total interval scores.

    (1) Gneiting, T. and A. E. Raftery (2007). Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association 102(477), 359–378.
    """

    if q_dict is None:
        if q_left is None or q_right is None:
            raise ValueError(
                "Either quantile dictionary or left and right quantile must be supplied."
            )
    else:
        if q_left is not None or q_right is not None:
            raise ValueError(
                "Either quantile dictionary OR left and right quantile must be supplied, not both."
            )
        q_left = q_dict.get(alpha / 2)
        if q_left is None:
            raise ValueError(f"Quantile dictionary does not include {alpha / 2}-quantile")

        q_right = q_dict.get(1 - (alpha / 2))
        if q_right is None:
            raise ValueError(
                f"Quantile dictionary does not include {1 - (alpha / 2)}-quantile"
            )

    if check_consistency and np.any(q_left > q_right):
        raise ValueError("Left quantile must be smaller than right quantile.")

    sharpness = q_right - q_left
    calibration = (
            (
                    np.clip(q_left - observations, a_min=0, a_max=None)
                    + np.clip(observations - q_right, a_min=0, a_max=None)
            )
            * 2
            / alpha
    )
    if percent:
        sharpness = sharpness / np.std(observations)
        calibration = calibration / np.std(observations)
    total = sharpness + calibration
    return np.mean(total)


def get_metrics(prediction_df: pd.DataFrame, real_col='gwl', forecast_col='forecast', metrics_subset=None):
    
    metrics = [(nrmse, 'nRMSE'), (rmse, 'RMSE'), (nse, 'NSE'), (rmbe, 'rMBE'), (interval_score, 'Interval Score'), (mae, 'MAE')]
    
    if metrics_subset is not None:
        metrics = [(metric_func, metric_name) for metric_func, metric_name in metrics if metric_name in metrics_subset]
    
    _metrics = []
    for (proj_id, horizon), group in prediction_df.groupby(
            [prediction_df.index.get_level_values('proj_id'),
             prediction_df.index.get_level_values('horizon')]):
        for metric in metrics:
            if metric[0] == interval_score:
                group = group[group['forecast_q10'] - EPSILON < group['forecast_q90'] + EPSILON]
                value = metric[0](group[real_col].values, 0.2, q_left=group['forecast_q10'].values - EPSILON, q_right=group['forecast_q90'].values + EPSILON, percent=True)
            else:
                value = metric[0](group[forecast_col], group[real_col])
            _df = pd.DataFrame([
                {
                    'metric': metric[1],
                    'value': round(value, 3),
                    'proj_id': proj_id,
                    'horizon': horizon,
                }
            ])
            _metrics.append(_df)
    metrics_df = pd.concat(_metrics)
    metrics_df = metrics_df.set_index(['proj_id', 'horizon', 'metric']).unstack()
    metrics_df = metrics_df.droplevel(axis=1, level=0)
    return metrics_df


def plot_predictions(predictions_df, proj_id, horizon=1, forecast_col='forecast', figsize=None, confidence=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if horizon == 'all':
        _df = predictions_df[predictions_df.index.get_level_values('proj_id') == proj_id].reset_index()
        _df['group'] = _df['time'] - (_df['horizon'] * pd.offsets.Week(1, weekday=6))
        _df[_df['horizon'] == 1].set_index('time')[['gwl']].plot(ax=ax, color=['#1f77b4'])
        i = 0
        for name, group in _df.groupby('group'):
            group.set_index('time')[[forecast_col]].plot(ax=ax, legend=i == 0, color=['#ff7f0e99'])
            i += 1
    else:
        _df = predictions_df[
            (predictions_df.index.get_level_values('proj_id') == proj_id) &
            (predictions_df.index.get_level_values('horizon') == horizon)
        ].droplevel(axis=0, level=[0, 2])
        _df[['gwl', forecast_col]].plot(ax=ax)
    if confidence:
        ax.fill_between(_df.index.values, _df[confidence[0]], _df[confidence[1]], color='orange', alpha=.33)

    ax.set(
        title=f"well id: {proj_id}",
        ylabel='gwl [m (asl)]',
    )
    return fig, ax

# Functions used in the preprocessing ################################################################################################
# Function to flag jumps in consecutive gwlevels
def detect_jumps(group, column, threshold):
    # calculate diff for each group
    x_diff = group[column].diff().abs()
    
    # calculate mean diff
    x_diff_mean = x_diff.mean()
    
    group['is_jump'] = x_diff > threshold*x_diff_mean
    return group

# Test dataset for detect jumps function
# df = train_df[train_df['proj_id'].isin(['SH_10L54086004', 'BY_25170'])]
# df = df.groupby('proj_id').head(10)
# df['gwl_diff'] = df.groupby('proj_id')['gwl'].diff()
# df['gwl_diff_mean'] = df.groupby('proj_id')['gwl_diff'].transform('mean') # mean gives just two values, 
# need to be transformed to give them correctly back to the df
# df.groupby('proj_id').apply(detect_jumps, column = 'gwl', threshold = 50)

# Interpolate groundwater levels up to 4 weeks (default limit = 4)
def interpolate_gwl(dataframe, interpolation_column, resample_freq='7D', limit=4):
    groups = []
    for well_id, group in dataframe.groupby('proj_id'):
        _df = group.set_index('time')[interpolation_column].resample(resample_freq).asfreq().interpolate(method='linear', limit=limit).reset_index().copy()
        for col in group.columns.difference([interpolation_column, 'time']):  # Retain other columns
            _df[col] = group[col].iloc[0]
        _df['proj_id'] = well_id
        groups.append(_df)
    result_df = pd.concat(groups).reset_index(drop=True)
    return result_df

# Test for linear interpolation:
# To perform the linear interpolation we first need to add NAs for the gaps 
# filtered_train[filtered_train['time_diff']=='28 days']
# test = filtered_train[(filtered_train['proj_id']=='BB_27390101') & (filtered_train['time']<='2003-12-07')].tail(5)
# test
# check after performing the interpolation
# filtered_train[(filtered_train['proj_id']=='BB_27390101') & (filtered_train['time']<='2003-12-07')].tail(5)
# interpolate_gwl(test, 'gwl')