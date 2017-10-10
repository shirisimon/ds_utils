from __future__ import division
import matplotlib.pylab as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import seaborn
from datetime import timedelta

#----------------------------------------------------------------------------------------------------------------------#
# Basic
class Basic():
    def __init__(self):
        pass

    def plot_categorical_distribution(self, df, column, normed=False, ax=None, color=plt.cm.Dark2(0)):
        if normed:
            s = df[column].value_counts() / df.shape[0]
            s.plot(kind='bar', alpha=.5, title=column, ax=ax, color=color);
        else:
            df[column].value_counts().plot(kind='bar', alpha=.5, title=column, ax=ax, color=color)

    def plot_numeric_distribution(self, df, column, normed=False, ax=None, color=plt.cm.Dark2(0)):
        if normed:
            s = df[column].dropna()
            w = np.ones_like(s) / len(s)
            s.plot(kind='hist', weights=w, alpha=.5, title=column, ax=ax, color=color)
        else:
            df[column].plot(kind='hist', alpha=.5, title=column, ax=ax, color=color)



    def plot_multiple_distributions(self, df, column_names='all', normed=False):
        """
        Plot the distributions of all numeric columns (dtypes: float, int) and the categorical columns
        (dtype: object) with cardinality (i.e. unique values) < 20
        For categorical columns with cardinality > 20, map the values to IDs (dtype = int)
        :param df: main (training set) DataFrame
        :param columns_names: default 'all', otherwise specify list of column names
        :param normed: in percentages
        :return:
        """
        if column_names != 'all':
            df=df[column_names]
        fig, axes = plt.subplots(nrows=int(df.columns.shape[0] / 3), ncols=3, figsize=(20, 2 * df.columns.shape[0]))
        ax_arr = axes.flatten(); i = 0
        for col in df.select_dtypes(include=['float64', 'int64']):
            self.plot_numeric_distribution(df, column=col, normed=normed, ax=ax_arr[i]); i = i + 1
        for col in df.select_dtypes(include=['object']):
            if df[col].unique().shape[0] < 20:
                self.plot_categorical_distribution(df, column=col, normed=normed, ax=ax_arr[i]); i = i + 1
        while i != ax_arr.shape[0]:
            ax_arr[i].set_visible(False); i = i + 1


    def plot_multiple_distributions_overlay(self, df, split_column, column_names='all', normed=False):
        """
        plot multiple distributions by split column (e.g. target column, train-test split column)
        For split column with cardinality > 3 it is recommended to iterate on each label vs. all
        :param df: main (training set) DataFrame
        :param split_column: String (Column name the column by which different distributions will be plotted
        :param columns_names: default 'all', otherwise specify list of column names
        :param normed: in percentages (usable in train-test split)
        :return:
        """
        if column_names != 'all':
            df = df[column_names]
        fig, axes = plt.subplots(nrows=int(df.columns.shape[0] / 3), ncols=3, figsize=(20, 2 * df.columns.shape[0]))
        ax_arr = axes.flatten();

        colormap = plt.cm.Dark2
        colors = [colormap(l) for l, _ in enumerate(df[split_column].unique())]
        legends = []
        for c, label in enumerate(df[split_column].unique()):
            label_df = df[df[split_column] == label];
            i = 0
            legends.append(mpatches.Patch(color=colors[c], label=label))
            for col in label_df.drop(labels=split_column, axis=1).select_dtypes(include=['float64', 'int64']):
                self.plot_numeric_distribution(label_df, column=col, normed=normed, ax=ax_arr[i], color=colors[c])
                i = i + 1
            for col in label_df.drop(labels=split_column, axis=1).select_dtypes(include=['object']):
                if label_df[col].unique().shape[0] < 20:
                    self.plot_categorical_distribution(label_df, column=col, normed=normed, ax=ax_arr[i], color=colors[c])
                    i = i + 1
        ax_arr[i - 1].legend(handles=legends, loc='lower right', fontsize=14)
        while i != ax_arr.shape[0]:
            ax_arr[i].set_visible(False)
            i = i + 1


    def main_context_maching_rate(self, main_df, context_df, key_column):
        return np.intersect1d(main_df[key_column].unique(), context_df[key_column].unique())

#----------------------------------------------------------------------------------------------------------------------#
# TS
class TimeSeries():
    def __init__(self):
        pass

    def add_ktw(self, row, key_column, date_column, time_resolution, offset):
        return {'Key': row[key_column], 'End': row[date_column],
                'Start': row[date_column] - np.timedelta64(offset, time_resolution)}

    def add_kts(self, row, context_df, key_column, date_column):
        key_data = context_df[context_df[key_column] == row['keyedTimeWindow']['Key']]
        key_TWslice = key_data[(key_data[date_column]<row['keyedTimeWindow']['End']) &
                               (key_data[date_column]>row['keyedTimeWindow']['Start'])]
        try:
            time_delta = key_TWslice[date_column].max() - key_TWslice[date_column]
            key_TWslice = key_TWslice.assign(timeDelta=time_delta.values)
            key_TWslice = key_TWslice.drop(date_column, axis=1)
            return dict(key_TWslice)
        except: # kts is empty
            return dict()


    def plot_categorical_distribution_overtime(self, df, var_column, date_column, time_resolution, normed=False):
        """
        Plot the value counts over time for every category in the column (if number of categories < 20).
        For the target column distribution, enable detecting whether a temporal train-test split fits
        :param df: main (training set) DataFrame
        :param var_column: String. the categorical column name
        :param date_column: date column (dtype: datetime64[ns])
        :param time_resolution: string. E.g. '2D' (two days), '1M' (one month)
        :param normed: normalize at the time stamp level. Visualize categories differences over time better
        :return:
        """
        if df[var_column].unique().shape[0] < 20:
            gt = df.groupby([var_column, date_column]).count()
            gt = gt.iloc[:, 1]
            gt = gt.unstack(level=0).resample(time_resolution).sum()
            if normed:
                gt = gt.apply(lambda row: row / row.sum(), axis=1)
            gt.plot.area(figsize=(20, 6), stacked=True, colormap='Dark2', title=var_column);


    def plot_numeric_fixed(self, df, var_column, date_column, time_resolution='1d', tw=False, key_column=None,
                        key_ids=None, reference_date=None, offset=None, color=plt.cm.Dark2(0), ax=None):
        """
        Plot numeric time series (enable plotting the averaged time series across keys, if key_column is specified).
        This function plot time series relative to fixed date (the reference_date)
        :param df: main (training set) DataFrame
        :param var_column: String. Numeric column name to plot
        :param date_column: String. Date column name
        :param time_resolution: String. E.g. '1s' for 1 sec, '2w' fro two weeks
        :param tw: whether to plot the ts within a time window
        :param reference_date: pandas datetime timestamp
        :param offset: time stamps number (multiplied by time_resolution)
        :param key_column: partition by key column (generate averaged time series across keys)
        :param key_ids: specify only various keys
        :param color:
        :return:
        """
        if tw:
            assert(reference_date)
            assert(offset)
            df = df[(df[date_column] < reference_date) &
                    (df[date_column] > reference_date - np.timedelta64(offset, time_resolution))]
        if key_column:
            if key_ids:
                keys_iter = iter(key_ids)
            else:
                keys_iter = iter(df[key_column].unique())
            lines = pd.DataFrame()
            for key in keys_iter:
                g = df[df[key_column]==key].groupby([date_column]).mean()
                g = g.resample(time_resolution).mean()[[var_column]]
                lines = lines.join(g, how='outer', rsuffix=str(key))
            lines_avg = lines.mean(axis=1)
            lines_std = lines.std(axis=1)
            if ax is None:
                plt.figure(figsize=(20,6))
                plt.title(var_column)
                plt.plot(lines_avg.index, lines_avg, color=color);
                plt.fill_between(lines_std.index, lines_avg - lines_std, lines_avg + lines_std,
                                 color=color, alpha=0.2)
            else:
                ax.plot(lines_avg.index, lines_avg, color=color)
                ax.fill_between(lines_std.index, lines_avg - lines_std, lines_avg + lines_std,
                                 color=color, alpha=0.2)
                plt.title(var_column)
        else:
            g = df.groupby([date_column]).mean()
            g = g.resample(time_resolution).mean()[[var_column]]
            if ax is None:
                g.plot(kind='line', figsize=(20,6), color=color, title=var_column)
            else:
                g.plot(kind='line', color=color, title=var_column, ax=ax)


    def plot_numeric_relative(self, main_df, context_df, var_column, date_column, key_column, key_ids=None,
                                 time_resolution='1d', offset=None, color=plt.cm.Dark2(0), ax=None):
        """
        Plot numeric time series relative to the end of the time window.
        Warning: this functions takes time to run as it requires Time series indexing and slicing .
        It is recommended to downsample before calling it.
        :param main_df: main (training set) DataFrame
        :param context_df: Time Series context DataFrame
        :param var_column: String. Numeric column Name to plot (from the context_df)
        :param date_column: String. Date column name
        :param key_column: String. The name of the column to partition the time series by
        (generate averaged time series across the key)
        :param keys_ids: list. Specify specific time series to plot by key
        :param time_resolution: String. E.g. '1s' for 1 sec, '2w' fro two weeks
        :param offset: time stamps number (multiplied by time_resolution)
        :return:
        """

        ktw = main_df.astype(object).apply(self.add_ktw, args=(key_column, date_column, time_resolution, offset), axis=1).values
        main_df = main_df.assign(keyedTimeWindow=list(ktw))
        its = main_df.astype(object).apply(self.add_kts, args=(context_df, key_column, date_column), axis=1).values
        main_df = main_df.assign(indexedTimeSeries=list(its))

        if key_ids:
            main_df = main_df[main_df[key_column].isin(key_ids)]
        lines = pd.DataFrame()
        for _, row in main_df.iterrows():
            try:
                g = pd.DataFrame(row['indexedTimeSeries'])
                g = g.set_index('timeDelta')
                g = g.resample(time_resolution).mean()[[var_column]]
                lines = lines.join(g, how='outer', rsuffix='_' + str(row[key_column]))
            except:  # skip empty ts
                continue
        lines_avg = lines.sort_index(ascending=False).mean(axis=1)
        #lines_std = lines.sort_index(ascending=False).std(axis=1)
        if ax is None:
            plt.figure(figsize=(20,6))
            plt.plot(lines_avg.index, lines_avg, color=color)
            plt.gca().invert_xaxis()
        else:
            ax.plot(lines_avg.index, lines_avg, color=color)
        # plt.fillbetween is not supported for TimeDeltaIndex
        plt.title(var_column)


    def plot_numeric_overlay(self, main_df, split_column, var_column, date_column, type='fixed', tw=False, reference_date=None,
                                context_df=None, key_column=None, key_ids=None, time_resolution='1d', offset=None):
        """
        Plot time series by split column (e.g. target column, train-test split column)
        For split column with cardinality > 3 it is recommended to iterate on each label vs. all
        Warning: this functions takes time to run when choosing type = 'relative' (Since it includes indexing of the Time Series).
        Down-sample before calling it!
        :param main_df: main (training set) DataFrame
        :param split_column: column name to to split the main_df by.
        :param var_column: String. Column Name to plot (from the context_df)
        :param date_column: String. Date column name
        :param key_column: String. The name of the column to partition the time series by
        (generate averaged time series across the key)
        :param keys_ids: list. Specify specific time series to plot by key
        :param time_resolution: String. E.g. '1s' for 1 sec, '2w' fro two weeks
        :param offset: time stamps number (multiplied by time_resolution)
        :param tw: Boolean. Whether to slice the TS to TW
        :param reference_date: fixed date which will be the end of the the TW
        :param context_df: Time Series context DataFrame
        :param type: 'fixed' or 'relative'. if relative, context_df should be provided.
        :return:
        """
        colormap = plt.cm.Dark2
        colors = [colormap(l) for l, _ in enumerate(main_df[split_column].unique())]
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,6))
        legends = []
        for c, label in enumerate(main_df[split_column].unique()):
            label_df = main_df[main_df[split_column] == label]
            legends.append(mpatches.Patch(color=colors[c], label=label))
            if type == 'fixed':
                self.plot_numeric_fixed(label_df, var_column, date_column, time_resolution=time_resolution,
                                      tw=tw, key_column=key_column, key_ids=key_ids,
                                      reference_date=reference_date, offset=offset, color=colors[c], ax=ax)
            else: # type = 'relative'
                self.plot_numeric_relative(label_df, context_df, var_column, date_column, key_column,
                                         key_ids=key_ids, time_resolution=time_resolution, offset=offset,
                                         color=colors[c], ax=ax)
        if type == 'relative':
            plt.gca().invert_xaxis()
        ax.legend(handles=legends, loc='lower right', fontsize=14)


    def plot_multiple_numeric(self, main_df, split_column, date_column, type='fixed', tw=False, reference_date=None,
                                    context_df=None, key_column=None, key_ids=None, time_resolution='1d', offset=None,
                                    overlay = False, column_names = 'all'):
        # TODO: efficient (if type is relative, index once and iterate on columns)
        pass


    def plot_categorical_column_time_series(self):
        # TODO: search on the web
        pass

#----------------------------------------------------------------------------------------------------------------------#
# TEXT
class Text():
    def __init__(self):
        pass

    def words_count(self):
        pass

    def plot_tfidf_by_class(self):
        pass


#----------------------------------------------------------------------------------------------------------------------#




