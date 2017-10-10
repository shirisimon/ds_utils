from __future__ import division
import numpy as np
import matplotlib.pylab as plt


def _convert_top_at(top_at, sample_size):
    """
    convert the percentage at the top of population to the representing number of records at the top of sample
    :param top_at: float. the percentage at the top of population
    :param population_ratio: float. the ratio of positives (in percentage) at the population
    :param population_size: int
    :param sample_size: int. number of rows in predicted_data
    :return: int. top of population in number
    """
    return  top_at * sample_size / 100


def _get_primary_measures(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
                       population_size, score_type = 'probability'):
    """
    util function to get all params for metrics evaluation
    :param predicted_data: pandas.DataFrame. data with prediction scores
    :param target_column:  string.
    :param positive_label: string.
    :param top_perc_of_population: float. the percentage at the top of population for which to calc the metric
    :param population_ratio: float. the ratio of positives (in percentage) at the population
    :param population size: int
    :param score_type: string. 'probability' in the predicted_data, or other metric if a score was
    calculated and added manually to the predicted_data object
    :return:
    pos_at_top_population:
    neg_at_top_population:
    pos_at_all_sample:
    pos_sampling_factor:
    """
    predicted_data_ranked = predicted_data.sort_values(by=score_type + '_' + positive_label, ascending=False)
    top_of_sample = _convert_top_at(top_perc_of_population, predicted_data.shape[0])
    predicted_data_at_top_of_sample = predicted_data_ranked.head(int(top_of_sample))

    pos_at_top_sample = \
        predicted_data_at_top_of_sample[predicted_data_at_top_of_sample[target_column] == positive_label].shape[0]  # tp+fn
    neg_at_top_sample = \
        predicted_data_at_top_of_sample[predicted_data_at_top_of_sample[target_column] != positive_label].shape[0]  # tp+fn
    pos_at_all_sample = predicted_data_ranked[predicted_data_ranked[target_column] == positive_label]
    neg_at_all_sample = predicted_data_ranked[predicted_data_ranked[target_column] != positive_label]
    # tpr = pos_at_top_sample / pos_at_all_sample
    # fpr = neg_at_top_sample / neg_at_all_sample
    pos_sampling_factor = (population_ratio / 100) * population_size / pos_at_all_sample
    neg_sampling_factor = (1 - (population_ratio / 100)) * population_size / neg_at_all_sample
    pos_at_top_population = pos_at_top_sample * pos_sampling_factor
    neg_at_top_population = neg_at_top_sample * neg_sampling_factor
    return pos_at_top_population, neg_at_top_population, pos_at_all_sample, pos_sampling_factor


def _get_secondary_measures(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio_primary,
                            population_ratio_secondary, primary_score_cutoff, population_size, score_type = 'probability'):
    """

    :param predicted_data: pandas.DataFrame. with prediction scores for primary and secondary models
        are joined before calling this function. Records not included in the secondary sample, should have the value "None" under the secondary labels columns
     :param target_column:  string. target column of a 2nd level model
    :param positive_label: string. positive label belongs to a target column of a 2nd level model
    :param top_perc_of_population: float. the percentage at the top of population for which to calc the metric.
    :param population_ratio: float. the ratio of the positive label (in percentage) at the population.
    :param population_size: int. number of records \ units in the entire population
    :param score_type: string. 'probability' as calculated in the predicted_data, or other metric if a score was
        calculated and added manually
    :return:
    """
    predictions_null = predicted_data.copy()
    predictions_null.loc[predictions_null.probability_Y < primary_score_cutoff,  score_type+'_'+positive_label] = 0.0  # nullify all examples where PREC < 1
    predicted_data_ranked= predictions_null.sort_values(by=score_type + '_' + positive_label, ascending=False)
    top_of_sample = _convert_top_at(top_perc_of_population, predicted_data.shape[0])
    predicted_data_at_top_of_sample = predicted_data_ranked.head(int(top_of_sample))

    pos_at_top_sample = \
        predicted_data_at_top_of_sample[predicted_data_at_top_of_sample[target_column] == positive_label].shape[0]  # tp+fn
    neg_none_at_top_sample = \
        predicted_data_at_top_of_sample[predicted_data_at_top_of_sample[target_column] == 'None'].shape[0]
    neg_othr_at_top_sample = \
        predicted_data_at_top_of_sample[predicted_data_at_top_of_sample[target_column] != positive_label].shape[0] \
        - neg_none_at_top_sample  # fp+tn

    pos_at_all_sample = predicted_data_ranked[predicted_data_ranked[target_column] == positive_label].shape[0]
    neg_none_at_all_sample = predicted_data_ranked[predicted_data_ranked[target_column] == 'None'].shape[0]
    neg_othr_at_all_sample = predicted_data_ranked[predicted_data_ranked[target_column] != positive_label].shape[0] \
                             - neg_none_at_all_sample

    pos_sampling_factor = (population_ratio_secondary / 100) * population_size / pos_at_all_sample
    neg_othr_sampling_factor = ((population_ratio_primary - population_ratio_secondary) / 100) * population_size / neg_othr_at_all_sample
    neg_none_sampling_factor = (1 - (population_ratio_primary / 100)) * population_size / neg_none_at_all_sample

    pos_at_top_population = pos_at_top_sample * pos_sampling_factor
    neg_othr_at_top_population = neg_othr_at_top_sample * neg_othr_sampling_factor
    neg_none_at_top_population = neg_none_at_top_sample * neg_none_sampling_factor
    neg_at_top_population = neg_none_at_top_population + neg_othr_at_top_population

    return pos_at_top_population, neg_at_top_population, pos_at_all_sample, pos_sampling_factor


def adjusted_precision(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
                       population_size, score_type='probability'):
    """
    calculate from predictions of differently distributed sample (with respect to the population), the adjusted precision at the top of population
    :param predicted_data: pandas.DataFrame. data with prediction scores
    :param target_column:  string.
    :param positive_label: string.
    :param top_perc_of_population: float. the percentage at the top of population for which to calc the metric
    :param population_ratio: float. the ratio of positives (in percentage) at the population
    :param population size: int
    :param score_type: string. 'probability' as calculated in the predicted_data, or other metric if a score was
    calculated and added manually to the predicted_data object
    :return: float ([0,1]). adjusted precision at the top of population
    """
    pos_at_top_population, neg_at_top_population, _, _ = _get_primary_measures(
        predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
        population_size, score_type)
    prec_at_top_population = pos_at_top_population / (pos_at_top_population + neg_at_top_population)
    return prec_at_top_population


def adjusted_recall(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
                       population_size, score_type = 'probability'):
    """
    calculate from predictions of differently distributed sample (with respect to the population), the adjusted recall at the top of population
    :param predicted_data: pandas.DataFrame. data with prediction scores
    :param target_column:  string.
    :param positive_label: string.
    :param top_perc_of_population: float. the percentage at the top of population for which to calc the metric
    :param population_ratio: float. the ratio of positives (in percentage) at the population
    :param population size: int
    :param score_type: string. 'probability' as calculated in the predicted_data, or other metric if a score was
    calculated and added manually to the predicted_data object
    :return: float ([0,1]). adjusted recall at the top of population
    """
    pos_at_top_population, neg_at_top_population, pos_at_all_sample, pos_sampling_factor = _get_primary_measures(
        predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
        population_size, score_type)
    recl_at_top_population = pos_at_top_population / (pos_at_all_sample * pos_sampling_factor)
    return recl_at_top_population


def adjusted_lift(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
                       population_size, score_type = 'probability'):
    """
    calculate from predictions of differently distributed sample (with respect to the population), the adjusted lift at the top of population
    :param predicted_data: pandas.DataFrame. data with prediction scores
    :param target_column:  string.
    :param positive_label: string.
    :param top_perc_of_population: float. the percentage at the top of population for which to calc the metric
    :param population_ratio: float. the ratio of positives (in percentage) at the population
    :param population size: int
    :param score_type: string. 'probability' as calculated in the predicted_data, or other metric if a score was
    calculated and added manually to the predicted_data object
    :return: float ([0,1]). adjusted recall at the top of population
    """
    pos_at_top_population, neg_at_top_population, pos_at_all_sample, pos_sampling_factor = _get_primary_measures(
        predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
        population_size, score_type)
    prec_at_top_population = pos_at_top_population / (pos_at_top_population + neg_at_top_population)
    lift_at_top_population = prec_at_top_population / (population_ratio / 100)
    return lift_at_top_population

def normalize_sample_size_to_population(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
                       population_size, score_type = 'probability'):
    """
    normalize the number of records \ units in the sample to the number of records in population
    :return: int. number of records \ units in population
    """
    pos_at_top_population, neg_at_top_population, _, _ = _get_primary_measures(
        predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
        population_size, score_type)
    return  pos_at_top_population + neg_at_top_population


def adjusted_metrics(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
                       population_size, score_type = 'probability'):
    pos_at_top_population, neg_at_top_population, pos_at_all_sample, pos_sampling_factor = _get_primary_measures(
        predicted_data, target_column, positive_label, top_perc_of_population, population_ratio,
        population_size, score_type)
    prec_at_top_population = pos_at_top_population / (pos_at_top_population + neg_at_top_population)
    recl_at_top_population = pos_at_top_population / (pos_at_all_sample * pos_sampling_factor)
    lift_at_top_population = prec_at_top_population / (population_ratio / 100)
    return prec_at_top_population, recl_at_top_population, lift_at_top_population


def adjusted_precision_secondary(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio_primary,
                            population_ratio_secondary, primary_score_cutoff, population_size, score_type = 'probability'):
    """
    calculate the adjusted precision at the top of population from predictions of differently distributed sample (with respect to the population distribution)
        The target was trained as a second level model, using only a subsample for which the positive label of the 1st model level is true.
    :param predicted_data: pandas.DataFrame. with all predictions from primary and secondary models
        are joined before calling this function. Records not included in the secondary sample, should have the value "None" under the secondary labels columns
    :param target_column:  string. target column of a 2nd level model
    :param positive_label: string. positive label belongs to a target column of a 2nd level model
    :param top_perc_of_population: float. the percentage at the top of population for which to calc the metric.
    :param population_ratio: float. the ratio of the positive label (in percentage) at the population.
    :param population_size: int. number of records \ units in the entire population
    :param score_type: string. 'probability' as calculated in the predicted_data, or other metric if a score was
        calculated and added manually
    :return: float ([0,1]). adjusted precision at the top of population
    """
    pos_at_top_population, neg_at_top_population, _, _ = _get_secondary_measures(predicted_data, target_column,
        positive_label, top_perc_of_population, population_ratio_primary, population_ratio_secondary,
        primary_score_cutoff, population_size, score_type)
    prec_at_top_population = pos_at_top_population / (pos_at_top_population + neg_at_top_population)
    return prec_at_top_population


def adjusted_recall_secondary(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio_primary,
                            population_ratio_secondary, primary_score_cutoff, population_size, score_type = 'probability'):
    """
    calculate the adjusted recall at the top of population from predictions of differently distributed sample (with respect to the population distribution)
        The target was trained as a second level model, using only a subsample for which the positive label of the 1st model level is true.
    :param predicted_data: pandas.DataFrame. with all predictions from primary and secondary models
        are joined before calling this function. Records not included in the secondary sample, should have the value "None" under the secondary labels columns
    :param target_column:  string. target column of a 2nd level model
    :param positive_label: string. positive label belongs to a target column of a 2nd level model
    :param top_perc_of_population: float. the percentage at the top of population for which to calc the metric.
    :param population_ratio: float. the ratio of the positive label (in percentage) at the population.
    :param population_size: int. number of records \ units in the entire population
    :param score_type: string. 'probability' as calculated in the predicted_data, or other metric if a score was
        calculated and added manually
    :return: float ([0,1]). adjusted recall at the top of population
    """
    pos_at_top_population, neg_at_top_population, pos_at_all_sample, pos_sampling_factor = _get_secondary_measures(predicted_data, target_column,
        positive_label, top_perc_of_population, population_ratio_primary, population_ratio_secondary,
        primary_score_cutoff, population_size, score_type)
    recl_at_top_population = pos_at_top_population / (pos_at_all_sample * pos_sampling_factor)
    return recl_at_top_population


def adjusted_lift_secondary(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio_primary,
                            population_ratio_secondary, primary_score_cutoff, population_size, score_type = 'probability'):
    """
    calculate the adjusted precision at the top of population from predictions of differently distributed sample (with respect to the population distribution)
        The target was trained as a second level model, using only a subsample for which the positive label of the 1st model level is true.
    :param predicted_data:pandas.DataFrame. with all predictions from primary and secondary models
        are joined before calling this function. Records not included in the secondary sample, should have the value "None" under the secondary labels columns
    :param target_column:  string. target column of a 2nd level model
    :param positive_label: string. positive label belongs to a target column of a 2nd level model
    :param top_perc_of_population: float. the percentage at the top of population for which to calc the metric.
    :param population_ratio: float. the ratio of the positive label (in percentage) at the population.
    :param population_size: int. number of records \ units in the entire population
    :param score_type: string. 'probability' as calculated in the predicted_data, or other metric if a score was
        calculated and added manually
    :return: float ([0,1]). adjusted precision at the top of population
    """
    pos_at_top_population, neg_at_top_population, pos_at_all_sample, pos_sampling_factor = _get_secondary_measures(
        predicted_data, target_column, positive_label, top_perc_of_population, population_ratio_primary,
        population_ratio_secondary, primary_score_cutoff, population_size, score_type)
    prec_at_top_population = pos_at_top_population / (pos_at_top_population + neg_at_top_population)
    lift_at_top_population = prec_at_top_population / (population_ratio_secondary / 100)
    return lift_at_top_population


def adjusted_metrics_secondary(predicted_data, target_column, positive_label, top_perc_of_population, population_ratio_primary,
        population_ratio_secondary, primary_score_cutoff, population_size, score_type = 'probability'):
    pos_at_top_population, neg_at_top_population, pos_at_all_sample, pos_sampling_factor = _get_secondary_measures(
        predicted_data, target_column, positive_label, top_perc_of_population, population_ratio_primary,
        population_ratio_secondary, primary_score_cutoff, population_size, score_type)
    prec_at_top_population = pos_at_top_population / (pos_at_top_population + neg_at_top_population)
    recl_at_top_population = pos_at_top_population / (pos_at_all_sample * pos_sampling_factor)
    lift_at_top_population = prec_at_top_population / (population_ratio_secondary / 100)
    return prec_at_top_population, recl_at_top_population, lift_at_top_population


def normalize_sample_size_to_population_secondary(predicted_data, target_column, positive_label, top_perc_of_population,
        population_ratio_primary, population_ratio_secondary, primary_score_cutoff, population_size, score_type = 'probability'):
    """
    normalize the number of records \ units in the sample to the number of records in population
    :return: int. number of records \ units in population
    """
    pos_at_top_population, neg_at_top_population, _, _ = _get_secondary_measures(predicted_data, target_column,
        positive_label, top_perc_of_population, population_ratio_primary, population_ratio_secondary,
        primary_score_cutoff, population_size, score_type)
    return pos_at_top_population + neg_at_top_population



def plot_adjusted_metrics(predicted_data, params, top_at=np.arange(0, 100, 50), secondary=False):
    """
    :param predicted_data:
    :param params:
    :param top_cst:
    :param secondary:
    :param max_or_mean:
    :param pop_ratio_primary:
    :return:
    """
    def get_dicts(positive_label, top_num_at_population, prec, recl, lift):
        line_dict[positive_label] = ax[0].plot(top_num_at_population, prec, label=positive_label, lw=3)
        ax[1].plot(top_num_at_population, recl, label=positive_label, lw=3)
        ax[2].plot(top_num_at_population, lift, label=positive_label, lw=3)
        prec_dict[positive_label] = zip(top_num_at_population, prec)
        recl_dict[positive_label] = zip(top_num_at_population, recl)
        lift_dict[positive_label] = zip(top_num_at_population, lift)
        return line_dict, prec_dict, recl_dict, lift_dict

    f, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    prec_dict, recl_dict, line_dict = [{} for i in range(3)]
    if secondary:
        for i, (positive_label, target_column, population_ratio_primary,
                population_ratio_secondary, primary_score_cutoff, population_size) in enumerate(params):
            prec, recl, lift = zip(*[adjusted_metrics_secondary(
                predicted_data, target_column, positive_label, c, population_ratio_primary,
                population_ratio_secondary, primary_score_cutoff, population_size, score_type='probability')
                for c in top_at])
            top_num_at_population = zip(*[normalize_sample_size_to_population_secondary(predicted_data,
                target_column, positive_label, c, population_ratio_primary, population_ratio_secondary,
                primary_score_cutoff, population_size, score_type='probability') for c in top_at])
            line_dict, prec_dict, recl_dict, lift_dict = get_dicts(positive_label, top_num_at_population, prec, recl, lift)
            # chances
            color = line_dict[positive_label][0].get_color()
            ax[0].axhline(y=population_ratio_secondary, linestyle=':', color=color, zorder=0)                                                   # prec_chance
            recl_chance = (np.array(top_num_at_population)*population_ratio_secondary) / (population_ratio_secondary / 100 * population_size)   # recl_chance
            ax[1].plot(top_num_at_population, recl_chance, linestyle=':', color=color)                                                          # lift chance
            ax[2].plot(y=1, linestyle=':', color='lightgray')
    else: # primary
        for i, (positive_label, target_column, population_ratio, population_size) in enumerate(params):
            prec, recl, lift = zip(*[adjusted_metrics(predicted_data, target_column, positive_label, c,
                population_ratio, population_size, score_type='probability') for c in top_at])
            top_num_at_population = zip(*[normalize_sample_size_to_population_secondary(predicted_data,
                target_column, positive_label, c, population_ratio, population_size, score_type='probability')
                for c in top_at])
            line_dict, prec_dict, recl_dict, lift_dict = get_dicts(positive_label, top_num_at_population, prec, recl, lift)
            # chances
            color = line_dict[positive_label][0].get_color()
            ax[0].axhline(y=population_ratio, linestyle=':', color=color, zorder=0)                                          # prec_chance
            recl_chance = (np.array(top_num_at_population) * population_ratio) / (population_ratio / 100 * population_size)  # recl_chance
            ax[1].plot(top_num_at_population, recl_chance, linestyle=':', color=color)                                       # lift chance
            ax[2].plot(y=1, linestyle=':', color='lightgray')

    ax[0].set_ylabel('Precision', fontweight='bold')
    ax[1].set_ylabel('Recall', fontweight='bold')
    ax[2].set_ylabel('Lift', fontweight='bold')
    ax[2].legend(loc='best')
    f.set_title(target_column, fontweight='bold')

    for i in range(3):
        ax[i].set_ylim(bottom=0)
        ax[i].set_xlabel('Top of population (#)', fontweight='bold')
        ax[i].grid(False)
        ax[i].autoscale(enable=True, axis='x', tight=True)

    return prec_dict, recl_dict, lift_dict, line_dict, top_num_at_population, ax

