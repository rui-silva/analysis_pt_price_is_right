import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import pandas as pd
import datetime as dt
import calmap


PT_MONTH_LABELS = [
    'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out',
    'Nov', 'Dez'
]
PT_DAY_LABELS = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom']


def aired_dates_to_pandas():
    df = []
    with open('./data/aired_dates.txt') as f:
        lines = f.readlines()
        for l in lines:
            l = l.rstrip()
            df.append(dt.datetime.strptime(l, '%Y-%m-%d'))

    return pd.DatetimeIndex(df)


def prize_data_to_pandas():
    df = pd.read_csv('./data/prizes.csv', parse_dates=['date', 'repeated_from'])
    df = df.set_index('date').sort_index()

    repeated = ~pd.isna(df['repeated_from'])
    repeated = repeated.astype(int)
    df['repeated'] = repeated

    # mark wins
    diff = df['value'] - df['bet']
    win_s = (diff >= 0) & (df['margin'] >= diff)
    df['win'] = 0
    df.loc[win_s, 'win'] = 1

    df['diff'] = diff

    return df


def filter_non_repeated(prizes_df):
    return prizes_df[prizes_df['repeated'] == 0]


def filter_year(prizes_df, year):
    return prizes_df[prizes_df.index.year == year]


def data_stats(prizes_df, aired_series):
    stats = prizes_df.groupby(prizes_df.index.year).agg({
        'value': ['count'],
    })
    print(stats)
    print(f'Total episodes: {stats.sum()}')

    for year in [2016, 2017, 2018, 2019, 2020]:
        prizes_count = len(prizes_df[prizes_df.index.year == year])
        aired_count = len(aired_series[aired_series.year == year])
        print(
            f'{year} - {prizes_count}/{aired_count} = {prizes_count/aired_count}'
        )


def yearly_stats(prizes_df):
    prizes_df = filter_non_repeated(prizes_df)
    stats = prizes_df.groupby(prizes_df.index.year).agg({
        'value': ['mean', 'sem', 'min', 'max'],
        'win': ['mean', 'sem', 'sum', 'count']
    })
    print(stats)


def day_of_week_stats(prizes_df):
    prizes_df = filter_non_repeated(prizes_df)
    stats = prizes_df.groupby(prizes_df.index.weekday).agg({
        'value': ['mean', 'sem', 'min', 'max'],
        'win': ['mean', 'sem', 'sum', 'count']
    })
    print(stats)


def win_loss_year_plot(prizes_df):
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(nrows=5,
                             ncols=1,
                             squeeze=False,
                             figsize=(10, 12.5),
                             subplot_kw={},
                             gridspec_kw={})
    axes = axes.T[0]

    for year_idx, year in enumerate([2016, 2017, 2018, 2019, 2020]):
        ax = axes[year_idx]
        year_df = filter_year(prizes_df, year)

        non_repeated_df = filter_non_repeated(year_df)
        win_pct = (non_repeated_df['win'] == 1).sum() / len(non_repeated_df)
        loss_pct = (non_repeated_df['win'] == 0).sum() / len(non_repeated_df)

        results_df = year_df['win'].resample('D').asfreq(fill_value=np.nan)

        text_df = year_df['repeated'].resample('D').asfreq(fill_value='')
        text_df[text_df == 0] = ''
        text_df[text_df == 1] = 'R'
        calmap.yearplot(results_df,
                        year=year,
                        daylabels=PT_DAY_LABELS,
                        monthlabels=PT_MONTH_LABELS,
                        how=None,
                        ax=ax,
                        vmin=0,
                        vmax=1,
                        cmap=ListedColormap(['red', 'green']),
                        text=text_df)
        ax.set_title(
            f'Montras ganhas e perdidas no Preço Certo em Euros ({year})')
        leg = ax.legend(
            handles=[
                Patch(facecolor='r',
                      edgecolor='r',
                      label=f'Montra perdida {loss_pct:.1%}'),
                Patch(facecolor='g',
                      edgecolor='g',
                      label=f'Montra ganha {win_pct:.1%}'),
            ],
            loc='lower center',
            ncol=4,
            bbox_to_anchor=(0.5, -.65),
            title='Probabilidades (ignorando episódios repetidos R)')
        leg._legend_box.align = 'left'

    plt.tight_layout()
    fig.savefig('./year_win_loss.png')


def win_vs_values_histogram(prizes_df):
    import matplotlib.ticker as mtick

    prizes_df = filter_non_repeated(prizes_df)
    bins = list(range(5000, 30000 + 5000, 5000))

    prizes_df['binned'] = pd.cut(prizes_df['value'], bins, include_lowest=True)

    grouped_df = prizes_df.groupby(prizes_df['binned']).agg(
        {'win': ['mean', 'sem', 'sum', 'count']})

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    grouped_df['win']['mean'].plot(kind='bar', ax=ax)

    # Iterate over the bars and add a text annotation with
    # win-count / total-count
    for idx, label in enumerate(list(grouped_df.index)):
        label_df = grouped_df[grouped_df.index == label]
        win_sum = label_df["win"]["sum"].sum()
        win_count = label_df["win"]["count"].sum()
        text = f'{win_sum}/{win_count}'
        x = ax.patches[idx].get_x() + .25
        y = ax.patches[idx].get_height()
        ax.annotate(text, (x, y),
                    xytext=(0, 15),
                    textcoords='offset points',
                    horizontalalignment='center')

    ax.set_xticklabels([
        '5000 - 10000', '10000 - 15000', '15000-20000', '20000-25000',
        '25000-30000'
    ],
                       rotation=45)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(
        'Percentagem de vitórias para diferentes valores da montra final')
    ax.set_xlabel('Valor da montra (€)')
    ax.set_ylabel('Percentagem de vitórias (%)')

    plt.tight_layout()
    fig.savefig('./win_vs_values_histogram.png')


def value_bet_histograms(prizes_df):
    fig, axes = plt.subplots(nrows=5,
                             ncols=1,
                             squeeze=False,
                             figsize=(7, 10),
                             subplot_kw={},
                             gridspec_kw={})
    axes = axes.T[0]

    prizes_df = filter_non_repeated(prizes_df)

    # Setup bins
    # - find min and max bet/value
    value_and_bets_min = prizes_df[['value', 'bet']].min().min()
    value_and_bets_max = prizes_df[['value', 'bet']].max().max()
    # - round to nearest thousands
    #   - we assume unitary bins, since the density histogram
    #     plots consider the sum of area under the bins = 1.
    #   - we then report the thousands by changing the xticklabels
    min_bin = round(value_and_bets_min / 1000)
    max_bin = round(value_and_bets_max / 1000)
    min_bin, max_bin = int(min_bin), int(max_bin)

    for year_idx, year in enumerate([2016, 2017, 2018, 2019, 2020]):
        ax = axes[year_idx]
        year_df = prizes_df[prizes_df.index.year == year]

        values = year_df['value'].to_numpy() / 1000
        bets = year_df['bet'].to_numpy() / 1000
        bins = np.array(range(min_bin, (max_bin + 1) + 1, 1))
        hresults = ax.hist([values, bets],
                           bins,
                           alpha=0.5,
                           label=['Valor da montra', 'Aposta'],
                           density=True)

        ax.set_xticks(bins)
        ax.set_xticklabels(bins * 1000, rotation=45, fontsize=7)
        max_prob = .18  # measured empirically
        ax.set_yticks(np.linspace(0, max_prob, 10))
        ax.set_yticklabels([f'{n}%' for n in range(0, 18 + 1, 2)], fontsize=8)

        ax.legend(['Valor da montra', 'Aposta'], prop={'size': 6})
        if year_idx == 0:
            ax.set_title(
                f'Distribuição dos valores da montra vs. apostas\n{year}',
                fontsize=9)
        else:
            ax.set_title(f'{year}', fontsize=9)
        ax.set_xlabel('Valor (€)')
        ax.set_ylabel('Ocorrências (%)')
        ax.grid()

    plt.tight_layout()
    fig.savefig('./value_bet_year_histograms.png')


def value_bet_2d_histograms(prizes_df):
    prizes_df = filter_non_repeated(prizes_df)

    # Setup bins
    # - find min and max bet/value
    value_and_bets_min = prizes_df[['value', 'bet']].min().min()
    value_and_bets_max = prizes_df[['value', 'bet']].max().max()
    # - round to nearest thousands
    min_bin = round(value_and_bets_min / 1000) * 1000
    max_bin = round(value_and_bets_max / 1000) * 1000
    min_bin, max_bin = int(min_bin), int(max_bin)

    for year_idx, year in enumerate([2016, 2017, 2018, 2019, 2020]):
        year_df = prizes_df[prizes_df.index.year == year]

        values = year_df['value'].to_numpy()
        bets = year_df['bet'].to_numpy()

        bins = list(range(min_bin, (max_bin + 1000) + 1000, 1000))
        joint = sns.jointplot(x=values,
                              y=bets,
                              marginal_kws=dict(bins=bins, rug=False))

        joint.ax_joint.set_xticks(np.array(bins))
        joint.ax_joint.set_xticklabels(bins, rotation=45)
        joint.ax_joint.set_yticks(np.array(bins))

        joint.fig.suptitle(
            f'Distribuição dos valores da montra vs. apostas {year}')
        joint.ax_joint.set_xlabel('Valor')
        joint.ax_joint.set_ylabel('Aposta')
        joint.ax_joint.grid()
        joint.ax_joint.plot(bins, bins, ':r')
        joint.fig.set_figwidth(7)
        joint.fig.set_figheight(7)
        plt.tight_layout()
        joint.savefig(f'./value_bet_2d_histograms_{year}.png')


def margins_pie(prizes_df):
    prizes_df = filter_non_repeated(prizes_df)
    fig = plt.figure(figsize=(9, 8))

    # We use subplot2grid to have 2 rows of pie charts. 3 on top, 2 at
    # the bottom 
    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)

    axes = [ax1, ax2, ax3, ax4, ax5]
    for year_idx, year in enumerate([2016, 2017, 2018, 2019, 2020]):
        ax = axes[year_idx]
        year_df = prizes_df[prizes_df.index.year == year]
        margins = year_df['margin']
        margins = margins.groupby(margins).count()
        colors = ['red', 'orange', '#F7FF00', 'greenyellow', 'forestgreen']
        ax.pie(margins.to_numpy(),
               labels=margins.index,
               autopct='%1.2f%%',
               colors=colors)
        ax.set_title(f'{year}')

    fig.suptitle('Margens obtidas no Preço Certo ao longo dos anos')
    fig.tight_layout()
    fig.savefig('./margins.png')


def margins_vs_win_prob(prizes_df):
    prizes_df = filter_non_repeated(prizes_df)
    margins = prizes_df['margin']
    print('=== MARGINS VS WIN PROB ===')
    print(
        prizes_df.groupby(margins).agg(
            {'win': ['mean', 'sem', 'sum', 'count']}))

    for year in [2016, 2017, 2018, 2019, 2020]:
        year_df = prizes_df[prizes_df.index.year == year]
        report = year_df.groupby(year_df['margin']).agg(
            {'win': ['mean', 'sum', 'count']})
        print(f'===== {year} =====')
        print(report)


def margins_vs_value(prizes_df):
    prizes_df = filter_non_repeated(prizes_df)

    margins = prizes_df['margin']
    print('=== MARGINS VS VALUES ===')
    print(
        prizes_df.groupby(margins).agg(
            {'value': ['mean', 'sem', 'count', 'min', 'max']}))



def analysis_gender(prizes_df):
    prizes_df = filter_non_repeated(prizes_df)

    prizes_df['blow'] = 0
    prizes_df.loc[prizes_df['diff'] < 0, 'blow'] = 1

    agg = {
        'win': ['mean', 'sem', 'sum', 'count'],
        'value': ['mean', 'sem'],
        'bet': ['mean', 'sem'],
        'margin': ['mean', 'sem'],
        'blow': ['mean', 'sem', 'sum', 'count']
    }

    print(prizes_df.groupby([prizes_df['gender']]).agg(agg))

    print(prizes_df.groupby([prizes_df.index.year, prizes_df['gender']]).agg(agg))

    print(
        prizes_df.groupby([prizes_df.index.weekday,
                           prizes_df['gender']]).agg(agg))



def main():
    prizes_df = prize_data_to_pandas()
    aired_series = aired_dates_to_pandas()

    print('=========== DATA STATS ===========')
    data_stats(prizes_df, aired_series)

    print('=========== YEAR STATS TABLE ===========')
    yearly_stats(prizes_df)

    print('=========== DOF STATS TABLE ===========')
    day_of_week_stats(prizes_df)

    print('=========== WIN / LOSS YEAR PLOT ===========')
    win_loss_year_plot(prizes_df)

    print('=========== VALUE / BET HISTOGRAM  ===========')
    value_bet_histograms(prizes_df)

    print('=========== MARGINS PLOT ===========')
    margins_pie(prizes_df)

    print('=========== MARGINS STATS ===========')
    margins_vs_win_prob(prizes_df)

    print('=========== VALUE / BET 2D HISTOGRAMS ===========')
    value_bet_2d_histograms(prizes_df)

    print('=========== WIN HISTS ===========')
    win_vs_values_histogram(prizes_df)


main()
