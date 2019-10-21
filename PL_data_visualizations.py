import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, InsetPosition
import joypy
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from random import random
from scipy import floor
import pickle

# # # define the teams of interest, seasons of interest, club colors and relevant features
seasons = ['2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19']
seasonsofinterest = ['2013-14', '2014-15', '2015-16', '2016-17', '2017-18', '2018-19']  # seasons.copy()
allTeams = ['MU', 'MC', 'LFC', 'CFC', 'TH', 'AFC']
teamsofinterest = ['MU', 'MC', 'LFC']  # allTeams.copy()
clubColors = {'MU': '#FF3030', 'MC': '#63B8FF', 'LFC': '#8B0000', 'CFC': '#000080', 'TH': '#FDF5E6', 'AFC': '#EE5C42'}
clubColors2 = {'MU': '#000000', 'MC': '#FFFAF0', 'LFC': '#EEC900', 'CFC': '#F8F8FF', 'TH': '#030303', 'AFC': '#00EEEE'}
excludedPositions = ['GK']  # playing positions NOT of interest
savedfileextension = 'pdf'

# # # all defensive stats considered in these analyses
defFeatures = ['accurate_back_zone_pass', 'accurate_layoffs', 'accurate_long_balls', 'accurate_through_ball', 'aerial_lost', 'aerial_won', 'backward_pass',
               'ball_recovery', 'blocked_cross', 'blocked_scoring_att', 'challenge_lost', 'clean_sheet', 'clearance_off_line', 'dispossessed',
               'duel_lost', 'duel_won', 'effective_blocked_cross', 'effective_clearance', 'effective_head_clearance', 'error_lead_to_goal', 'error_lead_to_shot',
               'fouls', 'hand_ball', 'head_clearance', 'head_pass', 'interception', 'interception_won', 'interceptions_in_box', 'last_man_tackle',
               'long_pass_own_to_opp', 'long_pass_own_to_opp_success', 'lost_corners', 'offside_provoked', 'open_play_pass', 'overrun', 'own_goals',
               'passes_left', 'passes_right', 'pen_goals_conceded', 'penalty_conceded', 'poss_lost_all', 'poss_lost_ctrl', 'poss_won_def_3rd',
               'poss_won_mid_3rd', 'six_yard_block', 'successful_open_play_pass', 'total_back_zone_pass', 'total_chipped_pass', 'total_clearance',
               'total_long_balls', 'total_pass', 'total_tackle', 'touches', 'unsuccessful_touch', 'was_fouled', 'won_contest', 'won_tackle', 'yellow_card']

# # # this is purely a division for plotting purposes
featuresForDistributions = {'featuresForDistributions1': ['accurate_back_zone_pass', 'open_play_pass', 'poss_lost_all', 'successful_open_play_pass', 'total_back_zone_pass', 'total_pass', 'touches'],
                            'featuresForDistributions2': ['accurate_long_balls', 'aerial_won', 'ball_recovery', 'effective_clearance', 'long_pass_own_to_opp_success', 'poss_won_def_3rd', 'total_long_balls'],
                            'featuresForDistributions3': ['blocked_scoring_att', 'challenge_lost', 'dispossessed', 'fouls', 'interception_won', 'unsuccessful_touch', 'won_contest', 'won_tackle']}
whichfeaturestoplot = 1  # 1, 2, or 3 from the dictionary above


def visualizethedata(**kwargs):
    '''
    This is a simple script for visualizing the publicy available statistics of PL players (from the "top 6" sides). It plots the following:
    [1] *wins, draws, losses* to appearances ratios of single players,
    [2] distributions of performance-related parameters compared across teams,
    [3] injury records relative to win percentage compared across teams,
    [4] the progress of teams on performance-related parameters,
    [5] the basic latent structre of performance-related parameters.

    Parameters
    ----------
    **kwargs: dictionary
        filename : str
            Name of the .pkl file where the data is stored. Must be in the same directory as the script to run.
        savetheplots : boolean (0/False or 1/True)
            Save (1/True) or don't save (0/False) produced plots.
        playerContributions : boolean
            Whether to plot [1] from above.
        parameterDistributions : boolean
            Whether to plot [2] from above.
        injuryRecords : boolean
            Whether to plot [3] from above.
        teamProgress : boolean
            Whether to plot [4] from above.
        dataDimensionality : boolean
            Whether to plot [5] from above.
    '''

    if 'filename' in kwargs.keys():
        thefilename = kwargs['filename']
    else:
        print('No such file, try again.')

    # valid values for arguments
    validArgs = [0, False, 1, True]

    savetheplots = [kwargs['savetheplots'] if 'savetheplots' in kwargs.keys() and kwargs['savetheplots'] in validArgs else 0][0]
    playerContributions = [kwargs['playerContributions'] if 'playerContributions' in kwargs.keys() and kwargs['playerContributions'] in validArgs else 0][0]
    parameterDistributions = [kwargs['parameterDistributions'] if 'parameterDistributions' in kwargs.keys() and kwargs['parameterDistributions'] in validArgs else 0][0]
    injuryRecords = [kwargs['injuryRecords'] if 'injuryRecords' in kwargs.keys() and kwargs['injuryRecords'] in validArgs else 0][0]
    teamProgress = [kwargs['teamProgress'] if 'teamProgress' in kwargs.keys() and kwargs['teamProgress'] in validArgs else 0][0]
    dataDimensionality = [kwargs['dataDimensionality'] if 'dataDimensionality' in kwargs.keys() and kwargs['dataDimensionality'] in validArgs else 0][0]

    # first, we load the data from the .pkl file
    thedata = pickle.load(open(thefilename, 'rb'))

    # get all the players in one dictionary, sorted in sub-dictionaries by season and team
    players = {ateam: {aseason: [ax for axind, ax in enumerate(list(thedata[aseason].index)) if thedata[aseason].loc[:, 'club'][axind] == ateam] for aseason in seasonsofinterest} for ateam in allTeams}

    # function which extracts the appendix for the plot name
    def filenameAppendix(allseasonsofinterest):
        whichseasons = ''
        if(len(seasonsofinterest) == 1):
            whichseasons = seasonsofinterest[0]
        else:
            whichseasons = '{} to {}'.format(seasonsofinterest[0], seasonsofinterest[-1])
        return whichseasons

    # function which gets a list with all players in it (either all players or only from teamsofinterest), irrespective of club affiliation
    def getAllPlayers(whatTeams):
        allPlayers = []
        for ateam in whatTeams:
            ateamxplayers = []
            for aseasonofint in seasonsofinterest:
                for xplayer in players[ateam][aseasonofint]:
                    if((thedata[aseasonofint].loc[xplayer, 'position'] not in excludedPositions) and (xplayer not in allPlayers)):
                        ateamxplayers.append(xplayer)
                    elif((thedata[aseasonofint].loc[xplayer, 'position'] not in excludedPositions) and (xplayer in allPlayers)):
                        ateamxplayers.append('{}{}'.format(xplayer, ateam))
            ateamxplayers = set(ateamxplayers)
            for xxplayer in ateamxplayers:
                allPlayers.append(xxplayer)
        return allPlayers

    defensiveFeatures = defFeatures.copy()  # all of defensive fetaures
    interestingFeatures = featuresForDistributions['featuresForDistributions{}'.format(whichfeaturestoplot)]  # features whose distributions are going to be plotted
    interestingFeaturesStripped = [x.replace('_', ' ') for x in interestingFeatures]  # interestting features stripped of irrelevant characters

    # # # plot the *wins, draws, losses* to appearances ratio
    if(playerContributions):
        for ateam in teamsofinterest:
            winsDrawsLosses = {}
            thewhichseasons = filenameAppendix(seasonsofinterest)
            availablePlayers = list(set([aplayer for aseasonofint in seasonsofinterest for aplayer in players[ateam][aseasonofint]]))
            for aplayerind, aplayer in enumerate(availablePlayers):
                total = {'appearances': 0, 'wins': 0, 'draws': 0, 'losses': 0}
                for aseasonind, aseason in enumerate(seasonsofinterest):
                    if(aplayer in players[ateam][aseason]):
                        total['appearances'] += thedata[aseason].loc[aplayer, 'appearances']
                        total['wins'] += thedata[aseason].loc[aplayer, 'wins']
                        total['draws'] += thedata[aseason].loc[aplayer, 'draws']
                        total['losses'] += thedata[aseason].loc[aplayer, 'losses']
                winsDrawsLosses[aplayer] = [total['wins']/total['appearances'], total['draws']/total['appearances'], total['losses']/total['appearances']]

            playersSortedByWins = [i[0] for i in sorted(winsDrawsLosses.items(), key=lambda item: item[1])]
            winsDrawsLossesArray = np.zeros((len(winsDrawsLosses.keys()), 3))
            for akeyind, akey in enumerate(playersSortedByWins):
                winsDrawsLossesArray[akeyind, :] = winsDrawsLosses[akey]
            df_plot = pd.DataFrame(winsDrawsLossesArray, index=playersSortedByWins, columns=['Won %', 'Drew %', 'Lost %'])
            f, axes = plt.subplots(1, 1, figsize=(12, 8))
            ax = plt.subplot(111)
            df_plot.plot(kind='barh', stacked=True, ax=ax, color=[clubColors[ateam], '#696969', '#000000'], alpha=.5)
            ax.legend(loc='upper center', bbox_to_anchor=(1.075, 1.0))
            for anitem in range(len(winsDrawsLosses.keys())):
                prevval = 0
                for aval in winsDrawsLossesArray[anitem, :]:
                    if(aval > 0.):
                        ax.annotate(s='', xy=(prevval, anitem+0.15), xytext=(aval+prevval, anitem+0.15), arrowprops=dict(arrowstyle='<->'))
                        ax.text(((prevval+aval+prevval)/2.)-0.02, -0.1+anitem, '{}'.format(round(aval*100, 1)), color=clubColors2[ateam], fontsize=10, fontweight='bold')
                        prevval += aval
            ax.set_title('How does the team do when specific defenders play?', fontsize=15, pad=25)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(['0', '50', '100'])
            ax.set_xlabel('Win-Draw-Loss breakdown in the PL (%) for {} from {} season'.format(ateam, thewhichseasons))
            ax.tick_params(axis='both', which='both', length=0)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            if(savetheplots):
                f.savefig('{} Players Contribution to Team Success from {}.{}'.format(ateam, thewhichseasons, savedfileextension), bbox_inches='tight', dpi=300)
            plt.plot()

    # # # look at the distributions of different game parameters compared across different teams
    if(parameterDistributions):
        allPlayers = getAllPlayers(teamsofinterest)
        dfBasicStats = pd.DataFrame({**{'Players': allPlayers*len(interestingFeatures), 'Metrics': np.repeat(interestingFeaturesStripped, len(allPlayers))}, **{ateam: [np.nan]*len(interestingFeatures)*len(allPlayers) for ateam in teamsofinterest}})
        thewhichseasons = filenameAppendix(seasonsofinterest)
        for ateam in teamsofinterest:
            availableTeamPlayers = list(set([aplayer for aseasonofint in seasonsofinterest for aplayer in players[ateam][aseasonofint] if thedata[aseasonofint].loc[aplayer, 'position'] not in excludedPositions]))
            for aplayerind, aplayerkey in enumerate(allPlayers):
                if(aplayerkey in availableTeamPlayers):
                    for ametricind, ametrickey in enumerate(interestingFeatures):
                        total = {'appearances': 0, ametrickey: 0}
                        for aseasonind, aseason in enumerate(seasonsofinterest):
                            if((aplayerkey in players[ateam][aseason]) and (ametrickey in thedata[aseason].columns)):
                                total['appearances'] += thedata[aseason].loc[aplayerkey, 'appearances']
                                total[ametrickey] += thedata[aseason].loc[aplayerkey, ametrickey]
                        if(total['appearances'] > 0):
                            dfBasicStats.loc[int(aplayerind+len(allPlayers)*ametricind), ateam] = total[ametrickey]/total['appearances']

        f2, ax2 = joypy.joyplot(dfBasicStats, column=[x for x in teamsofinterest], by='Metrics', ylim='own', linewidth=0.05, overlap=1.5, alpha=.6, color=[clubColors[x] for x in teamsofinterest], legend=True, figsize=(8, 5))
        ax2[-1].set_title('Distributions of defensive features', fontsize=15)
        ax2[-1].set_xlabel('Number of occurrences per game played', fontsize=12)
        if(savetheplots):
            f2.savefig('Teams Basic Stats from {} (comparison {}).{}'.format(thewhichseasons, whichfeaturestoplot, savedfileextension), bbox_inches='tight', dpi=300)
        plt.show()

    # # # plot the injuries stats relative to win percentage across different teams
    if(injuryRecords):
        totalGamesInSeason = 38  # total number of games in one PL season
        thewhichseasons = filenameAppendix(seasonsofinterest)
        injuriesData = {}
        for ateam in teamsofinterest:
            availablePlayers = list(set([aplayer for aseasonofint in seasonsofinterest for oneteam in teamsofinterest for aplayer in players[ateam][aseasonofint] if oneteam == ateam]))
            adf = pd.DataFrame(index=availablePlayers, columns=['Club', 'Win %', 'Games Injured %', 'Games played %'], dtype=float)
            adf.loc[:, 'Club'] = [ateam]*len(availablePlayers)
            for aplayer in availablePlayers:
                total = {'appearances': 0, 'wins': 0, 'games_missed': 0, 'days_injured': 0}
                totalSeasonsPlayed = 0
                for aseason in seasonsofinterest:
                    if(aplayer in players[ateam][aseason]):
                        totalSeasonsPlayed += 1
                        total['appearances'] += thedata[aseason].loc[aplayer, 'appearances']
                        total['wins'] += thedata[aseason].loc[aplayer, 'wins']
                        total['games_missed'] += thedata[aseason].loc[aplayer, 'games_missed']
                        total['days_injured'] += thedata[aseason].loc[aplayer, 'days_injured']
                adf.loc[aplayer, 'Win %'] = (total['wins']/total['appearances'])*100
                adf.loc[aplayer, 'Games played %'] = (total['appearances']/(totalGamesInSeason*totalSeasonsPlayed))*100
                # we want to calculate how many of the total number of games (totalGamesInSeason*totalSeasonsPlayed) the player missed
                # however, the website this is scrapped from offeres games from all competitions, not just the PL which can make the number of misseg games larger than the total possible number of games
                # to, at least somewhat, adjust for this - if the number is over 80% total games played, we set it to 80% plus some small random number
                estimatedGamesMissed = (total['games_missed']/(totalGamesInSeason*totalSeasonsPlayed))*100
                if(estimatedGamesMissed >= 80):
                    adf.loc[aplayer, 'Games Injured %'] = 80 + random()
                else:
                    adf.loc[aplayer, 'Games Injured %'] = estimatedGamesMissed
            injuriesData[ateam] = adf
            # print(adf)
        # concatenatedData = pd.concat([injuriesData['MU'], injuriesData['MC']])
        # print(pearsonr(concatenatedData.loc[:, 'Win %'], concatenatedData.loc[:, 'Games Injured %']))
        f3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
        for club, data in injuriesData.items():
            data.plot(kind='scatter', x='Games Injured %', y='Win %', xlim=((-10, 100)),  ylim=((40, 90)), s=1+data['Games played %']*1.5, fontsize=14, label=club, ax=ax3, color=clubColors[club], edgecolor=clubColors2[club], alpha=.7)
        ax3.xaxis.label.set_size(16)
        ax3.yaxis.label.set_size(16)
        ax3.tick_params(axis='both', which='both', length=0)
        ax3.set_title('Injuries and Win Percentage', fontsize=15, pad=20)
        lgd = ax3.legend(labelspacing=1.2, numpoints=1, loc=1, borderpad=1, frameon=True, framealpha=0.9, title='Club')
        for handle in lgd.legendHandles:
            handle.set_sizes([150.0])
        for di in [10, 25, 50, 100]:
            ax3.scatter([], [], s=1+di*1, c='#000000', label=str(int(di)))
        plotobjects, balls = f3.gca().get_legend_handles_labels()
        ax3.legend(plotobjects[len(teamsofinterest):], balls[len(teamsofinterest):], labelspacing=1.5, title='Games played %', borderpad=1, frameon=True, framealpha=0.9, loc=4, numpoints=1)
        f3.gca().add_artist(lgd)
        if(savetheplots):
            f3.savefig('Wins and Injuries from {}.{}'.format(thewhichseasons, savedfileextension), bbox_inches='tight', dpi=300)
        plt.show()

    # # # plot how teams progress throught time (from a given season onwards - makes sense to look at all seasons) on the measured defensiveFeatures
    if(teamProgress):
        thewhichseasons = filenameAppendix(seasonsofinterest)
        masterDict = {adeffeature: {oneteam: {oneseason: 0. for oneseason in seasonsofinterest} for oneteam in teamsofinterest} for adeffeature in defensiveFeatures}
        for afeature in defensiveFeatures:
            for ateam in teamsofinterest:
                for aseason in seasonsofinterest:
                    tempDict = {'feature': 0., 'appearances': 0.}
                    for aplayer in players[ateam][aseason]:
                        if(afeature in thedata[aseason].columns):
                            tempDict['appearances'] += thedata[aseason].loc[aplayer, 'appearances']
                            tempDict['feature'] += thedata[aseason].loc[aplayer, afeature]
                    if(tempDict['appearances'] > 0):
                        masterDict[afeature][ateam][aseason] = round(tempDict['feature']/tempDict['appearances'], 3)

        # function that initializes all zero-entities in the list to whatever the first non-zero entity is
        def killTheZeros(dictvalues):
            thelst = list(dictvalues)
            firstNonZeroIndex = thelst.index(next(filter(lambda x: x != 0, thelst)))
            return [thelst[firstNonZeroIndex] if ind < firstNonZeroIndex else itm for ind, itm in enumerate(thelst)]

        f4 = plt.figure(figsize=(35, 22))
        for afeatureind, afeature in enumerate(defensiveFeatures):
            empiricalLimits = [0, 0]
            ax4 = f4.add_subplot(6, 10, afeatureind+1)
            for ateam in teamsofinterest:
                interpolate = interp1d(range(len(seasonsofinterest)), killTheZeros(masterDict[afeature][ateam].values()), kind='quadratic')
                xnew = np.linspace(0, len(seasonsofinterest)-1, num=40, endpoint=True)
                ax4.plot(xnew, interpolate(xnew), linestyle='-', linewidth=2.5, color=clubColors[ateam], label=ateam)
                if(np.nanmax(interpolate(xnew)) > empiricalLimits[1]):
                    empiricalLimits[1] = np.nanmax(interpolate(xnew))+(np.nanmax(interpolate(xnew))/4)
                if(np.nanmin(interpolate(xnew)) < empiricalLimits[0]):
                    empiricalLimits[0] = np.nanmin(interpolate(xnew))
            ax4.set_xticklabels([])
            ax4.set_xlim(xmin=0.)
            ax4.set_ylim(empiricalLimits)
            ax4.set_ylabel('{}'.format(afeature.replace('_', ' ')))
            ax4.tick_params(axis='both', which='both', length=0)
            ax4.spines['left'].set_linewidth(2)
            ax4.spines['right'].set_visible(False)
            ax4.spines['top'].set_visible(False)
            ax4.spines['bottom'].set_visible(False)
        plt.subplots_adjust(wspace=0.45, hspace=0.25)
        plt.legend(loc='upper center', bbox_to_anchor=(1.575, 1.0), prop={'size': 15})
        if(savetheplots):
            f4.savefig('Defensive Stats Change Across Teams from {}.{}'.format(thewhichseasons, savedfileextension), bbox_inches='tight', dpi=1000)
        plt.show()

    # # # see if these defensive parameters are correlated (across all teams and players)
    if(dataDimensionality):
        defensiveFeatures = ['days_injured', 'games_missed'] + defFeatures
        featuresAndWins = ['wins'] + defensiveFeatures
        allPlayers = getAllPlayers(allTeams)
        totalGamesInSeason = 38  # total number of games in one PL season
        dfBasicStats = pd.DataFrame(index=allPlayers, columns=['club', 'position'] + featuresAndWins, dtype=float)
        thewhichseasons = filenameAppendix(seasonsofinterest)
        for ateam in allTeams:
            teamPlayers = list(set(['{}{}'.format(aplayer, ateam) if '{}{}'.format(aplayer, ateam) in allPlayers else aplayer for aseasonofint in seasonsofinterest for aplayer in players[ateam][aseasonofint] if thedata[aseasonofint].loc[aplayer, 'position'] not in excludedPositions]))
            for aplayerkey in teamPlayers:
                for ametricind, ametrickey in enumerate(featuresAndWins):
                    if(ametricind == 0):
                        dfBasicStats.loc[aplayerkey, 'club'] = ateam
                    total = {'seasons_played': 0, 'appearances': 0, ametrickey: 0}
                    for aseason in seasonsofinterest:
                        if(aplayerkey.replace(ateam, '') in players[ateam][aseason]):
                            total['seasons_played'] += 1
                            if(type(dfBasicStats.loc[aplayerkey, 'position']) != 'str'):
                                dfBasicStats.loc[aplayerkey, 'position'] = thedata[aseason].loc[aplayerkey.replace(ateam, ''), 'position']
                            if(ametrickey in thedata[aseason].columns):
                                total['appearances'] += thedata[aseason].loc[aplayerkey.replace(ateam, ''), 'appearances']
                                total[ametrickey] += thedata[aseason].loc[aplayerkey.replace(ateam, ''), ametrickey]
                    if(ametrickey == 'games_missed'):
                        estimatedGamesMissed = total[ametrickey]/(totalGamesInSeason*total['seasons_played'])
                        if(estimatedGamesMissed >= .8):
                            dfBasicStats.loc[aplayerkey, ametrickey] = .8 + random()*1e-2
                        else:
                            dfBasicStats.loc[aplayerkey, ametrickey] = estimatedGamesMissed
                    else:
                        if(total['appearances'] > 0):
                            dfBasicStats.loc[aplayerkey, ametrickey] = total[ametrickey]/total['appearances']

        # there's always a catch. Since not all the features were measured in every season (unfortunately)
        # I decided to loop through the df one more time and replace NaNs with mean values of that feature as measured across all other players
        for aplayerkey in allPlayers:
            for ametrickey in defensiveFeatures:
                if(pd.isna(dfBasicStats.loc[aplayerkey, ametrickey])):
                    dfBasicStats.loc[aplayerkey, ametrickey] = np.nanmean(dfBasicStats.loc[:, ametrickey])

        # determine what features correlate best with win percentage
        importantDefensiveFeatures = []
        for afeature in defensiveFeatures:
            corr = pearsonr(dfBasicStats.loc[:, afeature].values, dfBasicStats.loc[:, 'wins'].values)
            if(corr[1] < 0.01):  # abs(corr[0]) >= .3
                # print(afeature, corr)
                importantDefensiveFeatures.append(afeature)

        # create correlation matrix of defensiveFeatures
        corrMatrix = np.zeros((len(defensiveFeatures), len(defensiveFeatures)))
        for row, akey1 in enumerate(list(dfBasicStats.columns)[3:]):
            for col, akey2 in enumerate(list(dfBasicStats.columns)[3:]):
                corrMatrix[row, col] = pearsonr(dfBasicStats.loc[:, akey1], dfBasicStats.loc[:, akey2])[0]
        np.fill_diagonal(corrMatrix, 0)  # put zeros in diagonal

        # z-score all the variables for PCA
        rescaledData = StandardScaler().fit_transform(dfBasicStats.loc[:, defensiveFeatures].values)
        """rescaledData0 = dfBasicStats.copy()
        for afeature in defensiveFeatures:
            rescaledData0.loc[:, afeature] = dfBasicStats.loc[:, afeature] - np.nanmean(dfBasicStats.loc[:, afeature])
        rescaledData = rescaledData0.loc[:, defensiveFeatures].values"""

        # do PCA
        pca = PCA()
        components = pca.fit_transform(rescaledData)
        eigenvalues = pca.explained_variance_

        # choose whether you want to color-code player by club of playing position
        clubORposition = 0  # 1 is for club, other sybols are for playing position

        if(clubORposition):
            principalDf = pd.DataFrame(data=np.column_stack((dfBasicStats.loc[:, 'club'].values, components[:, :2])), index=allPlayers, columns=['club', 'Comp. 1', 'Comp. 2'])
            f5, ax5 = plt.subplots(figsize=(8, 8))
            ax5.plot([], [], marker='o', ms=10, linestyle='None', alpha=.7, color='#808080', markeredgecolor='#808080', label='Other')
            theclubs = {aclub: 0. for aclub in teamsofinterest}
            for aplayer in principalDf.index.values:
                if(principalDf.loc[aplayer, 'club'] in teamsofinterest):
                    if(theclubs[principalDf.loc[aplayer, 'club']] == 0):
                        ax5.plot(principalDf.loc[aplayer, 'Comp. 1'], principalDf.loc[aplayer, 'Comp. 2'], marker='o', ms=10, linestyle='None', alpha=.7, color=clubColors[principalDf.loc[aplayer, 'club']], markeredgecolor=clubColors2[principalDf.loc[aplayer, 'club']], label=principalDf.loc[aplayer, 'club'])
                    else:
                        ax5.plot(principalDf.loc[aplayer, 'Comp. 1'], principalDf.loc[aplayer, 'Comp. 2'], marker='o', ms=10, linestyle='None', alpha=.7, color=clubColors[principalDf.loc[aplayer, 'club']], markeredgecolor=clubColors2[principalDf.loc[aplayer, 'club']])
                    theclubs[principalDf.loc[aplayer, 'club']] += 1
                else:
                    ax5.plot(principalDf.loc[aplayer, 'Comp. 1'], principalDf.loc[aplayer, 'Comp. 2'], marker='o', ms=10, alpha=.7, color='#808080', markeredgecolor='#808080')
            ax5.legend(labelspacing=1.2, numpoints=1, loc='best', borderpad=1, frameon=True, framealpha=0.9, title='Club')
            ax5.tick_params(axis='both', which='both', length=0)
            ax5.set_xlabel('Comp. 1', fontsize=15)
            ax5.set_ylabel('Comp. 2', fontsize=15)

            ax6 = plt.axes([0, 0, 1, 1])
            ip = InsetPosition(ax5, [0.04, 0.74, 0.4, 0.25])
            ax6.set_axes_locator(ip)
            ax6.plot(range(len(eigenvalues)), eigenvalues, marker='o', ms=1, linestyle='-', alpha=.7, color='#000000')
            ax6.set_xticks([])
            ax6.set_yticks([])
            ax6.set_xlabel('Component')
            ax6.set_ylabel('Eigenvalue')

            ax7 = ax5.inset_axes([0.04, 0.035, 0.2, 0.2])
            im = ax7.imshow(np.tril(corrMatrix), cmap='bwr', vmin=-1, vmax=1)
            cbaxes = ax5.inset_axes([0.24, 0.035, 0.0075, 0.2])
            cbar = f5.colorbar(im, cax=cbaxes, label='Correlation', ticks=[-1, 0, 1])
            cbar.ax.set_yticklabels(['-{}'.format(1), '{}'.format(0), '+{}'.format(1)])
            ax7.set_xticks([])
            ax7.set_yticks([])
            ax7.set_xlabel('Defensive stats')
            ax7.set_ylabel('Defensive stats')

            if(savetheplots):
                f5.savefig('Dimensionality reduction from season {} (by club).{}'.format(thewhichseasons, savedfileextension), bbox_inches='tight', dpi=1000)
            plt.show()
        else:
            principalDf = pd.DataFrame(data=np.column_stack((dfBasicStats.loc[:, 'position'].values, components[:, :2])), index=allPlayers, columns=['position', 'Comp. 1', 'Comp. 2'])
            positionColors = {'CB': '#838B8B', 'LB': '#F0FFFF', 'RB': '#C1CDCD'}
            f5, ax5 = plt.subplots(figsize=(8, 8))
            thepositions = {aposition: 0. for aposition in positionColors.keys()}
            for aplayer in principalDf.index.values:
                if(thepositions[principalDf.loc[aplayer, 'position']] == 0):
                    ax5.plot(principalDf.loc[aplayer, 'Comp. 1'], principalDf.loc[aplayer, 'Comp. 2'], marker='o', ms=10, linestyle='None', alpha=.9, color=positionColors[principalDf.loc[aplayer, 'position']], markeredgecolor='#000000', label=principalDf.loc[aplayer, 'position'])
                else:
                    ax5.plot(principalDf.loc[aplayer, 'Comp. 1'], principalDf.loc[aplayer, 'Comp. 2'], marker='o', ms=10, linestyle='None', alpha=.9, color=positionColors[principalDf.loc[aplayer, 'position']], markeredgecolor='#000000')
                thepositions[principalDf.loc[aplayer, 'position']] += 1
            ax5.legend(labelspacing=1.2, numpoints=1, loc='best', borderpad=1, frameon=True, framealpha=0.9, title='Position')
            ax5.tick_params(axis='both', which='both', length=0)
            ax5.set_xlabel('Comp. 1', fontsize=15)
            ax5.set_ylabel('Comp. 2', fontsize=15)

            ax6 = plt.axes([0, 0, 1, 1])
            ip = InsetPosition(ax5, [0.04, 0.74, 0.4, 0.25])
            ax6.set_axes_locator(ip)
            ax6.plot(range(len(eigenvalues)), eigenvalues, marker='o', ms=1, linestyle='-', alpha=.7, color='#000000')
            ax6.set_xticks([])
            ax6.set_yticks([])
            ax6.set_xlabel('Component')
            ax6.set_ylabel('Eigenvalue')

            ax7 = ax5.inset_axes([0.04, 0.035, 0.2, 0.2])
            im = ax7.imshow(np.tril(corrMatrix), cmap='bwr', vmin=-1, vmax=1)
            cbaxes = ax5.inset_axes([0.24, 0.035, 0.0075, 0.2])
            cbar = f5.colorbar(im, cax=cbaxes, label='Correlation', ticks=[-1, 0, 1])
            cbar.ax.set_yticklabels(['-{}'.format(1), '{}'.format(0), '+{}'.format(1)])
            ax7.set_xticks([])
            ax7.set_yticks([])
            ax7.set_xlabel('Defensive stats')
            ax7.set_ylabel('Defensive stats')

            if(savetheplots):
                f5.savefig('Dimensionality reduction from season {} (by position).{}'.format(thewhichseasons, savedfileextension), bbox_inches='tight', dpi=1000)
            plt.show()


visualizethedata(filename='premierleague_data', playerContributions=1, parameterDistributions=0, injuryRecords=0, teamProgress=0, dataDimensionality=0, savetheplots=0)
