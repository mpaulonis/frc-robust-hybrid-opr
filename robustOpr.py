"""
Produce standard and robust OPR using FRC match data from The Blue Alliance.

USAGE python robustOpr.py eventKey [tbaApiKey]

If a tbaApiKey is not provided as an argument, it must be present in a tba_key.json
file with structure
{"tba_key" : "OF6MROks2K..."}

No component OPRs or no-foul OPRs are computed since this is intended to be
a season-independent program.

Note that this program computes robust OPR by removing outlier performances
identified with standard OPR rather than hybrid OPR as described in the paper.
Hybrid OPR is not being computed within this season-independent program.

This program is designed to be used after event qualification matches are 
complete.  The OPR system is solved by standard least-squares.  There are no
calculations to stabilize OPR while qualification matches are taking place.

Mike Paulonis
Team 4020 - Cyber Tribe
June 2020
"""
import sys
import json
import tbapy
import pandas as pd
from sklearn.linear_model import LinearRegression

def robustOpr():
    
    # make the multiplier of the IQR in the boxplot outlier calculation easily
    # accessible for hyperparameter tuning
    # for the 2020 data, 1.6 is optimal rather than the default 1.5
    IQR_MULT = 1.6
    
    # get the event key from the command-line argument
    try:
        event = sys.argv[1]
    except IndexError:
        sys.exit('ERROR - please specify an event key e.g. 2020scmb as the first argument')
    
    # optionally get the TBA API read key from the second argument
    if len(sys.argv) >= 3:
        tbaKey = sys.argv[2]
    else:
        try:
            # check for tba_key.json and read the TBA API key from the file
            with open('tba_key.json', 'r') as read_file:
                f = json.load(read_file)
                tbaKey = f['tba_key']
        except OSError:
            sys.exit('ERROR - must provide TBA API read key in a tba_key.json file or as the second argument')

    # get data for a given event key from TheBlueAlliance API
    matches = getRawMatchData(event, tbaKey)

    # convert the match data to match-alliance data as needed by the OPR functions
    maData = matchToAlliance(matches)

    # truncate data to only qualification matches for OPR
    maData = maData[maData['comp_level'] == 'qm']
    
    # get unique list of teams within the match data as needed for OPR calcs
    teams = pd.unique(maData[['team1','team2','team3']].values.ravel('K')).tolist()
    # sort the team numbers for easier perusing of the OPR output
    # note that the sort is alphabetical since the team numbers are strings
    teams.sort()

    # initialize an index counter for an outlier removal loop
    i = 1
    
    # loop over OPR calculations - removing outlier match-alliance records in each
    # iteration and recomputing robust OPR - stop when no match-alliance records
    # are identified as outliers
    while True:
    
        # compute OPR - maData may be the original data for the full event or
        # could be on a truncated set of match-alliances after outlier removal
        # assuming that outlier removal does not remove all matches for any team
        # that the teams list needs to be recomputed during iteration
        opr = calcOPR(teams, maData)

        if i == 1:
            # save the OPR before outlier removal
            oprAll = opr.copy()
        
        # compute prediction errors for the OPR dataset
        maData = predictionError(maData, opr)

        # identify outlier match-alliance records using non-parametric boxplot
        # outlier computations - values more extreme than IQR_MULT times the
        # interquartile range outside the respective quartile are identified
        # as outliers
        # upper quartile is the 75th percentile of the data
        q3 = maData['score.errorS'].quantile(0.75)
        # lower quartile is the 25th percentile of the data
        q1 = maData['score.errorS'].quantile(0.25)
        # interquartile range is the difference between the upper and lower quartiles
        iqr = q3 - q1
        # high outlier limit is IQR_MULT * iqr above the upper quartile
        lim_hi = q3 + IQR_MULT * iqr
        # low outlier limit is IQR_MULT * iqr below the lower quartile
        lim_lo = q1 - IQR_MULT * iqr
        
        # look for outliers where the match-alliance prediction error is beyond
        # the outlier limits just calculated
        outliers = maData[(maData['score.errorS'] > lim_hi) | (maData['score.errorS'] < lim_lo)]
        
        # if there are no outlier records, break out of the loop - the last OPR
        # calculated is the robust OPR
        if len(outliers) == 0:
            break
        
        # print to console if outliers are found
        print(f'Outlier(s) found on iteration {i}')
        for index, row in outliers.iterrows():
            print(f'Match {row.key} {row.color} ({row.team1} {row.team2} {row.team3}) - score {row.score} - pred {round(row["score.predS"], 1)}')
        
        # find the indexes of the outlier records
        toDrop = list(outliers.index.values)
        # remove the outlier records from the qualification match-alliance dataset
        # before re-computing robust OPR on the next iteration
        maData.drop(toDrop, axis=0, inplace = True)
        
        # update the loop counter
        i += 1
        
        # run another iteration after outlier removal

    # prepare the OPR results for export
    # add event into the dataframe
    oprAll.insert(loc=0, column='event', value=event)
    # get needed columns from oprAll which is "standard" OPR
    oprAll = oprAll[['event', 'opr']]
    # give columns better names for export
    oprAll.columns = ['event', 'oprStd']
    # get needed columns from opr which is robust OPR
    opr = opr[['opr']]
    # give columns better names for export
    opr.columns = ['oprRobust']
    # combine standard and robust OPR data
    opr = pd.concat([oprAll, opr], axis=1, sort=False)
    # rearrange columns for easier human interpretation of the exported data
    opr = opr[['event', 'oprStd', 'oprRobust']]
    
    # round numbers in the dataframe to 1 place
    opr = opr.round(1)
        
    # export the results to CSV
    try:
        filename = 'robustOpr_' + event + '.csv'
        opr.to_csv(filename)
    except OSError:
        sys.exit(f'ERROR - output file could not be written - is {filename} open for editing?')

def getRawMatchData(eventKey, tbaKey):
    """
    For a supplied event key, get match data from The Blue Alliance.  Flatten
    the internal dictionaries and lists in the returned data so the result
    is easier to use.

    Parameters
    ----------
    eventKey : string
        This is the event identifier string used by FIRST and The Blue Alliance.
        Consists of a four-digit year followed by a multi-character abbreviaion
        of the event which is often based on location (e.g. the Palmetto regional
        in Myrtle Beach, SC for 2020 has event key 2020scmb).
    tbaKey : string
        This is the TBA API read key which is generated on the TBA site

    Returns
    -------
    matches : DataFrame
        This is the match-by-match data from an event which has been flattened
        for future ease-of-use.  It includes qualification and playoff data.
        Columns in the dataframe have meaningful names.  Some column names will almost
        certainly change from year-to-year based on the game.

    """
     # instantiate an object for reading TBA API
    tba = tbapy.TBA(tbaKey)
    
    # get match data for the event from The Blue Alliance
    try:
        matches = tba.event_matches(eventKey)
    except ValueError:
        sys.exit(f'ERROR - cannot get match data - is {eventKey} a valid event key?\nIs this a valid TBA API key {tbaKey}?')
    
    if matches != []:
        
        # convert list of objects to a dataframe
        matches = pd.DataFrame(matches)
        
        # convert columns with dicts to individual columns
        matches = pd.concat([matches.drop('alliances', axis=1), pd.json_normalize(matches['alliances'])], axis=1)
        matches = pd.concat([matches.drop('score_breakdown', axis=1), pd.json_normalize(matches['score_breakdown'])], axis=1)
        
        # expand lists of team numbers to individual columns
        matches[['blue1', 'blue2', 'blue3']] = pd.DataFrame(matches['blue.team_keys'].values.tolist(), index = matches.index)
        matches = matches.drop('blue.team_keys', axis=1)
        matches[['red1', 'red2', 'red3']] = pd.DataFrame(matches['red.team_keys'].values.tolist(), index = matches.index)
        matches = matches.drop('red.team_keys', axis=1)
        
        # sort matches by match start time
        matches.sort_values(by = ['time'], inplace = True)
        
        # make the index be ascending integers after sorting
        matches.reset_index(drop = True, inplace = True)
        
    else:
       sys.exit(f'ERROR - there are no match records for {eventKey} - is it complete or underway?') 
        
    return matches

def matchToAlliance(matches):
    """
    Convert match data as provided by the API into match-alliance data as
    needed for the OPR calculation functions.  The API provides one row per
    match with blue and red alliance data on the row.  Split each API row into
    two rows - one for blue alliance and one for red alliance.  OPR does not
    need all of the data from the API, so just keep the relevant columns.
    Remove 'frc' from the team identifiers.

    Parameters
    ----------
    matches : DataFrame
        A DataFrame containing match data from TheBlueAlliance API.  The raw
        data from the API needs to be flattened using the getRawMatchData
        function before providing it to this function.

    Returns
    -------
    maData : DataFrame
        A DataFrame containing match data that has been split apart into
        separate rows for each alliance for each match.  The DataFrame column
        names are descriptive of the column contents.
    """
    # extract OPR-relevant match data for the blue alliances
    # the column names from the raw match data may change between competition
    # years - be prepared to make modifications each year
    b = matches[['key','blue1','blue2','blue3','blue.score',
                 'red1','red2','red3','red.score',
                 'comp_level', 'match_number']]
    # rename columns to remove references to 'blue' - this information will
    # now be in a column
    b = b.rename(columns = {'blue1':'team1', 'blue2':'team2', 'blue3':'team3',
                        'blue.score':'score',
                        'red1':'opp1', 'red2':'opp2', 'red3':'opp3',
                        'red.score': 'oppScore'})
    # add a column indicating that these data are for blue alliances
    b.loc[:, 'color'] = 'blue'

    # do the same thing for the red alliances
    r = matches[['key','red1','red2','red3','red.score',
                 'blue1','blue2','blue3','blue.score',
                 'comp_level', 'match_number']]
    r = r.rename(columns = {'red1':'team1', 'red2':'team2', 'red3':'team3',
                        'red.score':'score',
                        'blue1':'opp1', 'blue2':'opp2', 'blue3':'opp3',
                        'blue.score': 'oppScore'})
    r.loc[:, 'color'] = 'red'
    
    # stack the blue and red alliance data together
    # start with the blue data
    maData = b.copy()
    # add the red data
    maData = maData.append(r)
     
    # we don't need the 'frc' part of the team identifier that comes from 
    # the API - keep only the team number part
    maData[['team1','team2','team3','opp1','opp2','opp3']] = (
        maData[['team1','team2','team3','opp1','opp2','opp3']].applymap(numOnly))
    
    return maData

def calcOPR(teams, matches):
    """
    Calculate multiple OPR metrics for an event or group of matches. Standard
    OPR based on the match score is produced as well as component OPRs for
    autonomous, teleop, endgame, fouls received, fouls committed, and score
    without fouls.

    Parameters
    ----------
    teams : list of strings
        A list of team numbers as strings (e.g. '4020') at an event or are
        otherwise part of a group of matches being studied.
    matches : DataFrame
        A DataFrame of alliance match records for an event or some group of
        matches being studied.  Each row represents performance of one alliance
        for one match.  Required content is provided by function matchToAlliance().

    Returns
    -------
    opr : DataFrame
        A DataFrame with multiple OPR metrics by team for the event or group
        of matches being studied.  The DataFrame column names are descriptive
        regarding the specific OPR metric in the column.  The DataFrame index
        is the team number as a string.
    """
    # convert the input data into the A and b matrices needed to solve for OPR
    opr_A, opr_b  = oprMatrices(teams, matches)
    
    # solve the OPR Ax = b using LinearRegression from sklearn
    # lots of other ways to solve - this is just one easy option
    lin = LinearRegression(fit_intercept = False)
    lin.fit(opr_A, opr_b)
    opr_x = lin.coef_
    
    # convert the array of OPR values to a more descriptive DataFrame
    # transpose LinearRegression output so metrics are on columns
    opr = pd.DataFrame(opr_x.T)
    # add the team numbers as a column
    opr['team'] = teams
    # rename the columns to be descriptive
    # column order is consistent with the column order from function oprMatrices
    opr.rename(
        columns={0: 'opr'},
        inplace = True)
    # move the team column to be the index of the DataFrame
    opr = opr.set_index('team')
    
    return opr

def oprMatrices(teams, matches):
    """
    Produce the A and b matrices needed to solve the system Ax = b for OPR.
    The A matrix produced here has dimensions matches x teams.  This may not
    be the most efficient arrangement for solving the system, but it is the
    easiest to understand and it is easy to remove an outlier match for
    producing a robust OPR.

    Parameters
    ----------
    teams : list of strings
        A list of team numbers as strings (e.g. '4020') at an event or are
        otherwise part of a group of matches being studied.
    matches : DataFrame
        A DataFrame of alliance match records for an event or some group of
        matches being studied.  Each row represents performance of one alliance
        for one match.  Required content is provided by function matchToAlliance().

    Returns
    -------
    opr_A : list of lists of integers
        Nested lists representing the A matrix for OPR Ax = b.  The outer list
        has length matches.  The inner lists have dimension teams.
    opr_b : list of lists of floats
        Nested lists representing the b matrix for OPR Ax = b.  The outer list
        has length matches.  The inner lists have multiple score components of
        a match.  See the function for the specific components and their order.
    """
    # initialize list of lists for OPR A matrix
    opr_A = [[0]*len(teams) for _ in range(len(matches))]
    
    # form the b list of lists from the appropriate columns of the match data
    opr_b = matches[['score']].values.tolist()
    
    # reset the matches index to consecutive integers so it is useful for
    # building the A matrix
    matches.reset_index(drop = True, inplace = True)

    # build the A matrix row-by-row where rows represent a match-alliance pair
    for index, match in matches.iterrows():
        
        # get the 'column' position of each of the alliance teams based on the
        # order of the input teams list
        c1 = teams.index(match['team1'])
        c2 = teams.index(match['team2'])
        c3 = teams.index(match['team3'])

        # build a row of the A matrix with ones in the columns for the
        # participating teams
        opr_A[index][c1] = 1
        opr_A[index][c2] = 1
        opr_A[index][c3] = 1

    return opr_A, opr_b

def predictionError(maData, opr):
    """
    Use OPR to predict scores and compute errors for match-alliance records. 

    Parameters
    ----------
    maData : DataFrame
        A DataFrame of alliance match records for an event.
        Each row represents performance of one alliance for one match.
        Required content is provided by function matchToAlliance().
    opr : DataFrame
        A DataFrame with OPR metrics by team for an event.  The needed
        DataFrame is supplied by function calcOPR.

    Returns
    -------
    maData : DataFrame
        This is the input maData augmented with prediction, error and squared
        error columns based on predictions using OPR.
    """
    # predict using standard OPR where all components are based on alliance scores
    maData['score.predS'] = maData.apply(lambda r:
        oprScorePred([r['team1'], r['team2'], r['team3']], opr, 'opr'), axis=1)
    maData['score.errorS'] = maData.apply(lambda r: r['score.predS'] - r['score'], axis=1)
    
    return maData

def oprScorePred(alliance, opr, oprType = 'opr'):
    """
    Given an alliance of three teams, an OPR DataFrame, and which kind of OPR
    to use, compute a predicted score for that alliance in a match.  If a team
    is not found in the OPR DataFrame, return a zero for their contribution.
    This error checking was needed for event 2020waspo, where there were only
    seven alliances in the playoffs and the missing alliance was populated in 
    the FMS results with three team numbers that were not competing at the event.

    Parameters
    ----------
    alliance : list of strings
        A list of the team numbers as strings (e.g. '4020') which make up the
        alliance.
    opr : DataFrame
        A DataFrame with OPR metrics by team for an event.  The needed
        DataFrame is supplied by function calcOPR.
    oprType : string
        The column name in the OPR DataFrame for the typr of OPR to be used to
        compute the predicted alliance score.  Defaults to the OPR for the
        alliance score, 'opr'

    Returns
    -------
    scorePred : float
        The score predicted for the alliance by the type of OPR requested.

    """
    try:
        c1 = opr.at[alliance[0], oprType]
    except:
        c1 = 0.0
        print('warning: OPR not found for team ' + alliance[0])
    
    try:
        c2 = opr.at[alliance[1], oprType]
    except:
        c2 = 0.0
        print('warning: OPR not found for team ' + alliance[1])
    
    try:
        c3 = opr.at[alliance[2], oprType]
    except:
        c3 = 0.0
        print('warning: OPR not found for team ' + alliance[2])
        
    scorePred = c1 + c2 + c3
    
    return scorePred

def numOnly(s):
    # given a string input, return only the numeric characters in the string
    # all together in the same order as they appear in the input
    # note that the function returns a string, not a number
    return ''.join(i for i in s if i.isdigit())

if __name__ == '__main__':
    robustOpr()
