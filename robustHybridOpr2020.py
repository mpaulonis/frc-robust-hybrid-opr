"""
Produce several forms of OPR using FRC match data from The Blue Alliance.
- Standard OPR
- Robust OPR - OPR on a match dataset with outlier performances removed
- Hybrid OPR - OPR with some scoring components known exactly from FMS data
- Robust Hybrid OPR - combination of the two techniques above
- No-foul versions of all four forms above
- Component OPR for a select set of scoring components
- Robust component OPR - same as component, but with outlier performances removed

USAGE python robustHybridOpr2020.py eventKey [tbaApiKey]

If a tbaApiKey is not provided as an argument, it must be present in a tba_key.json
file with structure
{"tba_key" : "PF6MROks2K..."}

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

def robustHybridOpr2020():
    
    # make the multiplier of the IQR in the boxplot outlier calculation easily
    # accessible for hyperparameter tuning
    # for the 2020 data, 1.6 is optimal rather than the default 1.5
    IQR_MULT = 1.6
    
    # get the event key from the command-line argument
    try:
        event = sys.argv[1]
    except IndexError:
        sys.exit('ERROR - please specify an event key e.g. 2020scmb as the first argument')
    
    if event[:4] != '2020':
        sys.exit('ERROR - This program is only applicable to 2020 events')
    
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
    
    # calculate contributions known exactly from FMS data
    fmsPerRobotScoring, fmsContrib = calcFMSStats(matches)
    
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
        
        # join FMS contribution data to OPR - use default join key of index, which is team
        opr = opr.join(fmsContrib)
        
        # compute a hybrid OPR which sums FMS data for auto init line and endgame as well 
        # as OPR for auto cells, teleop without endgame, and fouls
        opr['hOpr'] = opr.apply(lambda r: r['fmsInit'] + r['fmsEnd'] + r['oprAutoCell'] +
                                r['oprTele'] + r['oprFoulDraw'], axis=1)
        
        # also add hybrid OPR with no fouls
        opr['hOprNoFoul'] = opr.apply(lambda r: r['fmsInit'] + r['fmsEnd'] + r['oprAutoCell'] +
                                r['oprTele'], axis=1)
        
        # rearrange columns for easier human comparison
        cols = ['hOpr','opr','hOprNoFoul','oprNoFoul','oprAutoCell','fmsInit','oprAuto',
                'oprTele','fmsEnd','oprEnd','oprFoulDraw','oprFoulMade','fmsHangCount']
        opr = opr[cols]
        
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
        q3 = maData['score.errorH'].quantile(0.75)
        # lower quartile is the 25th percentile of the data
        q1 = maData['score.errorH'].quantile(0.25)
        # interquartile range is the difference between the upper and lower quartiles
        iqr = q3 - q1
        # high outlier limit is IQR_MULT * iqr above the upper quartile
        lim_hi = q3 + IQR_MULT * iqr
        # low outlier limit is IQR_MULT * iqr below the lower quartile
        lim_lo = q1 - IQR_MULT * iqr
        
        # look for outliers where the match-alliance prediction error is beyond
        # the outlier limits just calculated
        outliers = maData[(maData['score.errorH'] > lim_hi) | (maData['score.errorH'] < lim_lo)]
        
        # if there are no outlier records, break out of the loop - the last OPR
        # calculated is the robust OPR
        if len(outliers) == 0:
            break
        
        # print to console if outliers are found
        print(f'Outlier(s) found on iteration {i}')
        for index, row in outliers.iterrows():
            print(f'Match {row.key} {row.color} ({row.team1} {row.team2} {row.team3}) - score {row.score} - pred {round(row["score.predH"], 1)}')
        
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
    oprAll = oprAll[['event', 'opr', 'hOpr', 'oprNoFoul', 'hOprNoFoul', 'fmsInit',
                     'oprAutoCell', 'oprAuto', 'oprTele', 'fmsEnd', 'oprEnd',
                     'oprFoulDraw', 'oprFoulMade']]
    # give columns better names for export
    oprAll.columns = ['event', 'oprStd', 'oprHybrid', 'oprNoFoul', 'oprHybridNoFoul', 'fmsInit',
                     'oprAutoCell', 'oprAuto', 'oprTeleNoEnd', 'fmsEnd', 'oprEnd',
                     'oprFoulDraw', 'oprFoulMade']
    # get needed columns from opr which is robust OPR
    opr = opr[['opr', 'hOpr', 'oprNoFoul', 'hOprNoFoul',
                     'oprAutoCell', 'oprAuto', 'oprTele', 'oprEnd',
                     'oprFoulDraw', 'oprFoulMade']]
    # give columns better names for export
    opr.columns = ['oprRobust', 'oprRobustHybrid', 'oprRobustNoFoul',
                   'oprRobustHybridNoFoul',
                     'oprRobustAutoCell', 'oprRobustAuto', 'oprRobustTeleNoEnd',
                     'oprRobustEnd',
                     'oprRobustFoulDraw', 'oprRobustFoulMade']
    # combine standard and robust OPR data
    opr = pd.concat([oprAll, opr], axis=1, sort=False)
    # rearrange columns for easier human interpretation of the exported data
    opr = opr[['event', 'oprStd', 'oprRobust', 'oprHybrid', 'oprRobustHybrid',
               'oprNoFoul', 'oprRobustNoFoul', 'oprHybridNoFoul', 'oprRobustHybridNoFoul',
               'fmsInit', 'oprAutoCell', 'oprRobustAutoCell', 
               'oprAuto', 'oprRobustAuto', 'oprTeleNoEnd', 'oprRobustTeleNoEnd',
               'fmsEnd', 'oprEnd', 'oprRobustEnd',
               'oprFoulDraw', 'oprRobustFoulDraw', 'oprFoulMade', 'oprRobustFoulMade']]
    
    # round numbers in the dataframe to 1 place
    opr = opr.round(1)
        
    # export the results to CSV
    try:
        filename = 'robustHybridOpr_' + event + '.csv'
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
                 'blue.autoPoints','blue.autoCellPoints','blue.teleopPoints',
                 'blue.endgamePoints','blue.foulPoints',
                 'red1','red2','red3','red.score','red.foulPoints',
                 'comp_level', 'match_number']]
    # rename columns to remove references to 'blue' - this information will
    # now be in a column
    b = b.rename(columns = {'blue1':'team1', 'blue2':'team2', 'blue3':'team3',
                        'blue.score':'score', 'blue.autoPoints':'auto',
                        'blue.autoCellPoints':'autoCell',
                        'blue.teleopPoints':'teleop', 'blue.endgamePoints':'endgame',
                        'blue.foulPoints':'foulsDrawn',
                        'red1':'opp1', 'red2':'opp2', 'red3':'opp3',
                        'red.score': 'oppScore',
                        'red.foulPoints': 'foulsMade'})
    # add a column indicating that these data are for blue alliances
    b.loc[:, 'color'] = 'blue'

    # do the same thing for the red alliances
    r = matches[['key','red1','red2','red3','red.score',
                 'red.autoPoints','red.autoCellPoints','red.teleopPoints',
                 'red.endgamePoints','red.foulPoints', 
                 'blue1','blue2','blue3','blue.score','blue.foulPoints',
                 'comp_level', 'match_number']]
    r = r.rename(columns = {'red1':'team1', 'red2':'team2', 'red3':'team3',
                        'red.score':'score', 'red.autoPoints':'auto',
                        'red.autoCellPoints':'autoCell',
                        'red.teleopPoints':'teleop', 'red.endgamePoints':'endgame',
                        'red.foulPoints':'foulsDrawn',
                        'blue1':'opp1', 'blue2':'opp2', 'blue3':'opp3',
                        'blue.score': 'oppScore',
                        'blue.foulPoints': 'foulsMade'})
    r.loc[:, 'color'] = 'red'
    
    # stack the blue and red alliance data together
    # start with the blue data
    maData = b.copy()
    # add the red data
    maData = maData.append(r)
    
    # make foulsMade a negative number - from an OPR perspective, more fouls
    # made is an indicator of poorer performance
    maData['foulsMade'] = maData.apply(lambda r: -r['foulsMade'], axis=1)
    
    # we don't need the 'frc' part of the team identifier that comes from 
    # the API - keep only the team number part
    maData[['team1','team2','team3','opp1','opp2','opp3']] = (
        maData[['team1','team2','team3','opp1','opp2','opp3']].applymap(numOnly))
    
    return maData

def calcFMSStats(matches):
    """
    In the 2020 game, the FMS captures information that can ascribe certain
    scoring actions to specific robots in an alliance.  Use this actual information
    with estimated component OPRs to get a hybrid OPR that should better
    represent actual robot point contribution compared to a fully estimated OPR.
    This function computes the scoring contributions that are precisely known
    from the FMS.

    Parameters
    ----------
    matches : DataFrame
        A DataFrame of alliance match records for an event or some group of
        matches being studied.  Each row represents performance of both alliances
        for one match.  Required content is provided by function getRawMatchData().

    Returns
    -------
    fmsPerRobotScoring : DataFrame
        FMS values and score contribution values per robot per match.
    fmsContrib : DataFrame
        Mean value across all matches for the endgame and initline exit
        point contributions for each robot.

    """
    # only look at qualification matches - consistent with OPR
    matches_comp = matches.loc[matches['comp_level'] == 'qm']
    # replace periods in column names to avoid column access problems later
    matches_comp.columns = matches_comp.columns.str.replace('.', '_')
    # initialize a list to capture scoring-related information per robot per match
    dataList = []
    # iterate over all matches
    for row in matches_comp.itertuples():
        # compute the number of robots hanging for both the blue and red alliances
        bluehang = 0
        if row.blue_endgameRobot1 == 'Hang':
            bluehang += 1
        if row.blue_endgameRobot2 == 'Hang':
            bluehang += 1
        if row.blue_endgameRobot3 == 'Hang':
            bluehang += 1
        redhang = 0
        if row.red_endgameRobot1 == 'Hang':
            redhang += 1
        if row.red_endgameRobot2 == 'Hang':
            redhang += 1
        if row.red_endgameRobot3 == 'Hang':
            redhang += 1
        # create one row for each of the six robots in the current match
        # capture the robot number, the match, the endgame status from FMS,
        # the switch level status from FMS, the initline exit status from FMS,
        # and the resulting point contributions for that robot
        #
        # Blue 1
        team = row.blue1[3:]
        endgame = row.blue_endgameRobot1
        level = row.blue_endgameRungIsLevel
        initline = row.blue_initLineRobot1
        endgamePoints, initlinePoints = calcPoints(endgame, level, bluehang, initline)
        dataList.append([team, row.key.strip().split('_')[1], endgame, level,
                         initline, bluehang, endgamePoints, initlinePoints])
        #
        # Blue 2
        team = row.blue2[3:]
        endgame = row.blue_endgameRobot2
        level = row.blue_endgameRungIsLevel
        initline = row.blue_initLineRobot2
        endgamePoints, initlinePoints = calcPoints(endgame, level, bluehang, initline)
        dataList.append([team, row.key.strip().split('_')[1], endgame, level,
                         initline, bluehang, endgamePoints, initlinePoints])
        #
        # Blue 3
        team = row.blue3[3:]
        endgame = row.blue_endgameRobot3
        level = row.blue_endgameRungIsLevel
        initline = row.blue_initLineRobot3
        endgamePoints, initlinePoints = calcPoints(endgame, level, bluehang, initline)
        dataList.append([team, row.key.strip().split('_')[1], endgame, level,
                         initline, bluehang, endgamePoints, initlinePoints])
        #
        # Red 1
        team = row.red1[3:]
        endgame = row.red_endgameRobot1
        level = row.red_endgameRungIsLevel
        initline = row.red_initLineRobot1
        endgamePoints, initlinePoints = calcPoints(endgame, level, redhang, initline)
        dataList.append([team, row.key.strip().split('_')[1], endgame, level,
                         initline, redhang, endgamePoints, initlinePoints])
        #
        # Red 2
        team = row.red2[3:]
        endgame = row.red_endgameRobot2
        level = row.red_endgameRungIsLevel
        initline = row.red_initLineRobot2
        endgamePoints, initlinePoints = calcPoints(endgame, level, redhang, initline)
        dataList.append([team, row.key.strip().split('_')[1], endgame, level,
                         initline, redhang, endgamePoints, initlinePoints])
        #
        # Red 3
        team = row.red3[3:]
        endgame = row.red_endgameRobot3
        level = row.red_endgameRungIsLevel
        initline = row.red_initLineRobot3
        endgamePoints, initlinePoints = calcPoints(endgame, level, redhang, initline)
        dataList.append([team, row.key.strip().split('_')[1], endgame, level,
                         initline, redhang, endgamePoints, initlinePoints])
    
    # convert the per robot per match scoring list into a DataFrame
    fmsPerRobotScoring = pd.DataFrame(dataList, columns=['team', 'match', 'endgame', 'level',
                                           'initline', 'fmsHangCount', 'fmsEnd',
                                           'fmsInit'])
    # compute the mean scoring values by team for all matches
    fmsContrib = fmsPerRobotScoring.groupby(['team']).mean().sort_values(['team'])
    
    return fmsPerRobotScoring, fmsContrib

def calcPoints(endgame, level, hangCount, initline):
    """
    Calculate point contributions for endgame and exiting the init line using
    values that come from the FMS via TheBlueAlliance API.
    
    Parameters
    ----------
    endgame : string
        The value that comes from teamcolor.endgameRobotX (None, Park, or Hang)
    level : string
        The value that comes from teamcolor.endgameRungIsLevel (NotLevel or IsLevel)
    hangCount : integer
        A computed sum of the number of alliance robots whose endgame value is Hang
    initline : string
        The value that comes from teamcolor.initLineRobotX (None or Exited)

    Returns
    -------
    endgamePoints : float
        The endgame point contribution for the robot with corresponding function
        inputs.  Note that level points, if there are any, are equally distributed
        across hanging robots.
    initlinePoints : float
        The point contribution for movement off the initialization line for the 
        robot with corresponding function inputs.

    """
    # initialize endgame points and add contributions
    endgamePoints = 0
    # a Park is worth 5 points
    if endgame == 'Park':
        endgamePoints += 5
    # a Hang is worth 25 points
    elif endgame == 'Hang':
        endgamePoints += 25
        # if hanging, then IsLevel is worth 15 points
        # divide the points equally for all hanging robots
        if level == 'IsLevel':
            endgamePoints += 15/hangCount
    # initline exit is worth 5 points
    if initline == 'Exited':
        initlinePoints = 5
    else:
        initlinePoints = 0
    return endgamePoints, initlinePoints

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
        columns={0: 'opr', 1: 'oprNoFoul', 2: 'oprAuto', 3: 'oprAutoCell',
                 4: 'oprTele', 5: 'oprEnd',
                 6: 'oprFoulDraw', 7: 'oprFoulMade'},
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
    
    # compute some additional metrics from the data - probably competition-year specific
    # teleop_alone is the teleop score with the endgame score removed
    # score_nf is the alliance score with foul points subtracted (nf = no fouls)
    matches['teleop_alone'] = matches.apply(lambda r: r['teleop'] - r['endgame'], axis=1)
    matches['score_nf'] = matches.apply(lambda r: r['score'] - r['foulsDrawn'], axis=1)
    
    # form the b list of lists from the appropriate columns of the match data
    opr_b = matches[['score', 'score_nf', 'auto', 'autoCell', 'teleop_alone', 'endgame',
                     'foulsDrawn', 'foulsMade']].values.tolist()
    
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
    # predict using hybrid OPR with some per-robot FMS component contributions
    maData['score.predH'] = maData.apply(lambda r:
        oprScorePred([r['team1'], r['team2'], r['team3']], opr, 'hOpr'), axis=1)
    maData['score.errorH'] = maData.apply(lambda r: r['score.predH'] - r['score'], axis=1)
    
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
    robustHybridOpr2020()
