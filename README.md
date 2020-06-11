
# Robust and hybrid OPR for FRC
Two alternate forms of offensive power rating (OPR) were developed and studied using 2020 FIRST Robotics Competition season data. **Hybrid OPR** combines per-robot data from the field management system (FMS) with component OPRs for scoring that is not uniquely identified by FMS to create an OPR where some components are known without error.  **Robust OPR** identifies and removes outlier alliance performances from the set of all performances used to compute OPR in the traditional way.  The hybrid and robust techniques can also be applied together.

The hybrid, robust, and hybrid robust OPRs were each better at predicting playoff alliance scores and scoring margin across 2020 events compared to standard OPR.  The hybrid robust approach reduced the sum of squared errors (SSE) of playoff alliance score predictions by almost 7% compared to standard OPR and reduced the SSE of playoff match scoring margin prediction by almost 5%.  Average improvement was similar for district and regional events.

Splitting out fouls to compute playoff scores by using no-foul OPRs for a given alliance plus fouls-committed component OPRs for the opponent alliance were found to have a detrimental effect on playoff score prediction regardless of OPR technique for the 2020 season data.

The study was motivated by the substantial disagreement between standard OPR and scouting point contribution estimates for team 4020 at the 2020 Palmetto regional.  The standard OPR for 4020 was 47.9 at Palmetto compared to a scouting estimate of 67.0 points per match and a hybrid robust OPR of 69.2.  The much improved agreement with scouting came from clearly identifiable sources of poor endgame standard component OPR which was replaced with accurate FMS-supplied endgame contribution and an outlier match where the best team at the event scored 0 points in an alliance with 4020.  Overall, hybrid robust OPR reduced the SSE of OPR vs team 4020 scouting estimates by 65% across all teams at 2020 Palmetto.

## Resources
A [paper](https://github.com/mpaulonis/frc-robust-hybrid-opr/blob/master/robust-hybrid-opr-frc.pdf) is available which contains detailed descriptions of the techniques and the results of the study on the 2020 events.

Interactive versions of the figures in the paper are available on Tableau Public:

 - [Most of the figures](https://public.tableau.com/profile/mike.paulonis#!/vizhome/RobustHybridOPRFRCMain/Fig1HybridDash)
 - [Scouting-related figure](https://public.tableau.com/profile/mike.paulonis#!/vizhome/RobustHybridOPRFRCScouting/Fig2ScoutvOPRDash)
 - [Parameter optimization figure](https://public.tableau.com/profile/mike.paulonis#!/vizhome/RobustHybridOPRFRCOutlierOpt/Fig3OutlierkDash)

Results datasets in .csv format

 - [Performance of all forms of OPR on 2020 event playoff predictions by several error measures](https://github.com/mpaulonis/frc-robust-hybrid-opr/blob/master/robustHybridOprFrcEventStats.csv)
 - [Playoff score predictions for 2020 events using all forms of OPR](https://github.com/mpaulonis/frc-robust-hybrid-opr/blob/master/robustHybridOprFrcPlayoffPreds.csv)
 - [Qualification match performances identified as robust OPR outliers for 2020 events](https://github.com/mpaulonis/frc-robust-hybrid-opr/blob/master/robustHybridOprFrcOutliers.csv)
 - [All forms of event OPRs and component OPRs](https://github.com/mpaulonis/frc-robust-hybrid-opr/blob/master/robustHybridOprFrcOpr.csv)

## Code
Python code is provided to compute the forms of OPR developed in the paper.  The files read data for completed or underway events from The Blue Alliance API and generate an output file with the OPR results. 

There are two files:

### robustOpr
This file only computes standard and robust OPR and intended to be season-independent.

### robustHybridOpr2020
This file is intended for use with match data from the 2020 season only because things like component OPR, no-foul OPR, and hybrid OPR are season-specific.  Changes for the 2021 season will likely be minor.  Modifications for future seasons are likely more extensive, but expected to be straightforward.

Output includes:
- Standard OPR
- Robust OPR - OPR on a match dataset with outlier performances removed
- Hybrid OPR - OPR with some scoring components known exactly from FMS data
- Robust Hybrid OPR - combination of the two techniques above
- No-foul versions of all four forms above
- Component OPR for a select set of scoring components
- Robust component OPR - same as component, but with outlier performances removed

### Usage
Both files require an FRC-standard event key to identify the event for which to compute OPRs.  One easy way to find an event key is to search for a desired event on [The Blue Alliance](https://www.thebluealliance.com/) and the URL of the main event page includes the event key at the end.  For example, the TBA URL for the 2020 Palmetto Regional is [https://www.thebluealliance.com/event/2020scmb](https://www.thebluealliance.com/event/2020scmb) and the event key is **2020scmb**.

Since the files acquire data from The Blue Alliance API, a TBA API key is required for access.  The API key is free, but must be requested from your [TBA account](https://www.thebluealliance.com/account).

A tba_key.json file template is provided in the repo.  You can paste your TBA API key into this file for easy, repeated access by the Python scripts.  Alternatively, the key can be optionally passed as an argument.

```python robustOpr.py eventKey [tbaAPIKey]```

```python robustHybridOpr2020.py eventKey [tbaApiKey]```

Output is provided as robustOpr_*eventKey*.csv or robustHybridOpr2020_*eventKey*.csv.



## License

The content of this project is licensed under the [Creative Commons Attribution 3.0 Unported license](https://creativecommons.org/licenses/by/3.0/), except for the Python code which is licensed under the [MIT license](LICENSE.md).

