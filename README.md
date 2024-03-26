This repository is provided as the replication package for the ICSE 2025 Submission #619 "Identifying Code Reading Patterns during Bug Localization Tasks". 

In this repository, we provide 
* The source code snippets used for our experiment,
* The detailed information of devices used and their settings (for eyetracker and fNIRS),
* Data post-processing and analysis scripts,
* The 113 manually annotated bug localization tasks,
* The complete tables and figures that are presetned particially or not have space to be presented in the paper due to the page limitation.

# code snippets
This folder contains the eight code snippets that employed in our experiment. The following table displays the origin, linearity, complexity, self-reported difficulty, and correctness rate for each code snippets.  

## Table I: Characteristics of the Code Snippets Used for the Experiment

| Code snippet   | Origin                     | Linearity | Complexity | Difficulty | Correctness |
|----------------|----------------------------|-----------|------------|------------|-------------|
| `moneyclass`   | Synthesized*               | 0         | 1          | 1.18       | 90%         |
| `numberchecker`| Synthesized*               | 9.9       | 2          | 1.67       | 90%         |
| `calculation`  | Synthesized*               | 1.64      | 1.5        | 2          | 75%         |
| `numberhrd`    | Open Source                | 0         | 9          | 2.38       | 74%         |
| `rectangle`    | Synthesized*               | 20        | 1          | 1.67       | 71%         |
| `numberhrn`    | Open Source                | 0         | 7          | 2.93       | 61%         |
| `insertion`    | Synthesized*               | 3.24      | 3.5        | 2.94       | 56%         |
| `graph`        | Open Source                | 19.23     | 2          | 3.62       | 31%         |

\* The code snippets whose origin is synthesized are collected from "Peitek, Norman, Janet Siegmund, and Sven Apel. "What drives the reading order of programmers? an eye tracking study." Proceedings of the 28th International Conference on Program Comprehension. 2020."

**Note:** Snippets are arranged in descending order based on their Correctness Rate. 



# data
This folder contains the interim and processed data we gathered during analysis, which will be used in our scripts. 

# device-configuration
This folder contains the detailed infomation of settings we employed for our eyetracking and fNIRs devices.

# manual-annotation
This folder contains the complete manual annotation results of 113 tasks completed by 53 participants.

# notebooks and scripts
These two folders contains all scripts that we used to perform the analysis conducted in RQ1, RQ2, and RQ3. We name the files in the order usage sequences. The next sub-section provides the enviroment requestment to run the code:
## Setup
1. Install [pdm](https://pdm.fming.dev/latest/).
2. Run `pdm install` to install all dependencies

# tables-and-figures
## pattern-characteristics-examples
Due to page limitations, we only provide examples for some values of the characteristics in the paper. A complete set of annotated examples illustrating each characteristic value is included in this folder. The following table provides a quick view of the 4 characteristics and their values.

| Characteristics | Values                    | 
|----------------|----------------------------|
| Level of Investment  | Low, Medium, High *       |
| Direction - Data flow (DF) | Forward, Backward, Back and forth the order of data flow   |
| Direction - Order of Definition (OoD) | Forward, Backward, Back and forth the order of definition  **    |
| Statement Selection | Linear, Dynamic      |
| Code Interaction | Inter-Procedural, Intra Procedural      |

**Terminology Mapping**  For * Level of Invesment and ** Direction, the terminology we leveraged during analysis is a bit different from the ones we presented in the paper. Here, we would like to provide a mapping to migrate inconsistancy: 

For * **Level of Invesment** : **Low = Quick**, Medium = Medium, **High = Indepth**

For ** **Direction - Order of Definition (OoD)**: **Forward the order of definition = In the order of definition**, **Back the order of definition = Reverse the order of definition**, Back and forth the order of definition  = Back and forth the order of definition


## tables
This folder consists of the complete tables that we might not get the change to shown in the paper due to space limitation:.




