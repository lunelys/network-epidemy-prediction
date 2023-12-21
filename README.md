# network-epidemy-prediction
Script to predict influenza outbreak events via tweets, based on the research of L. Zhao &amp; al., 2015 [1]

## Context
The anticipation and comprehension of epidemic spread emerge as pivotal factors. For instance, influenza kills 389,000 individuals globally each year. Examining past situations like the COVID pandemic highlights the potential benefits of predictive measures. Such foresight could aid hospitals in readiness, faster procurement of supplies, and efficient execution of vaccination campaigns.
A potentially interesting way for acquiring relevant data involves the utilization of social media platforms. Many individuals employ these platforms to communicate their health status, presenting an unconventional but potentially valuable source for data gathering. This approach, if harnessed effectively, could contribute to the monitoring and prediction of epidemics.

## Data
The dataset, extracted from the research conducted by L. Zhao et al. in 2015, spans the period from January 1, 2011, to April 1, 2015. The data encompasses 48 U.S. states, recorded on a daily basis, and includes occurrences of specific keywords associated with influenza. Additionally, the dataset denotes the presence or absence of an epidemic state in the week following the recorded data points, using a binary classification (0 for no epidemic state, 1 for epidemic state). Notably, this data originates from a collection of 525 flu-related keywords extracted from over 52,000 tweets, gathered through the Twitter API.

## Approach
A PCA, a decision tree, a random forest and finally a simple neural network (2 hidden layers with 5 neurons in the first layer and 2 neurons in the second layer) have been created in this project to analyze the data from the research of L. Zhao & al., 2015 [1] and propose a way to build a prediction model forecasting influenza outbreak via Twitter keywords.

[1] L. Zhao, J. Chen, F. Chen, W. Wang, C. -T. Lu and N. Ramakrishnan, "SimNest: Social Media Nested Epidemic Simulation via Online Semi-Supervised Deep Learning," 2015 IEEE International Conference on Data Mining, Atlantic City, NJ, USA, 2015, pp. 639-648, doi: 10.1109/ICDM.2015.39.
