# Neural-Decoding

This neural-decoding project is a research that I had been working on 
for about a year as my master's thesis topic. If you are interested in 
our research, you can look at the posters in the 'Posters' folder.

The 'nd_lib_v3.py' file is the main library written by me. It contains many features, 
including feature scaling, data labeling, data splitting, trials averaging, PCA, 
feature extraction, training/testing ML models, etc. 

On the other hand, Dmitry Patashov created the 'WU_MEG_DP.py' file. 
He is a post-doctor researcher at Waseda University, 
and we have been working together for about a year. 
'WU_MEG_DP' is the library for preprocessing, including epoching, 
filtering, outliers removal, etc.  

'Post_Processing.py' is the library used for creating the graphs for
evaluating the results we got from ML classifiers, including bar chart, 
time-series graph, etc. 

All those jupyter files (ipynb) are the scripts that 
run all the features written in the libraries.

'pkl' files are some results (a tiny part of all the results we got) after 
executing all the features from those main libraries,  
and you can use them to plot some graphs for fun.

Unfortunately, I can't upload the raw data 
we collected from the experiment we did in August 2021. 
Therefore, you cannot test all the features using those scripts from scratch yourself.

PS 1: Sorry about the terrible naming and lack of comments. 

PS 2: We are writing an article/thesis related to this research; if we publish the paper
anywhere, I will put a link here later.
