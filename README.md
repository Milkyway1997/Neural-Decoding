# Neural-Decoding

This neural-decoding project is a research that I had been working on for about a year as my master's thesis topic. If you are interested in our study, you can look at the posters in the 'Posters' folder.

The author of all the scripts except 'WU_MEG_DP.py' is Li Liu.

'nd_lib_v3'

➢ is the main library for steps before training machine learning models, 
including feature scaling, data labeling, and data shuffling. 

➢ is also the library for other procedures that aim to enhance 
classification accuracy, including Slicing Window, PCA, trials 
averaging, doubling data, CWT, cross-subjects, etc.

➢ also contained machine learning/Deep Learning stuff, including different classifiers 
(SVM, NN, GNB, LDA, RVM), cross-validation, grid search, CNN, etc.

'WU_MEG_DP.py' was written by Dr. Dmitry Patashov.
'WU_MEG_DP' is the main library for preprocessing, including 
filtering, resample, EMD, outlier removal, epoching, flipping, 
clustering, combination's averaging, etc.

'Post_Processing' is another library used for plotting the graphs, 
including Graph Per Window Size, bar chart, MVPA, classifiers 
comparison, AccPerObj, statistical graph, etc. Like this:

![Object_Classification](https://user-images.githubusercontent.com/73594399/204471918-e019126f-d072-451f-8681-46e18023b68e.png)

All those 'ipynb' files are the scripts that conduct some of the specific features written in those libraries. 

Files in the 'saved results' folder are all kinds of classification accuracy after executing some of the functions from the main libraries, and you can use them to plot some graphs for fun.

Unfortunately, due to the size of the raw data, I can't upload the raw data we collected from the experiment conducted in August 2021. Therefore, you can only test some features built for creating graphs using saved results.

PS 1: Sorry about the terrible naming and lack of comments. 
PS 2: We are preparing an article/thesis related to this research; if we publish the paper anywhere, I will put a link here later.
