#The following function performs stratified sampling. 
#data: DataFrame, the original data
#y: string, name of the label attribute
#train_fraction: float, fraction of the training set, e.g. 0.8
#We assume that y has two values 0,1.
#This function returns for DataFrames X_train, X_test, y_train, y_test 

def test(str):
    print(str)
    
def stratified_sample(data,y,train_fraction):

    import pandas as pd

    data1=data[data[y]==1] 
    data0=data[data[y]==0] 

    train1=data1.sample(frac=train_fraction,random_state=42) 
    test1=data1.drop(train1.index)

    train0=data0.sample(frac=train_fraction,random_state=42) 
    test0=data0.drop(train0.index)

    train=pd.concat([train1,train0]) 
    test=pd.concat([test1,test0])

    X_train = train.drop([y], axis = 1) 
    y_train = train[y] 

    X_test = test.drop([y], axis = 1) 
    y_test = test[y] 
    
    return X_train, X_test, y_train, y_test

def print_cat(data0, columns):
    import pandas as pd
    for i in columns:
        print(i)
        data0[i].value_counts()

def cat_to_dummy(data,list):
    import pandas as pd
    
    for i in list:
        data=pd.concat([data, pd.get_dummies(data[i], prefix = i)], axis = 1)
    data=data.drop(list, axis = 1)
    
    return data


## This function create dummy variables instead of 1 or 0, it weighs based on unique values
## example: education = ['secondary', 'tertiary', 'unknown' ]; If row contain secondary it sets = 0.33 (1/3) 
def cat_to_dummy_wt_adjustment(data_orig, data,list):
    import pandas as pd
    
    for i in list:
        data=pd.concat([data, pd.get_dummies(data[i], prefix = i)/(len(data_orig[i].unique()))], axis = 1)
    data=data.drop(list, axis = 1)
    
    return data

#
def print_details(grp, col, data_summary):
    import pandas as pd
    data_crosstab = pd.crosstab(data_summary[grp],data_summary[col],margins = True)
    print(data_crosstab)
    data_summary.groupby(grp)[col].hist(alpha=0.6, legend=True, stacked=True, figsize=[20, 15])
    
def print_num_details(grp, col, data_summary):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from itertools import zip_longest

    ageBoxplot = data_summary.boxplot(column=[col],  by=grp, showmeans=True, figsize=[20, 15])
    ageBoxplot

    # Dictionary of color for each species
    color_d = dict(zip_longest(data_summary.labels.unique(), 
                           plt.rcParams['axes.prop_cycle'].by_key()['color']))

    # Use the same bins for each
    xmin = data_summary.age.min()
    xmax = data_summary.age.max()
    bins = np.linspace(xmin, xmax, 20)

    # Set up correct number of subplots, space them out. 
    fig, ax = plt.subplots(nrows=data_summary.labels.nunique(), figsize=(10,16))
    plt.subplots_adjust(hspace=0.4)
    plt.xlabel(col)
    plt.ylabel('Count')

    for i, (lab, gp) in enumerate(data_summary.groupby('labels')):
        ax[i].hist(gp.age, ec='k', bins=bins, color=color_d[lab])
        ax[i].set_title(str(col) + " Cluster = " + str(lab))

        # same xlim for each so we can see differences
        ax[i].set_xlim(xmin, xmax)
        
def print_cat_pie(data0):
    import pandas as pd
    import matplotlib.pyplot as plt
    cat_cols = data0.select_dtypes(include=object).columns.tolist()
    for i in cat_cols:
        #print("------------------ Data for " + i + "--------------------------")
        a  = data0[i].value_counts().sort_index().plot(kind='pie', autopct='%1.2f%%', figsize=(10,7))
        plt.title(i.title(), fontweight ="bold", fontsize=20)
        plt.show()
        print(data0[i].value_counts(normalize=False))
        #print("---------------------------------------------------------------")
        a.clear()

        
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    '''
    source: https://github.com/DTrimarchi10/confusion_matrix
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title, fontweight ="bold")