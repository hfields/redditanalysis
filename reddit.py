"""
Author      : Yi-Chieh Wu
Class       : HMC CS 158
Date        : 2019 Feb 22
Description : Twitter
Huey Fields and Hakan Alpan
"""

from string import punctuation
from nltk.stem.wordnet import WordNetLemmatizer

# numpy libraries
import numpy as np
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

# matplotlib libraries
import matplotlib.pyplot as plt

# scikit-learn libraries
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer

######################################################################
# functions -- input/output
######################################################################

def read_vector_file(fname) :
    """
    Reads and returns a vector from a file.

    Parameters
    --------------------
        fname  -- string, filename

    Returns
    --------------------
        labels -- numpy array of shape (n,)
                    n is the number of non-blank lines in the text file
    """
    return np.genfromtxt(fname)


def write_label_answer(vec, outfile) :
    """
    Writes your label vector to the given file.

    Parameters
    --------------------
        vec     -- numpy array of shape (n,) or (n,1), predicted scores
        outfile -- string, output filename
    """

    # for this project, you should predict 70 labels
    if(vec.shape[0] != 70):
        print("Error - output vector should have 70 rows.")
        print("Aborting write.")
        return

    np.savetxt(outfile, vec)


######################################################################
# functions -- feature extraction
######################################################################

def extract_words(input_string) :
    """
    Processes the input_string, separating it into "words" based on the presence
    of spaces, and separating punctuation marks into their own words.

    Parameters
    --------------------
        input_string -- string of characters

    Returns
    --------------------
        words        -- list of lowercase "words"
    """

    for c in punctuation :
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def extract_dictionary(infile) :
    """
    Given a filename, reads the text file and builds a dictionary of unique
    words/punctuations.

    Parameters
    --------------------
        infile    -- string, filename

    Returns
    --------------------
        word_list -- dictionary, (key, value) pairs are (word, index)
    """
    stop_words=set(stopwords.words("english"))
    lem = WordNetLemmatizer()

    word_list = {}
    with open(infile, 'r') as fid :
        # part 1a: process each line to populate word_list
        # Create a variable to keep track of the index of each unique word
        index = 0

        for line in fid:
            # Call extract_words to separate the line into words
            #words = pos_tag(word_tokenize(line))
            words = extract_words(line)
            filter=[]


            for w in words:
               # wntag = t[0].lower()
                #wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
                #if not wntag:
                #    lemma = w
                #else:
                #    lemma = lem.lemmatize(w, wntag)
                #    print(w,lemma)
               # w = lemma
                if w not in stop_words:
                    filter.append(w)

            for word in filter:
                # If a word has not yet been encountered, save it in the dict and increment index
                word = word.lower()
                if word not in word_list.keys():
                    word_list[word] = index
                    index += 1

    #print(word_list)
    print('aaaaaaaaaa')
    return word_list


def extract_feature_vectors(infile, word_list) :
    """
    Produces a bag-of-words representation of a text file specified by the
    filename infile based on the dictionary word_list.

    Parameters
    --------------------
        infile         -- string, filename
        word_list      -- dictionary, (key, value) pairs are (word, index)

    Returns
    --------------------
        feature_matrix -- numpy array of shape (n,d)
                          boolean (0,1) array indicating word presence in a string
                            n is the number of non-blank lines in the text file
                            d is the number of unique words in the text file
    """

    num_lines = sum(1 for line in open(infile,'rU'))
    print(num_lines)
    print(word_list)
    num_words = len(word_list)
    feature_matrix = np.zeros((num_lines, num_words))

    with open(infile, 'rU') as fid :
        # part 1b: process each line to populate feature_matrix
        # Create a variable to keep track of the index of the lines in the feature_matrix
        index = 0

        for line in fid:
            # Call extract_words to separate the line into words
            words = extract_words(line)

            for word in words:
                # If the word in the tweet is in the word list, set the corresponding index of the feature_matrix to 1
                if word in word_list.keys():
                    feature_matrix[index][word_list[word]] = 1

            index += 1

    return feature_matrix


def test_extract_dictionary(dictionary) :
    err = "extract_dictionary implementation incorrect"

    assert len(dictionary) == 1811, err

    exp = [('2012', 0),
           ('carol', 10),
           ('ve', 20),
           ('scary', 30),
           ('vacation', 40),
           ('just', 50),
           ('excited', 60),
           ('no', 70),
           ('cinema', 80),
           ('frm', 90)]
    act = [sorted(dictionary.items(), key=lambda it: it[1])[i] for i in range(0,100,10)]
    assert exp == act, err


def test_extract_feature_vectors(X) :
    err = "extract_features_vectors implementation incorrect"

    assert X.shape == (630, 1811), err

    exp = np.array([[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
                    [ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.],
                    [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]])
    act = X[:10,:10]
    assert (exp == act).all(), err


######################################################################
# functions -- evaluation
######################################################################

def performance(y_true, y_pred, metric="accuracy") :
    """
    Calculates the performance metric based on the agreement between the
    true labels and the predicted labels.

    Parameters
    --------------------
        y_true -- numpy array of shape (n,), known labels
        y_pred -- numpy array of shape (n,), (continuous-valued) predictions
        metric -- string, option used to select the performance measure
                  options: 'accuracy', 'f1_score', 'auroc', 'precision',
                           'sensitivity', 'specificity'

    Returns
    --------------------
        score  -- float, performance score
    """
    # map continuous-valued predictions to binary labels
    y_label = np.sign(y_pred)
    y_label[y_label==0] = 1 # map points of hyperplane to +1

    # part 2a: compute classifier performance
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_label)

    elif metric == "f1_score":
        return metrics.f1_score(y_true, y_label)

    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred)

    elif metric == "precision":
        return metrics.precision_score(y_true, y_label)

    elif metric == "sensitivity":
        confMatrix = metrics.confusion_matrix(y_true, y_label, labels=[1, -1])
        return float(confMatrix[0][0]) / (confMatrix[0][0] + confMatrix[0][1])

    elif metric == "specificity":
        confMatrix = metrics.confusion_matrix(y_true, y_label, labels=[1, -1])
        return float(confMatrix[1][1]) / (confMatrix[1][1] + confMatrix[1][0])


def test_performance() :
    # np.random.seed(1234)
    # y_true = 2 * np.random.randint(0,2,10) - 1
    # np.random.seed(2345)
    # y_pred = (10 + 10) * np.random.random(10) - 10

    y_true = [ 1,  1, -1,  1, -1, -1, -1,  1,  1,  1]
    #y_pred = [ 1, -1,  1, -1,  1,  1, -1, -1,  1, -1]
    # confusion matrix
    #          pred pos     neg
    # true pos      tp (2)  fn (4)
    #      neg      fp (3)  tn (1)
    y_pred = [ 3.21288618, -1.72798696,  3.36205116, -5.40113156,  6.15356672,
               2.73636929, -6.55612296, -4.79228264,  8.30639981, -0.74368981]
    metrics = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]
    scores  = [     3/10.,      4/11.,   5/12.,        2/5.,          2/6.,          1/4.]

    import sys
    eps = sys.float_info.epsilon

    for i, metric in enumerate(metrics) :
        assert abs(performance(y_true, y_pred, metric) - scores[i]) < eps, \
            (metric, performance(y_true, y_pred, metric), scores[i])
    print('done')

def cv_performance(clf, X, y, kf, metrics=["accuracy"]) :
    """
    Splits the data, X and y, into k-folds and runs k-fold cross-validation.
    Trains classifier on k-1 folds and tests on the remaining fold.
    Calculates the k-fold cross-validation performance metric for classifier
    by averaging the performance across folds.

    Parameters
    --------------------
        clf     -- classifier (instance of SVC)
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metrics -- list of m strings, metrics

    Returns
    --------------------
        scores  -- numpy array of shape (m,), average CV performance for each metric
    """

    k = kf.get_n_splits(X, y)
    m = len(metrics)
    scores = np.empty((m, k))

    for k, (train, test) in enumerate(kf.split(X, y)) :
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        clf.fit(X_train, y_train)
        # use SVC.decision_function to make ``continuous-valued'' predictions
        y_pred = clf.decision_function(X_test)
        for m, metric in enumerate(metrics) :
            score = performance(y_test, y_pred, metric)
            scores[m,k] = score

    return scores.mean(axis=1) # average across columns


def select_param_linear(X, y, kf, metrics=["accuracy"], plot=True) :
    """
    Sweeps different settings for the hyperparameter of a linear-kernel SVM,
    calculating the k-fold CV performance for each setting and metric,
    then selects the hyperparameter that maximizes the average performance for each metric.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metrics -- list of m strings, metrics
        plot    -- boolean, make a plot

    Returns
    --------------------
        params  -- list of m floats, optimal hyperparameter C for each metric
    """

    C_range = 10.0 ** np.arange(-3, 3)
    scores = np.empty((len(metrics), len(C_range)))
    best_params = np.empty(len(metrics))

    # part 3b: for each metric, select optimal hyperparameter using cross-validation
    for i in range(len(C_range)):
        c = C_range[i]
        clf = SVC(C=c,kernel='linear')
        # compute CV scores using cv_performance(...)
        sc = cv_performance(clf,X,y,kf,metrics=metrics)

        for j in range(len(sc)):
            score = sc[j]
            scores[j][i] = score

    # get best hyperparameters
    for i in range(len(scores)):
        metric = scores[i]
        print(np.amax(metric))
        best_params[i] = C_range[np.argmax(metric)]

    # plot
    if plot:
        plt.figure()
        ax = plt.gca()
        ax.set_ylim(0, 1)
        ax.set_xlabel("C")
        ax.set_ylabel("score")
        for m, metric in enumerate(metrics) :
            lineplot(C_range, scores[m,:], metric)
        plt.legend()
        plt.show()

    return best_params


def select_param_rbf(X, y, kf, metrics=["accuracy"]) :
    """
    Sweeps different settings for the hyperparameters of an RBF-kernel SVM,
    calculating the k-fold CV performance for each setting and metric,
    then selects the hyperparameters that maximize the average performance for each metric.

    Parameters
    --------------------
        X       -- numpy array of shape (n,d), feature vectors
                     n = number of examples
                     d = number of features
        y       -- numpy array of shape (n,), binary labels {1,-1}
        kf      -- model_selection.KFold or model_selection.StratifiedKFold
        metrics -- list of m strings, metrics

    Returns
    --------------------
        params  -- list of m tuples, optimal hyperparameters (C,gamma) for each metric
    """

    # part 4b: for each metric, select optimal hyperparameters using cross-validation

    # create grid of hyperparameters
    # hint: use a small 2x2 grid of hyperparameters for debugging
    C_range = 10.0 ** np.arange(-8, 8)
    gamma_range = 10.0 ** np.arange(-3, 3)
    scores = np.empty((len(metrics), len(C_range), len(gamma_range)))
    best_params = np.empty(len(metrics), dtype = (float, 2))

    # Iterate through each combination of hyperparameters
    for i in range(len(C_range)):
        c = C_range[i]
        for j in range(len(gamma_range)):
            gamma = gamma_range[j]
            clf = SVC(C=c, gamma=gamma, kernel='rbf')
            # compute CV scores using cv_performance(...)
            sc = cv_performance(clf,X,y,kf,metrics=metrics)

            for k in range(len(sc)):
                score = sc[k]
                scores[k][i][j] = score

    # get best hyperparameters
    for i in range(len(scores)):
        metric = scores[i]
        indices = np.unravel_index(np.argmax(metric), metric.shape)
        print(np.amax(metric))
        best_params[i] = (C_range[indices[0]], gamma_range[indices[1]])

    return best_params


def performance_CI(clf, X, y, metric="accuracy") :
    """
    Estimates the performance of the classifier using the 95% CI.

    Parameters
    --------------------
        clf          -- classifier (instance of SVC or DummyClassifier)
                          [already fit to data]
        X            -- numpy array of shape (n,d), feature vectors of test set
                          n = number of examples
                          d = number of features
        y            -- numpy array of shape (n,), binary labels {1,-1} of test set
        metric       -- string, option used to select performance measure

    Returns
    --------------------
        score        -- float, classifier performance
        lower        -- float, lower limit of confidence interval
        upper        -- float, upper limit of confidence interval
    """

    try :
        y_pred = clf.decision_function(X)
    except :  # for dummy classifiers
        y_pred = clf.predict(X)
    score = performance(y, y_pred, metric)

    # part 5c: use bootstrapping to compute 95% confidence interval
    # hint: use np.random.randint(...) to sample
    tval = 100
    t = tval
    n = X.shape[0]
    bootstrap_scores = []

    # Bootstrap t times
    while t > 0:
        # Find n random indices and find a corresponding array of n true vals and predictions
        random_indices = np.random.randint(0, n, n)
        y_boot = np.take(y, random_indices)
        y_pred_boot = np.take(y_pred, random_indices)

        # Find performance metric with this bootstrap attempt
        bootstrap_scores += [performance(y_boot, y_pred_boot, metric)]

        t -= 1

    t = tval
    bootstrap_scores.sort()

    # Return tuple with score and 2.5th and 97.5th percentiles of bootstrap scores
    return score, bootstrap_scores[int(t * 0.025)], bootstrap_scores[int(t * 0.975)]


######################################################################
# functions -- plotting
######################################################################

def lineplot(x, y, label):
    """
    Make a line plot.

    Parameters
    --------------------
        x            -- list of doubles, x values
        y            -- list of doubles, y values
        label        -- string, label for legend
    """

    xx = range(len(x))
    plt.plot(xx, y, linestyle='-', linewidth=2, label=label)
    plt.xticks(xx, x)


def plot_results(metrics, classifiers, *args):
    """
    Make a results plot.

    Parameters
    --------------------
        metrics      -- list of strings, metrics
        classifiers  -- list of strings, classifiers (excluding baseline classifier)
        args         -- variable length argument
                          results for baseline
                          results for classifier 1
                          results for classifier 2
                          ...
                        each results is a list of tuples ordered by metric
                          typically, each tuple consists of a single element, e.g. (score,)
                          to include error bars, each tuple consists of three elements, e.g. (score, lower, upper)
    """

    num_metrics = len(metrics)
    num_classifiers = len(args) - 1

    ind = np.arange(num_metrics)  # the x locations for the groups
    width = 0.7 / num_classifiers # the width of the bars

    fig, ax = plt.subplots()

    # loop through classifiers
    rects_list = []
    for i in range(num_classifiers):
        results = args[i+1] # skip baseline

        # mean
        means = [it[0] for it in results]
        rects = ax.bar(ind + i * width, means, width, label=classifiers[i])
        rects_list.append(rects)

        # errors
        if len(results[0]) == 3:
            errs = [(it[0] - it[1], it[2] - it[0]) for it in results]
            ax.errorbar(ind + i * width, means, yerr=np.array(errs).T, fmt='none', ecolor='k')

    # baseline
    results = args[0]
    for i in range(num_metrics) :
        xlim = (ind[i] - 0.8 * width, ind[i] + num_classifiers * width - 0.2 * width)

        # mean
        mean = results[i][0]
        plt.plot(xlim, [mean, mean], color='k', linestyle='-', linewidth=2)

        # errors
        if len(results[i]) == 3:
            err_low = results[i][1]
            err_high = results[i][2]
            plt.plot(xlim, [err_low, err_low], color='k', linestyle='--', linewidth=2)
            plt.plot(xlim, [err_high, err_high], color='k', linestyle='--', linewidth=2)

    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.set_xticks(ind + width / num_classifiers)
    ax.set_xticklabels(metrics)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar displaying its height"""
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    '%.2f' % height, ha='center', va='bottom')

    for rects in rects_list:
        autolabel(rects)

    plt.show()


######################################################################
# main
######################################################################

def main() :
    # read the comments and their labels, without unit tests
    comments = []
    f = open("anacigin.txt")

    for line in f:
        comments += [line]

    vectorizer = TfidfVectorizer()
    response = vectorizer.fit_transform(comments)

    dictionary = vectorizer.get_feature_names()

    X = response
    y = read_vector_file('anneannen.txt')

    # shuffle data (since file has comments ordered by bernie)
    X, y = shuffle(X, y, random_state=0)

    # set random seed
    np.random.seed(1234)

    # split the data into training (training + cross-validation) and testing set
    test_size = 1000
    X_train, X_test = X[:test_size], X[test_size:]
    y_train, y_test = y[:test_size], y[test_size:]

    # part 2a: metrics, with unit test
    # (nothing to implement, just make sure the test passes)
    metrics = ["accuracy", "f1_score", "auroc", "precision", "sensitivity", "specificity"]

    # folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=np.random.randint(1234))

    # hyperparameter selection for linear-SVM
    # c = 1 is the best for everything except sensitivity
    best_params = select_param_linear(X_train, y_train, skf, metrics)

    # hyperparameter selection using RBF-SVM
    # c = 100 and gamma = 0.01 is best for everything except sensitivity
    #best_params_rbf = select_param_rbf(X_train, y_train, skf, metrics)

    print(best_params)
    #print(best_params_rbf)

    # part 5a: train linear- and RBF-SVMs with selected hyperparameters
    # hint: use only linear-SVM (comment out the RBF-SVM) for debugging

    # Create and train baseline, linear, and rbf classifiers
    baseline = DummyClassifier()
    baseline.fit(X_train, y_train)

    clf_linear = SVC(C=0.1, kernel='linear', class_weight = 'balanced')
    clf_linear.fit(X_train, y_train)

    clf_rbf = SVC(C=0.1, gamma=0.01, kernel='rbf',class_weight = 'balanced')
    clf_rbf.fit(X_train, y_train)

    clf_rfc = RandomForestClassifier(max_depth = 100, class_weight = 'balanced')
    clf_rfc.fit(X_train,y_train)

    # part 5b: report performance on train data
    #          use plot_results(...) to make plot
    classifiers = ["linear", "rbf", "rfc"]

    # Predict y vals for all classifiers
    y_pred_baseline = baseline.predict(X_train)
    y_pred_linear = clf_linear.predict(X_train)
    y_pred_rbf = clf_rbf.predict(X_train)
    y_pred_rfc = clf_rfc.predict(X_train)

    # Keep a list of results for each metric
    results_baseline = []
    results_linear = []
    results_rbf = []
    results_rfc = []

    for metric in metrics:
        results_baseline += [(performance(y_train, y_pred_baseline, metric),)]
        results_linear += [(performance(y_train, y_pred_linear, metric),)]
        results_rbf += [(performance(y_train, y_pred_rbf, metric),)]
        results_rfc += [(performance(y_train, y_pred_rfc, metric),)]

    plot_results(metrics, classifiers, results_baseline, results_linear, results_rbf, results_rfc)

    # part 5d: use bootstrapping to report performance on test data
    #          use plot_results(...) to make plot

    # Keep a list of confidence intervals for each metric
    CIs_baseline = []
    CIs_linear = []
    CIs_rbf = []
    CIs_rfc = []

    for metric in metrics:
        CIs_baseline += [(performance_CI(baseline, X_test, y_test, metric))]
        CIs_linear += [(performance_CI(clf_linear, X_test, y_test, metric))]
        CIs_rbf += [(performance_CI(clf_rbf, X_test, y_test, metric))]
        CIs_rfc += [(performance_CI(clf_rfc, X_test, y_test, metric))]

    plot_results(metrics, classifiers, CIs_baseline, CIs_linear,CIs_rbf, CIs_rfc)

    ### ========== TODO : START ========== ###
    # part 6: identify important features
    #invDict = dict((v, k) for k, v in dictionary.items())s

    #features = []

    #for index in np.argsort(clf_linear.coef_)[0]:
    #    features += [invDict[index]]

    #print(features[0:10])
    #print(features[-11:])

    ### ========== TODO : END ========== ###

    ### ========== TODO : START ========== ###
    # Twitter contest
    # uncomment out the following, and be sure to change the filename
    """
    X_held = extract_feature_vectors('../data/held_out_tweets.txt', dictionary)
    # your code here
    # y_pred = best_clf.decision_function(X_held)
    write_label_answer(y_pred, '../data/yjw_twitter.txt')
    """
    ### ========== TODO : END ========== ###


if __name__ == "__main__" :
    main()
