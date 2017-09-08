from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, make_scorer, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV
import visuals_PCA as vs_pca
import time
import matplotlib.pyplot as plt
import visuals as vs
import numpy as np
import pandas as pd



# Set the learning curve parameters; you'll need this for learning_curves

def plot_mean_split_score(data, labels, reglist, n_reps):
    # repeats the train_test_split n_reps and calulates r2 score with learners from reglist
    # the mean r2 score is calculated and r2 score for every iteration shown in figure
    a = n_reps

    for e in reglist:
        score_l = []
        print e
        for i in range(0, a, 1):
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)   # , random_state=23
            y_train = y_train.values.ravel()  # change column vector to 1d array to avoid conversion warning @ regressor.fit()

            regressor = e
            regressor.fit(X_train, y_train)
            score = regressor.score(X_test, y_test)
            score_l.append(score)
        avg_score = np.mean(score_l)
        print('average R2 score for all {} splits'.format(a), avg_score)

        plt.figure(figsize=(5, 3))
        plt.plot(score_l)
        plt.ylabel("R2 Score")
        plt.xlabel("number of splits")
        plt.show()

        print('{} = (datapoints, features) for training'.format(np.shape(X_train)))
        print('{} = (datapoints, features) for testing'.format(np.shape(X_test)))
        print('')

def plot_learn_curve(X_train, y_train, X_test, y_test, reglist):

    for e in reglist:
        print e
        e.fit(X_train, y_train)
        print "Regressor R2 score on the test set: {:.4f}".format(e.score(X_test,y_test))
        print('size of the test set (x,y)', np.shape(X_test), np.shape(y_test))

        # TODO: Use learning_curve imported above to create learning curves for both the
        #       training data and testing data. You'll need 'size', 'cv' and 'score' from above.

        train_sizes, train_scores, test_scores = learning_curve(
            e, X_train, y_train, cv=KFold(n_splits=10),
            scoring=make_scorer(r2_score),
            train_sizes=np.linspace(.1, 1, 20), n_jobs=8)

        # TODO: Plot the training curves and the testing curves
        #       Use plt.plot twice -- one for each score. Be sure to give them labels!

        plt.figure(figsize=(10, 7))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, color="g",
                 label="Cross-validation score")

        # Plot aesthetics
        plt.ylim(-1.1, 1.1)
        plt.ylabel("R2 Score")
        plt.xlabel("Training Points")
        plt.legend(bbox_to_anchor=(1.0, 1.15))
        plt.show()
        try:
            importances = e.feature_importances_
            vs.feature_plot(importances, X_train, y_train)
        except AttributeError:
            print('No feature importance avalable for this learner')
        print('')
    return

def plot_time_split_score(features, labels, valid_features, valid_labels, reglist, n_TSSplits):
#def plot_time_split_score(features, labels, reglist, n_TSSplits):
    tscv = TimeSeriesSplit(n_splits=n_TSSplits)

    for e in reglist:
        score_l = []
        print e
        for train_index, test_index in tscv.split(features):
            X_train, X_test = features.loc[train_index], features.loc[test_index]
            y_train, y_test = labels.loc[train_index], labels.loc[test_index]
            y_train = y_train.values.ravel()  # change column vector to 1d array to avoid conversion warning @ regressor.fit()

            regressor = e
            regressor.fit(X_train, y_train)
            score = regressor.score(X_test, y_test)
            score_l.append(score)
        print('')
        print "Regressor R2 score on the test set: {:.4f}".format(score)
        #print('size of the training set (x,y)', np.shape(X_train), np.shape(y_train))
        #print('size of the test set (x,y)', np.shape(X_test), np.shape(y_test))
        print('')

        if score > -20.20:

            preds = regressor.predict(features)
            preds = pd.DataFrame(preds)

            #print('mean of orig. sr. (validation data set)', valid_labels.sr_highres.mean())
            print('mean of predicted sr. (validation data set)', preds.values.mean())

            plt.figure(figsize=(6, 3))
            plt.plot(score_l)
            plt.ylabel("R2 Score")
            plt.xlabel("number of TimeSeriesSplits")
            plt.show()

            fig = plt.figure(figsize=(18, 3))
            labels['sr_highres'].plot()
            preds[0].plot()
            plt.ylim(-4, 4)
            plt.title('labels vs. predictions')
            plt.legend(loc='best')
            plt.show()
            print('test set labels mean sr:', labels.mean())
            print('test set predicted mean sr:', preds.mean())

            preds2 = regressor.predict(valid_features)
            preds2 = pd.DataFrame(preds2)
#
            fig = plt.figure(figsize=(18, 3))
            valid_labels['sr_highres'].plot()
            preds2[0].plot()
            plt.ylim(-4, 4)
            plt.show()
            print('validation set labels mean sr:', valid_labels.mean())
            print('validation set predicted mean sr:', preds2.mean())
            try:
                importances = regressor.feature_importances_
                vs.feature_plot(importances, X_train, y_train)
            except AttributeError:
                print('')
                print('No feature importance avalable for this learner')
                print('')
                print('')

    return regressor, preds

def plot_kfold_split_score(features, labels, valid_features, valid_labels, reglist, n_Splits):
    kfold = KFold(n_splits=n_Splits, random_state=0, shuffle=True)

    for e in reglist:
        score_l = []
        print e
        for train_index, test_index in kfold.split(features):
            X_train, X_test = features.loc[train_index], features.loc[test_index]
            y_train, y_test = labels.loc[train_index], labels.loc[test_index]
            y_train = y_train.values.ravel()  # change column vector to 1d array to avoid conversion warning @ regressor.fit()

            regressor = e
            start = time.time()
            regressor.fit(X_train, y_train)
            elapsed = time.time() - start
            print("time to fit: %f" % (elapsed))
            score = regressor.score(X_test, y_test)
            score_l.append(score)
        print('')
        print "Regressor R2 score on the validation set: {:.4f}".format(score)
        # print('size of the training set (x,y)', np.shape(X_train), np.shape(y_train))
        # print('size of the test set (x,y)', np.shape(X_test), np.shape(y_test))
        print('')

        if score > 0.2:

            preds = regressor.predict(features)
            preds = pd.DataFrame(preds)

            # print('mean of orig. sr. (validation data set)', valid_labels.sr_highres.mean())
            print('mean of predicted sr. (validation data set)', preds.values.mean())

            plt.figure(figsize=(6, 3))
            plt.plot(score_l)
            plt.ylabel("R2 Score")
            plt.xlabel("number of ShuffleSplits")
            plt.show()

            fig = plt.figure(figsize=(18, 3))
            labels['sr_highres'].plot()
            preds[0].plot()
            plt.ylim(-4, 4)
            plt.title('labels vs. predictions')
            plt.legend(loc='best')
            plt.show()

            print('validation set r2 score:', r2_score(labels['sr_highres'], preds[0]))
            print('validation set mean squared error: {0:.2f}%'.format(mean_squared_error(labels['sr_highres'], preds[0]) * 100))

            preds2 = regressor.predict(valid_features)
            preds2 = pd.DataFrame(preds2)

            fig = plt.figure(figsize=(18, 3))
            valid_labels['sr_highres'].plot()
            preds2[0].plot()
            plt.ylim(-4, 4)
            plt.show()

            print('test set r2 score:', r2_score(valid_labels['sr_highres'], preds2[0]))
            print('test set mean squared error: {0:.2f}%'.format(mean_squared_error(valid_labels['sr_highres'], preds2[0]) * 100))
            try:
                importances = regressor.feature_importances_
                vs.feature_plot(importances, X_train, y_train)
            except AttributeError:
                print('')
                print('No feature importance available for this learner')
                print('')
                print('')
            if type(e).__name__ == "XGBRegressor":
                fig = plt.figure(figsize=(15, 5))
                plot_importance(regressor)
                plt.show()

    return regressor, preds, preds2

def plot_shuffle_split_score(features, labels, features2, labels2, reglist, n_Splits, earlyStopRounds):
    sscv = ShuffleSplit(n_splits=n_Splits, test_size=.25, random_state=None)

    for e in reglist:
        score_l = []
        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')
        print e
        i = 0
        for train_index, test_index in sscv.split(features):
            print('---------------------------------------------------------------------------------------------------')
            print('ShuffledSplit iteration {} of {}'.format(i + 1, n_Splits))
            i += 1
            X_train, X_test = features.loc[train_index], features.loc[test_index]
            y_train, y_test = labels.loc[train_index], labels.loc[test_index]
            y_train = y_train.values.ravel()  # change column vector to 1d array to avoid conversion warning @ regressor.fit()

            regressor = e
            test_set = [(X_test, y_test), (features2, labels2)]
            #test_set = [(features2, labels2)]
            start = time.time()

            if type(e).__name__ in ("XGBRegressor", "MLPRegressor"):
                if earlyStopRounds > 0:
                    regressor.fit(X_train, y_train, early_stopping_rounds=earlyStopRounds, eval_metric='rmse',
                              eval_set=test_set, verbose=False)
                    elapsed = time.time() - start
                elif earlyStopRounds == 0:
                    print 'earlyStop disabled'
                    regressor.fit(X_train, y_train, eval_metric='rmse')
                    elapsed = time.time() - start

                results = regressor.evals_result()
                epochs = len(results['validation_0']['rmse'])
                x_axis = range(0, epochs)
                # plot regression error
                fig, ax = plt.subplots()
                ax.plot(x_axis, results['validation_0']['rmse'], label='Validation')
                ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
                ax.legend()
                plt.xlabel('number of epochs')
                plt.ylabel('Regression RMSE')
                plt.title('XGBReg. RMSE')
                plt.show()

            else:
                regressor.fit(X_train, y_train)
                elapsed = time.time() - start

            print("time to fit: %f" % (elapsed))
            score = regressor.score(X_test, y_test)
            score_l.append(score)

        print('')
        print "Regressor R2 score on the validation set: {:.4f}".format(score)
        print('---------------------------------------------------------------------')
        print('size of the training set (features, labels)', np.shape(X_train), np.shape(y_train))
        print('size of the validation set (features, labels)', np.shape(X_test), np.shape(y_test))
        print('size of the test set (features, labels)', np.shape(features2), np.shape(labels2))
        print('---------------------------------------------------------------------')
        print('Variance of the train/valid. set: {}'.format(labels['sr_highres'].var()))
        print('Variance of the test set: {}'.format(labels2['sr_highres'].var()))
        print('')

        preds = regressor.predict(features)
        preds = pd.DataFrame(preds)
        preds.rename(columns={0: 'sr_predicted'}, inplace=True)

        plt.figure(figsize=(6, 3))
        plt.plot(score_l)
        plt.ylabel("R2 Score")
        plt.xlabel("number of ShuffleSplits")
        plt.show()

        fig = plt.figure(figsize=(18, 3))
        labels['sr_highres'].plot()
        preds['sr_predicted'].plot()
        plt.ylim(-4, 4)
        plt.title('validation data: labels vs. predictions')
        plt.legend(loc='best')
        plt.show()

        print('validation set r2 score:', r2_score(labels['sr_highres'], preds['sr_predicted']))
        print('validation set mean squared error: {0:.2f}%'.format(mean_squared_error(labels['sr_highres'], preds['sr_predicted']) * 100))

        lmean = labels['sr_highres'].mean()
        predmean = preds['sr_predicted'].mean()
        devmean = -100 / lmean * (lmean - predmean)
        print('sr mean: {} | predicted mean: {} | pred. deviation from sr: {}%'.format(lmean, predmean, devmean))

        preds2 = regressor.predict(features2)
        preds2 = pd.DataFrame(preds2)
        preds2.rename(columns={0: 'sr_predicted'}, inplace=True)

        fig = plt.figure(figsize=(18, 3))
        labels2['sr_highres'].plot()
        preds2['sr_predicted'].plot()
        plt.ylim(-4, 4)
        plt.title('test data: labels vs. predictions')
        plt.legend(loc='best')
        plt.show()

        print('test set r2 score:', r2_score(labels2['sr_highres'], preds2['sr_predicted']))
        print('test set mean squared error: {0:.2f}%'.format(mean_squared_error(labels2['sr_highres'], preds2['sr_predicted']) * 100))

        lmean2 = labels2['sr_highres'].mean()
        predmean2 = preds2['sr_predicted'].mean()
        devmean2 = -100 / lmean2 * (lmean2 - predmean2)
        print('sr mean: {} | predicted mean: {} | pred. deviation from sr: {}%'.format(lmean2, predmean2, devmean2))

        try:
            importances = regressor.feature_importances_
            vs.feature_plot(importances, X_train, y_train)
        except AttributeError:
            print('')
            print('No feature importance available for this learner')
            print('')
            print('')
        if type(e).__name__ == "XGBRegressor":
            fig, ax = plt.subplots(1,1,figsize=(8,13))
            plot_importance(regressor, ax=ax)
            plt.show()

    return regressor, preds, preds2

def spotCheck(features, labels, features2, labels2, reglist):
    sscv = ShuffleSplit(n_splits=2, test_size=.25, random_state=None)

    for e in reglist:

        print('-------------------------------------------------------------------------------------------------------')
        print('-------------------------------------------------------------------------------------------------------')
        print e
        i = 0
        for train_index, test_index in sscv.split(features):
            print('---------------------------------------------------------------------------------------------------')
            print('ShuffledSplit iteration {} of {}'.format(i + 1, 2))
            i += 1
            X_train, X_test = features.loc[train_index], features.loc[test_index]
            y_train, y_test = labels.loc[train_index], labels.loc[test_index]
            y_train = y_train.values.ravel()  # change column vector to 1d array to avoid conversion warning @ regressor.fit()

            regressor = e
            start = time.time()

            regressor.fit(X_train, y_train)
            elapsed = time.time() - start

            print("time to fit: %f" % (elapsed))
            score = regressor.score(X_test, y_test)

        print('---------------------------------------------------------------------')
        print('size of the training set (features, labels)', np.shape(X_train), np.shape(y_train))
        print('size of the validation set (features, labels)', np.shape(X_test), np.shape(y_test))
        print('size of the test set (features, labels)', np.shape(features2), np.shape(labels2))
        print('---------------------------------------------------------------------')

        preds = regressor.predict(features)
        preds = pd.DataFrame(preds)
        preds.rename(columns={0: 'sr_predicted'}, inplace=True)

        print('validation set r2 score:', r2_score(labels['sr_highres'], preds['sr_predicted']))
        print('validation set mean squared error: {0:.2f}%'.format(
            mean_squared_error(labels['sr_highres'], preds['sr_predicted']) * 100))

        preds2 = regressor.predict(features2)
        preds2 = pd.DataFrame(preds2)
        preds2.rename(columns={0: 'sr_predicted'}, inplace=True)

        print('test set r2 score:', r2_score(labels2['sr_highres'], preds2['sr_predicted']))
        print('test set mean squared error: {0:.2f}%'.format(
            mean_squared_error(labels2['sr_highres'], preds2['sr_predicted']) * 100))

    return

def testset_score(features3, labels3, regressor):

    preds3 = regressor.predict(features3)
    preds3 = pd.DataFrame(preds3)
    preds3.rename(columns={0: 'sr_predicted'}, inplace=True)

    fig = plt.figure(figsize=(18, 3))
    labels3['sr_highres'].plot()
    preds3['sr_predicted'].plot()
    plt.ylim(-4, 4)
    plt.title('test data: labels vs. predictions')
    plt.legend(loc='best')
    plt.show()

    print('test set r2 score:', r2_score(labels3['sr_highres'], preds3['sr_predicted']))
    print('test set mean squared error: {0:.2f}%'.format(mean_squared_error(labels3['sr_highres'], preds3['sr_predicted']) * 100))

    lmean3 = labels3['sr_highres'].mean()
    predmean3 = preds3['sr_predicted'].mean()
    devmean3 = -100 / lmean3 * (lmean3 - predmean3)
    print('sr mean: {} | predicted mean: {} | pred. deviation from sr: {}%'.format(lmean3, predmean3, devmean3))

def rebuild_index(data):
    # Rebuild the Index with continuous values, without gaps
    index = pd.DataFrame({'Index': range(0, len(data))})
    data = data.set_index(index['Index'])
    return data

def vis_pca(data, data2, n_comp):
    # TODO: Apply PCA by fitting the good data with the same number of dimensions as features
    pca = PCA(n_components=n_comp).fit(data)

    # Generate PCA results plot
    pca_results = vs_pca.pca_results(data, pca)

    expl_var = pca.explained_variance_ratio_
    cum_var_n = 0

    for i in range(0, n_comp):
        cum_var_n += expl_var[i]

    print "the cumulative explained variance of all commponents is {}".format(cum_var_n)
    print''
    print "The cumulated sums of the explained variances are: {}".format(np.cumsum(expl_var))
    print''

    # save transformed data
    pca_data = pca.fit_transform(data)
    pca_data2 = pca.fit_transform(data2)
    pca_data = pd.DataFrame(pca_data)
    pca_data2 = pd.DataFrame(pca_data2)
    #pca_data.columns = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6']
    print np.shape(pca_data)


    # create an x-axis variable for each pca component
    x = np.arange(1, n_comp+1)

    # plot the cumulative variance
    fig2 = plt.figure(figsize=(15, 7))
    plt.plot(x, np.cumsum(pca.explained_variance_ratio_), '-o', color='black')

    # plot the components' variance
    plt.bar(x, pca.explained_variance_ratio_, align='center', alpha=0.5)

    # plot styling
    plt.ylim(0, 1.05)
    plt.annotate('Cumulative explained variance', xy=(3.7, .88), arrowprops=dict(arrowstyle='->'), xytext=(4.5, .6))
    for i, j in zip(x, np.cumsum(pca.explained_variance_ratio_)):
        plt.annotate(str(j.round(4)), xy=(i + .2, j - .02))
    plt.xticks(range(1, n_comp+3))
    plt.xlabel('PCA components')
    plt.ylabel('Explained Variance')
    plt.show()

    return pca_data, pca_data2

def removeNaN(data, featurelist):
    # featurelist -> all features to be use to search NaNs
    a = np.shape(data)
    print 'number of data points BEFORE dropping all rows with any NaN elements for selected features:', a[0]

    # In den PartDiam features alle Zeilen mit NaN Werten loeschen
    # featurelist = data.columns.values
    # setup featurelist with PartDiam features

    for e in featurelist:
        data = data.dropna(axis=0, how='any', subset=[e])

    b = np.shape(data)
    print 'number of data points AFTER dropping all rows with any NaN elements for selected features:', b[0]
    print a[0] - b[0], 'number of data points deleted'
    print ('')
    return data

def extractDays(data, nan_len):
    # first value of data has to be NaN!

    # nan_len = define number of NaNs in between windows

    wds = {}  # day window dict
    i = 0  # running index of df
    i0 = 0  # window index
    a = 0  # count Nan

    for i in range(0, len(data), 1):
        if not 'day{:02d}'.format(i0) in wds.keys():  # create day-key for nex window
            wds['day{:02d}'.format(i0)] = {}
            wds['day{:02d}'.format(i0)]['start'] = np.nan
            wds['day{:02d}'.format(i0)]['end'] = np.nan

        if np.isnan(data['cleanlinessraw'].iloc[i]):
            a += 1
            if a >= nan_len:
                if not np.isnan(wds['day{:02d}'.format(i0)]['start']):
                    wds['day{:02d}'.format(i0)]['end'] = i - nan_len
                    i0 += 1
        else:
            if a >= nan_len:
                if np.isnan(wds['day{:02d}'.format(i0)]['start']):
                    wds['day{:02d}'.format(i0)]['start'] = i
            a = 0
    return wds

def outlierRem(data, feature):

    outliers = []
    #featurelist = data.columns.values
    featurelist = [feature]
    # print featurelist
    total_outl = 0

    # For each feature find the data points with extreme high or low values
    for e in featurelist:
        # TODO: Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.nanpercentile(data[e], 25)
        # TODO: Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.nanpercentile(data[e], 75)
        # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)

        outl_list = data[~((data[e] >= Q1 - step) & (data[e] <= Q3 + step))]

        print "number of Data points considered outliers for the feature '{}': {}".format(e, len(outl_list))

        outl_indices = outl_list.index.values  # get the indices (clients for each outlier)

        for e in outl_indices:  # go through indices
            outliers.append(e)

    count_outl = {}  # create dict with all outliers and their occurrence number
    for i in outliers:
        count_outl[i] = count_outl.get(i, 0) + 1

    mult_outl = {}  # dict with outliers that appear more than once
    for e in count_outl:
        if count_outl[e] > 1:
            mult_outl[e] = count_outl[e]

    print "number of Outliers that occur various times and thus are NOT removed '{}':".format(len(mult_outl))

    outliers_set = list(set(outliers))  # create list where each outlier appears only once
    # print "Outliers_set: {}".format(outliers_set)

    for e in mult_outl:  # remove all outliers that occur more than once
        outliers_set.remove(e)

    #print "number of Outliers that occur only once and thus are removed '{}':".format(len(outliers_set))
    print('')

    # Remove the outliers, if any were specified
    good_data = data.drop(data.index[outliers_set]).reset_index(drop=True)

    return good_data

def outlierDet(data, feature):

    featurelist = [feature]
    nan = np.nan

    # For each feature find the data points with extreme high or low values
    for e in featurelist:
        # TODO: Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = np.nanpercentile(data[e], 25)
        # TODO: Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = np.nanpercentile(data[e], 75)
        # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)

        outl_list = data[~((data[e] >= Q1 - step) & (data[e] <= Q3 + step))]

        print "number of Data points considered outliers for the feature '{}': {}".format(e, len(outl_list))

        outl_indices = outl_list.index.values  # get the indices (clients for each outlier)
        start = time.time()
        for a in outl_indices:  # go through indices
            data[e][a] = np.nan
        elapsed = time.time() - start
        print("time to write 'Nan' for feature {}: %f".format(e) % (elapsed))


    return data

def createSoilingRate(data):

    data['sr_highres'] = data['cleanlinesscorr'].diff() / data['time'].diff()

    fig = plt.figure(figsize=(15, 3))
    data['cleanlinesscorr'].plot()
    plt.title('corrected cleanliness')
    plt.show()
    fig = plt.figure(figsize=(15, 3))
    data['sr_highres'].plot()
    plt.ylim(-0.05, 0.01)
    plt.title('calculated soiling rate(label)')
    plt.show()

    print('mean of soiling rate:', data.sr_highres.mean())

    return data




