from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class vectorizer:

    def __init__(self, typeVectorizer, model, features, label):
        self.typeVectorizer = typeVectorizer
        self.features = features
        self.label = label
        self.model = model

    def trainingModel(self):

        cv = self.typeVectorizer(stop_words='english', ngram_range=(1, 2))
        # The fit(data) method is used to compute the mean and std dev for a given feature to be used further for scaling.
        # The transform(data) method is used to perform scaling using mean and std dev calculated using the.fit() method.
        # The fit_transform() method does both fits and transform.
        self.features = cv.fit_transform(self.features)
        # Split the feature and label into random train and test subsets.
        # There are 4 subsets: X_train, X_test, y_train, y_test
        # Test size represents the absolute number of test samples.
        # The train size value is set to the complement of the test size.
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.label,
                                                            test_size=0.25, random_state=24)
        # Definition of the model used
        model = self.model
        # Training the model
        model.fit(X_train, y_train)
        p_train = model.predict(X_train)
        p_test = model.predict(X_test)
        # Accuracy score
        acc_train = accuracy_score(y_train, p_train)
        acc_test = accuracy_score(y_test, p_test)
        print(f'Train acc. {acc_train}, Test acc. {acc_test}')
        # Confusion matrix
        #                | Positive Prediction | Negative Prediction
        # Positive Class | True Positive (TP)  | False Negative (FN)
        # Negative Class | False Positive (FP) | True Negative (TN)
        cm_lr = confusion_matrix(y_test, p_test)
        tn, fp, fn, tp = confusion_matrix(y_test, p_test).ravel()
        print(f'TRUE POSITIVE: {tp}')
        print(f'TRUE NEGATIVE: {tn}')
        print(f'FALSE POSITIVE: {fp}')
        print(f'FALSE NEGATIVE: {fn}')
        # True positive and true negative rates
        tpr_lr = round(tp / (tp + fn), 4)
        tnr_lr = round(tn / (tn + fp), 4)
        print(f'TRUE POSITIVE RATES: {tpr_lr}')
        print(f'TRUE NEGATIVE RATES: {tnr_lr}')
        # F1Score
        # Changing the pos_label value default in the f1_score() to consider positive scenarios when 'positive' appears
        f1score_train = f1_score(y_train, p_train, pos_label='positive')
        f1score_test = f1_score(y_test, p_test, pos_label='positive')
        print(f'TRAIN F1 SCORE: {f1score_train}, TEST F1 SCORE {f1score_test}')
        # Precision Score
        # Changing the pos_label value default in the precision_score() to consider positive scenarios when 'positive' appears
        precisionscore_train = precision_score(y_train, p_train, pos_label='positive')
        precisionscore_test = precision_score(y_test, p_test, pos_label='positive')
        print(f'TRAIN PRECISION SCORE: {precisionscore_train}, TEST PRECISION SCORE:  {precisionscore_test}')
        # Recall score
        # Changing the pos_label value default in the recall_score() to consider positive scenarios when 'positive' appears
        recallscore_train = recall_score(y_train, p_train, pos_label='positive')
        recallscore_test = recall_score(y_test, p_test, pos_label='positive')
        print(f'TRAIN RECALL SCORE:  {recallscore_train}, TEST RECALL SCORE:  {recallscore_test}')

