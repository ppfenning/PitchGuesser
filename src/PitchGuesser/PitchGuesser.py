from dataclasses import dataclass
from datetime import datetime as dt
from datetime import timedelta as td
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import scipy.optimize as opt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import os
import pandas as pd
import pybaseball as bball
import tempfile

bball.cache.enable()


@dataclass
class PitchData:
    """Class for modeling pitch types"""
    start_dt: str = '2022-03-17'  # start day for
    end_dt: str = dt.today().strftime("%Y-%m-%d")
    refresh: bool = False

    def __post_init__(self):
        pdir = Path(__file__).parent.absolute()
        self.datadir = os.path.join(pdir, 'data')
        tmpdir = tempfile.gettempdir()
        self.__pkl = f"{tmpdir}/pitch_data.pkl"
        self.__start_ts = dt.strptime(self.start_dt, "%Y-%m-%d")
        self.__end_ts = dt.strptime(self.end_dt, "%Y-%m-%d")
        self.raw_data = self.__get_data()

    def __get_from_date(self, df):
        if self.__start_ts < df.game_date.min():
            df = pd.concat([df, bball.statcast(
                start_dt=self.start_dt,
                end_dt=dt.strftime(df.game_date.min(), "%Y-%m-%d")
            )])
        return df

    def __get_to_date(self, df):
        if self.__end_ts > df.game_date.max() + td(days=1):
            df = pd.concat([df, bball.statcast(
                start_dt=dt.strftime(df.game_date.max(), "%Y-%m-%d"),
                end_dt=self.end_dt
            )])
        return df

    def __clean_data(self, df):
        df = df.dropna(subset=['release_speed', 'release_pos_x', 'release_pos_z'])
        df.to_pickle(self.__pkl)
        df = self.__preprocess(df)
        df = self.__get_cols(df)
        df = df.fillna(method='ffill')
        return df.loc[(df['game_date'] >= self.__start_ts) & (df['game_date'] <= self.__end_ts)]

    def __scaler(self, df):
        scaled = pd.read_csv(f"{self.datadir}/scaled.csv", header=None).squeeze().to_list()
        for scale in scaled:
            data = df[scale]
            df[scale] = (data - np.min(data))/(np.max(data)-np.min(data))
        return df

    def __preprocess(self, df):
        df2 = df.copy()
        df2['lefty'] = (df.p_throws == 'L').astype(int)
        df2['righty'] = (df.p_throws == 'R').astype(int)
        df2['ball'] = (df.type == 'B').astype(int)
        df2['strike'] = (df.type == 'S').astype(int)
        df2['hit_in_play'] = (df.type == 'X').astype(int)
        le = LabelEncoder()
        df2['pitch_type'] = le.fit_transform(df['pitch_type'])
        df2 = self.__scaler(df2)
        return df2

    def __get_data(self):
        if os.path.exists(self.__pkl) and not self.refresh:
            df = pd.read_pickle(self.__pkl)
        else:
            df = bball.statcast(self.start_dt, self.end_dt)
        df = self.__get_from_date(df)
        df = self.__get_to_date(df)
        return self.__clean_data(df)

    def __get_cols(self, df):
        cols = pd.read_csv(f"{self.datadir}/cols.csv", header=None).squeeze().to_list()
        return df[cols]

@dataclass
class PitchModelBase(PitchData):
    test_size = 0.30

    def __post_init__(self):
        super(PitchModelBase, self).__post_init__()
        self.__splitter()

    def __set_X_y(self):

        skip_cols = (
            'game_date',
            'pitcher',
            'player_name',
            'pitch_type',
            'pitch_name',
            'p_throws',
            'type',
        )

        X = self.raw_data.loc[:, ~self.raw_data.columns.isin(skip_cols)].copy()
        y = self.raw_data.pitch_type

        return X, y

    def __splitter(self):

        X, y = self.__set_X_y()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=42
        )


@dataclass
class PitchLR(PitchModelBase):
    
    def __post_init__(self):
        super(PitchLR, self).__post_init__()
        self.X_test = self.__get_interc(self.X_test)
        self.X_train = self.__get_interc(self.X_train)
        self.classifiers = self.__getClassifier()

    @staticmethod
    def __get_interc(df):
        df2 = df.copy()
        df2.loc[:, 'inter'] = 1.
        return df2.loc[:, ['inter'] + list(df.columns)]

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z.astype(np.float64)))

    def __cost(self, theta, X, y):
        predictions = self.__sigmoid(X @ theta)
        predictions[predictions == 1] = 0.999  # log(1)=0 causes error in division
        error = -y * np.log(predictions) - (1 - y) * np.log(1 - predictions)
        return sum(error) / len(y)

    def __cost_gradient(self, theta, X, y):
        predictions = self.__sigmoid(X @ theta)
        return X.transpose() @ (predictions - y) / len(y)

    def __getClassifier(self):
        num_labels = len(self.y_train.unique())
        num_features = self.X_train.shape[1]
        classifiers = np.zeros(shape=(num_labels, num_features))
        for c in range(num_labels):
            label = (self.y_train == c).astype(int)
            init_theta = np.zeros(num_features)
            classifiers[c, :] = opt.fmin_cg(
                self.__cost,
                init_theta,
                self.__cost_gradient,
                (self.X_train, label),
            )
        return classifiers

@dataclass
class PitchDTC(PitchModelBase):

    def __post_init__(self):
        super(PitchDTC, self).__post_init__()
        self.model = self.__model()
        self.prediction = self.__prediction()
        self.score = self.__score()

    def __model(self):
        model = DecisionTreeClassifier(max_depth=2)
        model.fit(self.X_train, self.y_train)
        return model

    def __prediction(self):
        return self.model.predict(self.X_test)

    def __score(self):
        return accuracy_score(self.y_test, self.prediction)

if __name__ == '__main__':
    test = PitchDTC(start_dt='2021-03-17')
