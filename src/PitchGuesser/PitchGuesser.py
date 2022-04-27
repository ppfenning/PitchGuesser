from dataclasses import dataclass
from datetime import datetime as dt
from datetime import timedelta as td
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pybaseball as bball
import seaborn as sns
import tempfile

bball.cache.enable()
tmpdir = f'{tempfile.gettempdir()}/PitchGuesser'
if not os.path.exists(tmpdir):
    os.mkdir(tmpdir)

@dataclass
class PitchData:
    """Class for modeling pitch types"""
    start_dt: str = '2022-03-17'  # start day for
    end_dt: str = dt.today().strftime("%Y-%m-%d")
    refresh: bool = False

    def __post_init__(self):
        pdir = Path(__file__).parent.absolute()
        self.datadir = os.path.join(pdir, 'data')
        self.__pkl = f"{tmpdir}/pitch_data.pkl"
        self.pitches = ['4-Seam Fastball', 'Changeup', 'Curveball', 'Cutter', 'Sinker', 'Slider']
        self.__start_ts = dt.strptime(self.start_dt, "%Y-%m-%d")
        self.__end_ts = dt.strptime(self.end_dt, "%Y-%m-%d")
        self.raw_data = self.__get_data()
        self.features = {
            'numeric': 1,
            'categorical': 1
        }

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
        # most common pitches
        df = df.loc[df.pitch_name.isin(self.pitches), :].sort_values(by=['pitch_name', 'game_date', 'pitcher'])
        return df.loc[(df['game_date'] >= self.__start_ts) & (df['game_date'] <= self.__end_ts), :]

    def __scaler(self, df):
        scaled = pd.read_csv(f"{self.datadir}/scaled.csv", header=None).squeeze().to_list()
        for scale in scaled:
            data = df[scale]
            df[scale] = (data - np.min(data))/(np.max(data)-np.min(data))
        return df

    def __preprocess(self, df):
        df2 = df.copy()
        df2['lefty'] = df.p_throws == 'L'
        df2['righty'] = df.p_throws == 'R'
        df2['ball'] = df.type == 'B'
        df2['strike'] = df.type == 'S'
        df2['hit_in_play'] = df.type == 'X'
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
class PitchModelBuild(PitchData):
    model = None
    model_pkl = None
    test_size: float = 0.30
    random_state = 42

    def __post_init__(self):
        if self.model is None:
            raise ValueError('A model must be passed')
        if self.model_pkl is None:
            raise ValueError('A model pickle file must be passed')
        super(PitchModelBuild, self).__post_init__()
        self.__splitter()
        self.__fit()

    def __set_X_y(self):

        skip_cols = (
            'game_date',
            'pitcher',
            'player_name',
            'pitch_name',
            'p_throws',
            'type',
        )

        X = self.raw_data.loc[:, ~self.raw_data.columns.isin(skip_cols)].copy()
        le = LabelEncoder()
        y = self.raw_data.pitch_name
        y = le.fit_transform(y)

        return X, y

    def __splitter(self):

        X, y = self.__set_X_y()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    def __fit(self):
        if not os.path.exists(self.model_pkl) or self.refresh:
            self.model.fit(self.X_train, self.y_train)
            with open(self.model_pkl, 'wb') as fout:
                pickle.dump(self.model, fout)
        else:
            with open(self.model_pkl, 'rb') as fin:
                self.model = pickle.load(fin)

@dataclass
class PitchGuessPost(PitchModelBuild):
    
    def __post_init__(self):
        super(PitchGuessPost, self).__post_init__()
        self.y_predict = self.__prediction()
        self.cm = self.__get_cm()

    @property
    def __feature_types(self):
        categorical = list(self.X_test.select_dtypes(include=['bool']).columns) + ['zone']
        numeric = list(self.X_test[~self.X_test.columns.isin(categorical)].columns)
        return {
            'categorical': categorical,
            'numeric': numeric,
        }

    def __prediction(self):
        return self.model.predict(self.X_test)

    def __get_cm(self):
        return pd.DataFrame(
            confusion_matrix(self.y_test, self.y_predict),
            index=self.pitches,
            columns=self.pitches
        )

    def score(self):
        return accuracy_score(self.y_test, self.y_predict)

    def class_report(self):
        print(classification_report(self.y_test, self.y_predict))

@dataclass
class PitchRFC(PitchGuessPost):
    #model = GridSearchCV(
    #    RandomForestClassifier(n_estimators=100),
    #    params,
    #    cv=3,
    #    n_jobs=-1,
    #    verbose=1,
    #    scoring='f1_micro'
    #)
    model = RandomForestClassifier(n_estimators=100)
    model_pkl = f'{tmpdir}/RFC.pkl'

if __name__ == '__main__':
    test = PitchRFC(start_dt='2022-03-17')
