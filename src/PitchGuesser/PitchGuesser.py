from dataclasses import dataclass
from datetime import datetime as dt
from datetime import timedelta as td
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
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
random_state = 42
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
            random_state=random_state
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

    model_name = ''
    
    def __post_init__(self):
        super(PitchGuessPost, self).__post_init__()
        self.y_predict = self.__prediction()
        self.cm = self.__get_cm()
        self.score = self.__score()
        self.feature_types = self.__feature_types()

    def __feature_types(self):
        categorical = list(self.X_test.select_dtypes(include=['bool']).columns) + ['zone']
        numeric = list(self.X_test.loc[:, ~self.X_test.columns.isin(categorical)].columns)
        return {
            'categorical': categorical,
            'numeric': numeric,
        }

    def show_correlation(self):
        corr_df = self.X_train[self.feature_types['numeric']]  # New dataframe to calculate correlation between numeric features
        cor = corr_df.corr(method='pearson')
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.title("Correlation Plot")
        sns.heatmap(
            cor,
            mask=np.zeros_like(cor, dtype=np.bool_),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True,
            ax=ax
        )
        plt.show()
        return cor

    def show_pair_plot(self):
        sns.set()
        fcols = self.feature_types['numeric'] + ['pitch_name']
        df = self.raw_data[fcols].sample(300, replace=False).reset_index(drop=True)
        plt.title("Pair Plot")
        sns.pairplot(df, hue="pitch_name")
        plt.show()

    def __prediction(self):
        return self.model.predict(self.X_test)

    def __get_cm(self):
        return pd.DataFrame(
            confusion_matrix(self.y_test, self.y_predict),
            index=self.pitches,
            columns=self.pitches
        )

    def __score(self):
        return accuracy_score(self.y_test, self.y_predict)

    def class_report(self):
        print(classification_report(self.y_test, self.y_predict))

def _get_grid_search(model, params):
    return GridSearchCV(
        model,
        params,
        cv=2,
        n_jobs=-1,
        verbose=1,
        scoring='f1_micro'
    )

@dataclass
class PitchRFC(PitchGuessPost):
    __params = {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'entropy']
    }
    model_name = 'Random Forest'
    model = _get_grid_search(RandomForestClassifier(random_state=random_state), __params)
    model_pkl = f'{tmpdir}/RFC.pkl'

@dataclass
class PitchGBC(PitchGuessPost):
    # this took too long for gridsearch
    model_name = 'Gradient Boosting'
    model = GradientBoostingClassifier(random_state=random_state)
    model_pkl = f'{tmpdir}/GBC.pkl'

@dataclass
class PitchKNN(PitchGuessPost):
    __params = {
        'n_neighbors': [5, 10, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean']
    }
    model_name = 'K-Nearest Neighbor'
    model = _get_grid_search(KNeighborsClassifier(), __params)
    model_pkl = f'{tmpdir}/KNN.pkl'

if __name__ == '__main__':
    rfc = PitchRFC(start_dt='2022-03-17')
    gbc = PitchGBC(start_dt='2022-03-17')
    knn = PitchKNN(start_dt='2022-03-17')
