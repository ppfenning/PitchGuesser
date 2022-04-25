from dataclasses import dataclass
import pandas as pd
import os
import tempfile
from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
import pybaseball as bball
from sklearn.model_selection import train_test_split

bball.cache.enable()


@dataclass
class PitchGuesser:
    """Class for modeling pitch types"""
    start_dt: str = '2022-03-17'  # start day for
    end_dt: str = dt.today().strftime("%Y-%m-%d")
    refresh: bool = False

    def __post_init__(self):
        pdir = Path(__file__).parent.absolute()
        datadir = os.path.join(pdir, 'data')
        tmpdir = tempfile.gettempdir()
        self.__pkl = f"{tmpdir}/pitch_data.pkl"
        self.__cols = pd.read_csv(f"{datadir}/cols.csv", header=None).squeeze().to_list()
        self.__start_ts = dt.strptime(self.start_dt, "%Y-%m-%d")
        self.__end_ts = dt.strptime(self.end_dt, "%Y-%m-%d")
        self.raw_data = self.__get_data()
        self.__splitter()

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
        df = self.__get_cols(df)
        return df.loc[(df['game_date'] >= self.__start_ts) & (df['game_date'] <= self.__end_ts)]

    def __splitter(self):

        skip_cols = (
            'game_date',
            'pitcher',
            'player_name',
            'pitch_type',
            'pitch_name',
        )

        X = self.raw_data.loc[:, ~self.raw_data.columns.isin(skip_cols)]
        y = self.raw_data.pitch_type

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42
        )

    def __get_data(self):
        if os.path.exists(self.__pkl) and not self.refresh:
            df = pd.read_pickle(self.__pkl)
        else:
            df = bball.statcast(self.start_dt, self.end_dt)
        df = self.__get_from_date(df)
        df = self.__get_to_date(df)
        return self.__clean_data(df)

    def __get_cols(self, df):
        return df[self.__cols]


if __name__ == '__main__':
    test = PitchGuesser()
