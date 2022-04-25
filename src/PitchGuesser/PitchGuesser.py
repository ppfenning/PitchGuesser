from dataclasses import dataclass
import pandas as pd
import os
import tempfile
from pathlib import Path
from datetime import datetime as dt
from datetime import timedelta as td
import pybaseball as bball

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
        self.start_ts = dt.strptime(self.start_dt, "%Y-%m-%d")
        self.end_ts = dt.strptime(self.end_dt, "%Y-%m-%d")
        self.raw_data = self.__refresh_data()

    def __get_from_date(self, df):
        if self.start_ts < df.game_date.min():
            df = pd.concat([df, bball.statcast(
                start_dt=self.start_dt,
                end_dt=dt.strftime(df.game_date.min(), "%Y-%m-%d")
            )])
        return df

    def __get_to_date(self, df):
        if self.end_ts > df.game_date.max() + td(days=1):
            df = pd.concat([df, bball.statcast(
                start_dt=dt.strftime(df.game_date.max(), "%Y-%m-%d"),
                end_dt=self.end_dt
            )])
        return df

    def __get_end_dates(self, df):
        df = self.__get_from_date(df)
        df = self.__get_to_date(df)
        df.to_pickle(self.__pkl)
        return df.loc[(df['game_date'] >= self.start_ts) & (df['game_date'] <= self.end_ts)]

    def __get_cols(self, df):
        return df[self.__cols]

    def __refresh_data(self):
        if os.path.exists(self.__pkl) and not self.refresh:
            df = pd.read_pickle(self.__pkl)
        else:
            df = bball.statcast(self.start_dt, self.end_dt)
        df = self.__get_end_dates(df)
        df = self.__get_cols(df)
        return df


if __name__ == '__main__':
    test = PitchGuesser()
