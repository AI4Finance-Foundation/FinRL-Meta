import numpy as np
from pandas import DataFrame

class WtKlineData:
    def __init__(self, size:int, bAlloc:bool = True):
        self.capacity:int = size
        self.size:int = 0

        if bAlloc:
            self.bartimes = np.zeros(self.capacity, np.int64)
            self.opens = np.zeros(self.capacity)
            self.highs = np.zeros(self.capacity)
            self.lows = np.zeros(self.capacity)
            self.closes = np.zeros(self.capacity)
            self.volumes = np.zeros(self.capacity)
        else:
            self.bartimes = None
            self.opens = None
            self.highs = None
            self.lows = None
            self.closes = None
            self.volumes = None

    def append_bar(self, newBar:dict):

        pos = self.size
        if pos == self.capacity:
            self.bartimes[:-1] = self.bartimes[1:]
            self.opens[:-1] = self.opens[1:]
            self.highs[:-1] = self.highs[1:]
            self.lows[:-1] = self.lows[1:]
            self.closes[:-1] = self.closes[1:]
            self.volumes[:-1] = self.volumes[1:]

            pos = -1
        else:
            self.size += 1
        self.bartimes[pos] = newBar["bartime"]
        self.opens[pos] = newBar["open"]
        self.highs[pos] = newBar["high"]
        self.lows[pos] = newBar["low"]
        self.closes[pos] = newBar["close"]
        self.volumes[pos] = newBar["volume"]

    def is_empty(self) -> bool:
        return self.size==0

    def clear(self):
        self.size = 0

        self.bartimes:np.ndarray = np.zeros(self.capacity, np.int64)
        self.opens:np.ndarray = np.zeros(self.capacity)
        self.highs:np.ndarray = np.zeros(self.capacity)
        self.lows:np.ndarray = np.zeros(self.capacity)
        self.closes:np.ndarray = np.zeros(self.capacity)
        self.volumes:np.ndarray = np.zeros(self.capacity)

    def get_bar(self, iLoc:int = -1) -> dict:
        if self.is_empty():
            return None

        lastBar = dict()
        lastBar["bartime"] = self.bartimes[iLoc]
        lastBar["open"] = self.opens[iLoc]
        lastBar["high"] = self.highs[iLoc]
        lastBar["low"] = self.lows[iLoc]
        lastBar["close"] = self.closes[iLoc]
        lastBar["volume"] = self.volumes[iLoc]

        return lastBar

    def slice(self, iStart:int = 0, iEnd:int = -1, bCopy:bool = False):
        if self.is_empty():
            return None

        bartimes = self.bartimes[iStart:iEnd]
        cnt = len(bartimes)
        ret = WtKlineData(cnt, False)
        ret.size = cnt

        if bCopy:
            ret.bartimes = bartimes.copy()
            ret.opens = self.opens[iStart:iEnd].copy()
            ret.highs = self.highs[iStart:iEnd].copy()
            ret.lows = self.lows[iStart:iEnd].copy()
            ret.closes = self.closes[iStart:iEnd].copy()
            ret.volumes = self.volumes[iStart:iEnd].copy()
        else:
            ret.bartimes = bartimes
            ret.opens = self.opens[iStart:iEnd]
            ret.highs = self.highs[iStart:iEnd]
            ret.lows = self.lows[iStart:iEnd]
            ret.closes = self.closes[iStart:iEnd]
            ret.volumes = self.volumes[iStart:iEnd]

        return ret

    def to_df(self) -> DataFrame:
        ret = DataFrame({
            "bartime":self.bartimes,
            "open":self.opens,
            "high":self.highs,
            "low":self.lows,
            "close":self.closes,
            "volume":self.volumes
        })
        ret.set_index(self.bartimes)
        return ret

class WtHftData:
    def __init__(self, capacity:int):
        self.capacity:int = capacity
        self.size:int = 0

        self.items = [None]*capacity

    def append_item(self, newItem:dict):
        pos = self.size
        if pos == self.capacity:
            self.items[:-1] = self.items[1:]
            pos = -1
        else:
            self.size += 1

        self.items[pos] = newItem

    def is_empty(self) -> bool:
        return self.size==0

    def clear(self):
        self.size = 0
        self.items = []*self.capacity

    def get_item(self, iLoc:int=-1) -> dict:
        if self.is_empty():
            return None

        return self.items[iLoc]

    def to_df(self) -> DataFrame:
        ret = DataFrame(self.items)
        return ret