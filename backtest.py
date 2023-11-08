import backtrader as bt
import pandas as pd
import datetime
import os.path
import sys
import json
from colorama import Fore, Back, Style
from common_utils import *

class Strategy(bt.Strategy):
    # self.datas[0] is the default data for trading operations and to keep all strategy elements synchronized
    # self.dataclose = self.datas[0].close keeps a reference to the close line.
    # The strategy next method will be called on each bar of the system clock (self.datas[0])
    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return
        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

class BackTest:
    def __init__(self, data_dir="/home/sarthak/btp/data", commission=0.001) -> None:
        self.cerebro = bt.Cerebro()
        self.data_dir = data_dir
        self.cerebro.broker.setcommission(commission=commission)
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    def set_cash(self, cash):
        self.cerebro.broker.setcash(cash)
    def add_symbols(self, symbols, common=True, test_train_split=False, train=False):
        dfs = []
        for symbol in symbols:
            path = os.path.join(self.data_dir, symbol + ".json")
            with open(path) as f:
                data = json.load(f)
            df = pd.DataFrame({
                'datetime': data['t'],
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data['v']
            })
            df.set_index('datetime', inplace=True)
            df.index = pd.to_datetime(df.index, unit='s')
            dfs.append(df)
        if common:
            dfs = get_common_data(dfs)
        for df in dfs:
            if(test_train_split):
                df_train, df_test = train_test_split(df)
                if(train):
                    data = bt.feeds.PandasData(dataname=df_train)
                else:
                    data = bt.feeds.PandasData(dataname=df_test)
            else:
                data = bt.feeds.PandasData(dataname=df)
            self.cerebro.adddata(data)
        
    def add_symbol(self, symbol):
        path = os.path.join(self.data_dir, symbol + ".json")
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame({
            'datetime': data['t'],
            'open': data['o'],
            'high': data['h'],
            'low': data['l'],
            'close': data['c'],
            'volume': data['v']
        })
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index, unit='s')
        data = bt.feeds.PandasData(dataname=df)
        self.cerebro.adddata(data)

    def add_strategy(self, strategy, **kwargs):
        self.cerebro.addstrategy(strategy, **kwargs)

    def add_sizer(self, **kwargs):
        self.cerebro.addsizer(bt.sizers.SizerFix, **kwargs)

    def run(self):
        results = self.cerebro.run()
        for result in results:
            print(f"Sharpe Ratio: {result.analyzers.sharpe.get_analysis()['sharperatio']}")
            print(f"Drawdown: {result.analyzers.drawdown.get_analysis()['drawdown']}")
            print(f"Returns: {result.analyzers.returns.get_analysis()['rnorm100']}")
        return results
    
    def get_value(self):
        return self.cerebro.broker.getvalue()
    
    def plot(self, **kwargs):
        self.cerebro.plot(**kwargs)

    def optstrategy(self, strategy, **kwargs):
        self.cerebro.optstrategy(strategy, **kwargs)
    