from backtest import BackTest, Strategy
import backtrader.indicators as btind
from common_utils import *

class Strat1(Strategy):
    def __init__(self):
        # Indicators for the plotting show
        # btind.ExponentialMovingAverage(self.datas[0], period=25)
        # btind.WeightedMovingAverage(self.datas[0], period=25,
        #                                     subplot=True)
        # btind.StochasticSlow(self.datas[0])
        # btind.MACDHisto(self.datas[0])
        # rsi = btind.RSI(self.datas[0])
        # btind.SmoothedMovingAverage(rsi, period=10)
        # btind.ATR(self.datas[0], plot=False)
        super().__init__()

    def next(self):
        # self.log('Close, %.2f' % self.dataclose[0])
        if self.dataclose[0] < self.dataclose[-1]:
            # current close less than previous close
            if self.dataclose[-1] < self.dataclose[-2]:
                # previous close less than the previous close

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                # The order is executed “At Market”. The broker (shown in previous examples) executes this using the opening price of the next bar, because that’s the 1st tick after the current under examination bar.
                # The order is executed so far without any commission
                self.buy()

if __name__ == "__main__":

    # data = get_all_data()
    # data_train = {}
    # data_test = {}
    # for symbol in data:
    #     train, test = train_test_split(data[symbol])
    #     data_train[symbol] = train
    #     data_test[symbol] = test

    # score, pvalue, pairs = find_cointegrated_pairs(data_train)
    # for pair in pairs:
    #     print(pair)

    bt = BackTest()
    bt.set_cash(100000)
    bt.add_symbol("AAATECH")
    bt.add_strategy(Strat1)
    bt.add_sizer(stake=20)
    print(f"Initial Portfolio Value: {bt.get_value()}")
    bt.run()
    print(f"Final Portfolio Value: {bt.get_value()}")
    bt.plot()