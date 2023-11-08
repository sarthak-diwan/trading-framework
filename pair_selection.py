from common_utils import *

data = get_all_data(data_dir="/home/sarthak/trading/build/data_min")

# Merge all dataframe in data
df = pd.DataFrame()
for symbol in data.keys():

    # Satisfies ESG Criteria
    if check_esg(symbol):
        # Use only train data to find pairs
        data_train, data_test = train_test_split(data[symbol], train_size=0.8)
        print(len(data_train), len(data_test))
        df = pd.concat([df, data_train['close']], axis=1)

print(df.head())
