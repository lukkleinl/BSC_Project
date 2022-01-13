import seaborn as sns


def data_quick_check(df):
    print(df.info())
    print(df.describe())
    print(df.head())


def plot_data(df):
    plot = sns.pairplot(df)
    plot.savefig("output.png")
