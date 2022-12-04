from parsers.main import get_data

if __name__ == '__main__':
    df = get_data(minimal=True)
    print(df.head())
    print(df.columns)