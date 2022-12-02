from parsers.main import get_all_paths

if __name__ == '__main__':
    df = get_all_paths()
    print(df.head())