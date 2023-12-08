import pandas as pd
from scipy.io import arff


def load_organic():
    cm1 = arff.loadarff('./src/data/organic/cm1.arff')
    jm1 = arff.loadarff('./src/data/organic/jm1.arff')
    kc1 = arff.loadarff('./src/data/organic/kc1.arff')
    kc2 = arff.loadarff('./src/data/organic/kc2.arff')
    pc1 = arff.loadarff('./src/data/organic/pc1.arff')

    df = process_organic(cm1, jm1, kc1, kc2, pc1)

    return df

def process_organic(cm1, jm1, kc1, kc2, pc1):
    df_cm1 = pd.DataFrame(cm1[0])
    df_jm1 = pd.DataFrame(jm1[0])
    df_kc1 = pd.DataFrame(kc1[0])
    df_kc2 = pd.DataFrame(kc2[0])
    df_pc1 = pd.DataFrame(pc1[0])

    df_kc2 = df_kc2.rename(columns={
      'lOCodeAndComment': 'locCodeAndComment',
      'problems': 'defects',
    })
    df_pc1 = df_pc1.rename(columns={
      'iv(G)': 'iv(g)',
      'N': 'n',
      'V': 'v',
      'L': 'l',
      'D': 'd',
      'I': 'i',
      'E': 'e',
      'B': 'b',
      'T': 't',
    })

    df = pd.concat([df_cm1, df_jm1, df_kc1, df_kc2, df_pc1])
    df = df.replace({
        b'true': True,
        b'false': False,
        b'yes': True,
        b'no': False,
    })
    df = df.drop_duplicates()
    df = df.dropna()

    return df


def load_synthetic():
    df = pd.read_csv('./src/data/synthetic/train.csv')
    df = process_synthetic(df)

    return df


def process_synthetic(df):
    df = df.drop_duplicates()
    df = df.dropna()

    return df
