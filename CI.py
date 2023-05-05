from pingouin import mediation_analysis, read_dataset


def mediation(df, X, Y, mediators):
    #df = read_dataset('mediation')
    print(len(df))
    #df = df.sample(frac = 0.8)
    print(len(df))
    analysis = mediation_analysis(data=df, x=X, m=mediators, y=Y, alpha=0.05,
                       seed=42)

    print(analysis)



if __name__ == '__main__':
    mediation("","","","")