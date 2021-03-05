from pgmpy.models import BayesianModel as BM
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd


def read_input():
    entrada = pd.read_csv('high_diamond_ranked_10min.csv')
    entrada['blueTotalVisao'] = entrada['blueWardsPlaced'] + entrada[
        'blueWardsDestroyed']  #- entrada['redWardsPlaced']
    entrada['blueAbates'] = entrada['blueKills'] + entrada['blueAssists']
    entrada['blueAvgLevelRounded'] = entrada['blueAvgLevel'].astype(int)
    entrada.drop([
        'gameId', 'redKills', 'redDeaths', 'redGoldDiff', 'redExperienceDiff',
        'redCSPerMin', 'blueAssists', 'redAssists', 'blueWardsPlaced',
        'redWardsPlaced', 'blueWardsDestroyed', 'redWardsDestroyed',
        'redFirstBlood', 'blueAvgLevel', 'redAvgLevel', 'blueHeralds',
        'redHeralds', 'redTotalJungleMinionsKilled', 'blueTowersDestroyed',
        'redTowersDestroyed', 'blueKills', 'blueDeaths', 'blueEliteMonsters',
        'blueTotalGold', 'blueGoldPerMin', 'redEliteMonsters', 'redDragons',
        'redTotalGold', 'redTotalExperience', 'redTotalMinionsKilled',
        'redGoldPerMin', 'blueFirstBlood'
    ],
                 axis=1,
                 inplace=True)
    return entrada


def Model_def(dataFrame):
    #min,max,mean blueTotalExperience = 10098, 22224, 17928.1
    #min,max,mean blueTotalMinionsKilled = 90, 283, 215.
    #min,max,mean blueGoldDiff = -10830, 11467, 14.4
    #min,max,mean blueCSPerMin = 9, 28.3, 21.6
    #min,max,mean blueTotalVisao = 5, 254, 25.11
    #min,max,mean blueAbates = 0, 51, 21.3
    #min,max,mean blueAvgLevelRounded = 4, 8, 6.5
    #min,max,mean blueExperienceDiff = -9333, 8348, -33
    dataFrame['blueTotalExperience'] = pd.cut(
        x=dataFrame['blueTotalExperience'],
        bins=[0, 12000, 15000, 19000, 23000],
        labels=['mBaixo', 'abMedia', 'Media', 'acMedia'])
    dataFrame['blueExperienceDiff'] = pd.cut(
        x=dataFrame['blueExperienceDiff'],
        bins=[-10000, 0, 10000],
        labels=['difNegativa', 'difPositiva'])
    dataFrame['blueTotalMinionsKilled'] = pd.cut(
        x=dataFrame['blueTotalMinionsKilled'],
        bins=[0, 215, 300],
        labels=['abMedia', 'acMedia'])
    dataFrame['blueGoldDiff'] = pd.cut(x=dataFrame['blueGoldDiff'],
                                       bins=[-11000, 0, 11000],
                                       labels=['difNegativa', 'difPositiva'])
    dataFrame['blueCSPerMin'] = pd.cut(
        x=dataFrame['blueCSPerMin'],
        bins=[0, 11, 15, 19, 22, 30],
        labels=['mBaixo', 'Baixo', 'abMedia', 'Media', 'acMedia'])
    dataFrame['blueTotalVisao'] = pd.cut(
        x=dataFrame['blueTotalVisao'],
        bins=[0, 9, 18, 25, 255],
        labels=['mBaixo', 'abMedia', 'Media', 'acMedia'])
    dataFrame['blueAbates'] = pd.cut(
        x=dataFrame['blueAbates'],
        bins=[0, 10, 15, 25, 55],
        labels=['mBaixo', 'abMedia', 'Media', 'acMedia'])
    model = BM([#pais
    			('blueTotalJungleMinionsKilled', 'blueCSPerMin'),
                ('blueTotalMinionsKilled', 'blueCSPerMin'),
                ('blueTotalVisao', 'blueDragons'),
                #nao influenciam diretamente no blueWins
                ('blueCSPerMin', 'blueGoldDiff'),
                ('blueCSPerMin', 'blueTotalExperience'),
                ('blueTotalExperience', 'blueAvgLevelRounded'),
                ('blueAvgLevelRounded', 'blueAbates'),
                ('blueTotalExperience','blueExperienceDiff'),
                #influenciam diretamente no blueWins 
                ('blueGoldDiff', 'blueWins'),
                ('blueExperienceDiff', 'blueWins'),
                ('blueDragons', 'blueWins')])
    #Calculo de CPD com a estrategia MaximumLikehoodEstimator(default).
    model.fit(dataFrame)
    modelF = VariableElimination(model)
    #print(model.get_cpds('blueAvgLevelRounded'))
    return modelF


def main():
    df = read_input()
    modelo = Model_def(df)
    print('\n Sendo blueWins(0) como probabilidade do time azul perder(e, porventura, o time vermelho ganhar).')
    print('\n Probabilidade do time Azul ganhar caso esteja com vantagem em experiencia, porem o time vermelho esteja com vantagem em gold: ')
    q1=modelo.query(variables=['blueWins'],evidence={('blueExperienceDiff'):'difPositiva' , ('blueGoldDiff'):'difNegativa'})
    print(q1)
    print('\n Probabilidade do time Azul ganhar caso tenha um placar de visao acima da media, nenhum dragao, e um numero de abates na media(entre 15 e 25): ')
    q2=modelo.query(variables=['blueWins'],evidence={('blueTotalVisao'):'acMedia' , ('blueDragons'):0 , ('blueAbates'):'Media'})
    print(q2)
    #print('\n Probabilidade do time Azul ganhar caso a media de level do time azul seja de 7 , o CS seja acima da Media,  ')
    #q3=modelo.query(variables=['blueWins'],evidence={('blueFirstBlood'):1 , ('blueDragons'):0,('blueTotalExperience'):'muito' })
    #print(q3)


if __name__ == '__main__':
    main()
