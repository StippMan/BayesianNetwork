from pgmpy.models import BayesianModel as BM
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd

def read_input():
	entrada = pd.read_csv('high_diamond_ranked_10min.csv')
	#print(entrada['blueCSPerMin'].mean(axis=0))
	entrada['blueTotalVisao'] = entrada['blueWardsPlaced'] + entrada['blueWardsDestroyed'] #- entrada['redWardsPlaced']
	entrada['blueAbates'] = entrada['blueKills'] + entrada['blueAssists']
	print(entrada['blueAbates'].min(),entrada['blueAbates'].max(),entrada['blueAbates'].mean(axis=0))
	entrada.drop(['gameId','redKills','redDeaths','redGoldDiff','blueExperienceDiff','redExperienceDiff','redCSPerMin','blueAssists','redAssists','blueWardsPlaced','redWardsPlaced','blueWardsDestroyed','redWardsDestroyed','redFirstBlood','blueAvgLevel',
                          'redAvgLevel','blueHeralds','redHeralds','blueTotalJungleMinionsKilled',
                          'redTotalJungleMinionsKilled','blueTowersDestroyed','redTowersDestroyed'],axis = 1,inplace=True)
	return entrada

def Model_def(dataFrame):
	#min,max,mean blueTotalMinionsKilled = 90, 283, 215.
	#min,max,mean blueGoldDiff = -10830, 11467, 14.4
	#min,max,mean blueCSPerMin = 9, 28.3, 21.6
	#min,max,mean blueTotalVisao = 5, 254, 25.11
	dataFrame.drop(['blueKills','blueDeaths','blueEliteMonsters','blueTotalGold','blueGoldPerMin','redEliteMonsters',
					'redDragons','redTotalGold','redTotalExperience','redTotalMinionsKilled','redGoldPerMin'], axis=1, inplace=True)
	dataFrame['blueTotalExperience'] = pd.cut(x=dataFrame['blueTotalExperience'], bins=[0,11000,14000,17000,30000],labels = ['pouco',
																										   'media','acima-media','muito']) 
	dataFrame['blueTotalMinionsKilled'] = pd.cut(x=dataFrame['blueTotalMinionsKilled'], bins=[0,215,300],labels = ['abaixo-media','acima-media'])
	dataFrame['blueGoldDiff'] = pd.cut(x=dataFrame['blueGoldDiff'], bins=[-11000,0,11000],labels = ['difNegativa','divPositiva'])
	dataFrame['blueCSPerMin'] = pd.cut(x=dataFrame['blueCSPerMin'], bins=[0,11,15,19,22,30],labels = ['mBaixo','Baixo','abMedia','Media','acMedia'])
	dataFrame['blueTotalVisao'] = pd.cut(x=dataFrame['blueTotalVisao'], bins=[0,9,18,25,255],labels = ['Baixo','abMedia','Media','acMedia'])	
	dataFrame['blueAbates'] = pd.cut(x=dataFrame['blueAbates'], bins=[0,10,15,25,55],labels = ['Baixo','abMedia','Media','acMedia'])
	print(dataFrame.info())
	#model = BM([('blueFirstBlood','blueWins'),('blueTotalExperience','blueWins'),('blueDragons','blueWins')])
	model = BM([('blueCSPerMin','blueGoldDiff'),('blueTotalMinionsKilled','blueTotalExperience'),('blueGoldDiff','blueWins'),('blueTotalExperience','blueWins'),
				('blueAbates','blueGoldDiff'),('blueDragons','blueWins'),('blueTotalVisao','blueWins')])
	#Calculo de CPD com a estrategia MaximumLikehoodEstimator(default).
	model.fit(dataFrame)
	modelF = VariableElimination(model)
	return modelF


def main():	
	df=read_input()
	modelo = Model_def(df)	
	#print('\n Probabilidade do time Azul ganhar,caso o time vermelho de first blood. ')
	#q1=modelo.query(variables=['blueWins'],evidence={'blueFirstBlood':0})
	#print(q1)
	#print('\n Probabilidade do time Azul ganhar,caso o time azul de first blood e tenha um dragao: ')
	#q2=modelo.query(variables=['blueWins'],evidence={('blueFirstBlood'):1 , ('blueDragons'):1})
	#print(q2)
	#print('\n Probabilidade do time Azul ganhar,caso o time azul nao de first blood , muita xp e nenhum dragao ')
	#q3=modelo.query(variables=['blueWins'],evidence={('blueFirstBlood'):1 , ('blueDragons'):0,('blueTotalExperience'):'muito' })
	#print(q3)
if __name__ == '__main__':
    main()
