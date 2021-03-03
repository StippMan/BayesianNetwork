from pgmpy.models import BayesianModel as BM
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import pandas as pd

def read_input():
	entrada = pd.read_csv('high_diamond_ranked_10min.csv')
	entrada.drop(['gameId','redKills','redDeaths','blueGoldDiff','redGoldDiff','blueExperienceDiff','redExperienceDiff','blueCSPerMin','redCSPerMin','blueAssists','redAssists','blueWardsPlaced','redWardsPlaced','blueWardsDestroyed','redWardsDestroyed','redFirstBlood','blueAvgLevel',
                          'redAvgLevel','blueHeralds','redHeralds','blueTotalJungleMinionsKilled',
                          'redTotalJungleMinionsKilled','blueTowersDestroyed','redTowersDestroyed'],axis = 1,inplace=True)
	return entrada

def Model_def(dataFrame):
	predict_data = dataFrame[:2500]
	train_data = dataFrame[2500:]
	model = BM([('blueFirstBlood','blueWins'),('blueTotalExperience','blueWins'),('blueDragons','blueWins')])
	#Calculo de CPD com a estrategia MaximumLikehoodEstimator.
	model.fit(train_data,estimator=MaximumLikelihoodEstimator)
	#print(model.get_cpds('blueWins'))
	#pra avaliacao final da predicao.
	predict_data = predict_data.copy()
	#predict_data.drop(['blueFirstBlood','blueTotalExperience','blueDragons',
					   #'blueKills','blueDeaths','blueEliteMonsters','blueTotalGold','blueTotalMinionsKilled','blueGoldPerMin',
					   #'redEliteMonsters','redDragons','redTotalGold','redTotalExperience','redTotalMinionsKilled','redGoldPerMin'], axis=1, inplace=True)
	predict_data.drop('blueWins', axis=1, inplace=True)
	#print(predict_data.info())
	y_pred = model.predict(predict_data)
	#inferenciando
	print(y_pred['blueWins'])
	modelF = VariableElimination(model)
	return modelF


def main():	
	df=read_input()
	modelo = Model_def(df)	
	#print('\n Probabilidade do time Azul ganhar,caso o time vermelho de first blood. ')
	#q1=modelo.query(variables=['blueWins'],evidence={'blueFirstBlood':0})
	#print(q1)
	#print('\n Probabilidade do time Azul ganhar,caso o time azul de first blood e tenha mais dragoes: ')
	#q2=modelo.query(variables=['blueWins'],evidence={('blueFirstBlood'):1 , ('blueDragons'):1})
	#print(q2)
	#print('\n Probabilidade do time Azul ganhar,caso o time azul de first blood , tenha menos xp e menos dragoes ')
	#q3=modelo.query(variables=['blueWins'],evidence={('blueFirstBlood'):1 , ('blueDragons'):0 ,('blueTotalExperience'):0 })
	#print(q3)
if __name__ == '__main__':
    main()
