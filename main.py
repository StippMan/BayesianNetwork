import math
#import pomegranate as pm
from pgmpy.models import BayesianModel as BM
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import numpy as np
#pro dataset
import pandas as pd

def read_input():
	#Le a entrada, exclui todas as linhas que possuem algum valor Nulo.(' ?' no dataset)..
	d = ["?"]
	entrada = pd.read_csv('adult.data',header=None)
	entrada.columns = ['age','workclass',"fnlwgt","education","education-num","marital-status",
					  "occupation","relationship","race","sex","capital-gain","capital-loss",
					  "hours-per-week","native-country", "wage"]
	entrada.replace(' ?',np.nan,inplace = True)
	entrada.dropna(inplace = True)
	return entrada

def Model_def(dataFrame):
	model = BM([('age','wage'),('age','education'),('education','workclass'),('workclass','wage')])
	#CPD com estimador maxima
	model.fit(dataFrame,estimator=MaximumLikelihoodEstimator)
	#inferenciando
	modelF = VariableElimination(model)
	return modelF

def main():	
	df=read_input()
	modelo = Model_def(df)
	print('\n Probabilidade do cidadao ganhar >50k, dado a idade = 25')
	q2=modelo.query(variables=['wage'],evidence={'age':25})
	#q1=HeartDiseasetest_infer.query(variables=['heartdisease'],)
	print(q2)
if __name__ == '__main__':
    main()
