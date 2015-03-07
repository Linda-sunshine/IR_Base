package Classifier.supervised;

import java.util.Arrays;
import java.util.Collection;

import optimization.gradientBasedMethods.ProjectedGradientDescent;
import optimization.gradientBasedMethods.stats.OptimizerStats;
import optimization.linesearch.ArmijoLineSearchMinimizationAlongProjectionArc;
import optimization.linesearch.InterpolationPickFirstStep;
import optimization.linesearch.LineSearchMethod;
import optimization.stopCriteria.CompositeStopingCriteria;
import optimization.stopCriteria.ProjectedGradientL2Norm;
import optimization.stopCriteria.StopingCriteria;
import posteriorRegularization.logisticRegression.PointwisePR;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

public class PRLogisticRegression extends LogisticRegression {

	double[][] m_doc_pr;//to store exp(\lambda\phi(x,y))
	
	public PRLogisticRegression(_Corpus c, int classNo, int featureSize, double lambda){
		super(c, classNo, featureSize, lambda);
	}
	
	@Override
	public String toString() {
		return String.format("PR Logistic Regression[C:%d, F:%d, L:%.2f]", m_classNo, m_featureSize, m_lambda);
	}

	//integrate LR posterior calculation with PR posterior calculation
	void calcPosterior(_SparseFeature[] spXi, double[] scaler, double[] prob) {
		int offset;
		for(int i = 0; i < m_classNo; i++){
			offset = i * (m_featureSize + 1);
			m_cProbs[i] = Utils.dotProduct(m_beta, spXi, offset);
		}
		
		double logSum = Utils.logSumOfExponentials(m_cProbs);
		for(int i = 0; i < m_classNo; i++) {
			prob[i] = Math.exp(m_cProbs[i]-logSum);
			if (scaler != null)
				prob[i] *= scaler[i];
		}
		
		if (scaler!=null)
			Utils.scaleArray(prob, 1.0/Utils.sumOfArray(prob));
	}
	
	/*
	 * Calculate the beta by using bfgs. In this method, we give a starting
	 * point and iterating the algorithm to find the minimum value for the beta.
	 * The input is the vector of feature[14], we need to pass the function
	 * value for the point, together with the gradient vector. When the iflag
	 * turns to 0, it finds the final point and we get the best beta.
	 */	
	@Override
	public void train(Collection<_Doc> trainSet) {
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue = 0, lastFValue = 1, converge;
		int fSize = m_beta.length, iter = 0;
		m_doc_pr = new double[trainSet.size()][m_classNo];//the dimensionality of PR factors is fixed
		
		init();
		do {
			//E-step
			Estep(trainSet);
			
			//M-step
			try{			
				do {
					fValue = calcFuncGradient(trainSet);
					LBFGS.lbfgs(fSize, 5, m_beta, fValue, m_g, false, m_diag, iprint, 1e-4, 1e-20, iflag);
				} while (iflag[0] != 0);
			} catch (ExceptionWithIflag e){
				e.printStackTrace();
			}
			
			converge = (lastFValue - fValue) / lastFValue;
			lastFValue = fValue;
			System.out.print(lastFValue + ", ");
		} while (++iter<10 && Math.abs(converge)>1e-3);
		System.out.println();
	}
	
	void Estep(Collection<_Doc> trainSet) {
		double gdelta = 1e-5, istp = 1.0;
		int maxStep = 50;
		
		//The computation complexity is n*classNo.
		int doc_index = 0;
		for (_Doc doc: trainSet) {
			// getting all the class probability using basic LR
			calcPosterior(doc.getSparse(), null, m_cache);
			
			// then we are regularizing the PR here
			//PairwisePR testcase = new PairwisePR(m_cache, doc.getYLabel(), m_classNo); // Yi is the true label	
			PointwisePR testcase = new PointwisePR(m_cache, doc.getYLabel(), m_classNo); // Yi is the true label		
			testcase.setDebugLevel(-1);
			
			LineSearchMethod ls = new ArmijoLineSearchMinimizationAlongProjectionArc(new InterpolationPickFirstStep(istp));
			ProjectedGradientDescent optimizer = new ProjectedGradientDescent(ls);
			StopingCriteria stopGrad = new ProjectedGradientL2Norm(gdelta);
			CompositeStopingCriteria compositeStop = new CompositeStopingCriteria();
			compositeStop.add(stopGrad);
			optimizer.setMaxIterations(maxStep);
			
			if (optimizer.optimize(testcase, new OptimizerStats(), compositeStop))
				testcase.getPosteriorScaler(m_doc_pr[doc_index]); // get the regularized PR scaler here
			else
				Arrays.fill(m_doc_pr[doc_index], 0); // no posterior regularization
			
			doc_index++;
		}
	}
	
	//This function is used to calculate the value and gradient with the new beta.
	@Override
	protected double calcFuncGradient(Collection<_Doc> trainSet) {
		double gValue = 0, fValue = 0;
		double Pij = 0, logPij = 0;
		
		// Add the L2 regularization.
		double L2 = 0, b;
		for(int i = 0; i < m_beta.length; i++) {
			b = m_beta[i];
			m_g[i] = 2 * m_lambda * b;
			L2 += b * b;
		}
		
		int Yi;
		_SparseFeature[] fv;
		int doc_index = 0;
		for (_Doc doc: trainSet) {
			fv = doc.getSparse();
			Yi = doc.getYLabel();
			
			//compute posterior regularized q(y=j|xi)\propto p(y=j|xi)exp(\lambda\phi(y=j, y)
			calcPosterior(fv, m_doc_pr[doc_index], m_cache);
			
			for(int j = 0; j < m_classNo; j++){
				Pij = m_cache[j];
				logPij = Math.log(Pij);
				
				if (Yi == j){
					gValue =  Pij - 1;
					fValue += logPij;
				} else
					gValue = Pij;				
				
				int offset = j * (m_featureSize + 1);
				m_g[offset] += gValue;
				//(Yij - Pij) * Xi
				for(_SparseFeature sf: fv)
					m_g[offset + sf.getIndex() + 1] += gValue * sf.getValue();
			}
			
			doc_index++;
		}
		
		return m_lambda*L2 - fValue;
	}
}
