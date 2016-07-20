package Classifier.supervised.modelAdaptation.DirichletProcess;

import Classifier.supervised.modelAdaptation.CoLinAdapt._LinAdaptStruct;
import structures._Doc;
import structures._User;
import structures._thetaStar;
import utils.Utils;

public class _DPAdaptStruct extends _LinAdaptStruct {

	_thetaStar m_thetaStar = null;
	double[] m_cluPosterior;
	
	public _DPAdaptStruct(_User user) {
		super(user, 0); // will not perform adaptation
	}
	
	public _DPAdaptStruct(_User user, int dim) {
		super(user, dim);
	}
	
	public _thetaStar getThetaStar(){
		return m_thetaStar;
	}
	
	public void setThetaStar(_thetaStar s){
		m_thetaStar = s;
	}
	
	public void setClusterPosterior(double[] posterior) {
		m_cluPosterior = new double[posterior.length];
		System.arraycopy(posterior, 0, m_cluPosterior, 0, posterior.length);
	}
	
	@Override
	public double getScaling(int k){
		return m_thetaStar.getModel()[k];
	}
	
	@Override
	public double getShifting(int k){
		return m_thetaStar.getModel()[m_dim+k];
	}
	
	@Override
	public int predict(_Doc doc) {
		double prob = 0, sum;
		for(int k=0; k<m_cluPosterior.length; k++) {
			sum = Utils.dotProduct(CLogisticRegressionWithDP.m_thetaStars[k].getModel(), doc.getSparse(), 0);//need to be fixed: here we assumed binary classification
			prob += m_cluPosterior[k] * Utils.logistic(sum); 
		}
		return prob>0 ? 1:0;
	}
}
