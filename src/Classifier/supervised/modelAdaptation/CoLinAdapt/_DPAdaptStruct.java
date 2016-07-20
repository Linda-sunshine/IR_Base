package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.Arrays;

import structures._Doc;
import structures._SparseFeature;
import structures._User;
import structures._thetaStar;
import utils.Utils;

public class _DPAdaptStruct extends _LinAdaptStruct{

	_thetaStar m_thetaStar = null;
	_thetaStar[] m_thetaStars;
	double[] m_cProb;
	public _DPAdaptStruct(_User user) {
		super(user);
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
	
	public double getScaling(int k){
		return m_thetaStar.m_beta[k];
	}
	public double getShifing(int k){
		return m_thetaStar.m_beta[m_dim+k];
	}
	// The probability that the user belongs to different clusters.
	public void setCProb(double[] prob){
		m_cProb = Arrays.copyOf(prob, prob.length);
	}
	public void setThetaStars(_thetaStar[] ts){
		m_thetaStars = ts;
	}
	@Override
	// We need different prediction method.
	public int predict(_Doc doc) {
		_SparseFeature[] fv = doc.getSparse();
		double prob = 0;
		for(int i=0; i<m_cProb.length; i++){
			prob += m_cProb[i]*Utils.dotProduct(m_thetaStars[i].m_beta, fv, 0);			
		}
		return prob>0?1:0;
	}
}
