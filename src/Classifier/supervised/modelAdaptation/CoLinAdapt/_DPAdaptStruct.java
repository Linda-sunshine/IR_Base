package Classifier.supervised.modelAdaptation.CoLinAdapt;

import structures._User;
import structures._thetaStar;

public class _DPAdaptStruct extends _LinAdaptStruct{

	_thetaStar m_thetaStar = null;
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
}
