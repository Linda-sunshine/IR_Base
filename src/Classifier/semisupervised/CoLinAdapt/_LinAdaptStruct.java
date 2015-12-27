package Classifier.semisupervised.CoLinAdapt;

import java.util.ArrayList;

import structures._Doc;
import structures._Review;
import structures._User;

public class _LinAdaptStruct {
	protected double[] m_A; // transformation matrix which is 2*(k+1) dimension.
	protected _User m_user; // unit to store train/adaptation/test data
	protected int m_dim; // number of feature groups
	
	public _LinAdaptStruct(_User user, int dim) {
		m_user = user;
		
		m_dim = dim;
		m_A = new double[dim*2];		
		for(int i=0; i < m_dim; i++)
			m_A[i] = 1;//Scaling in the first dim dimensions. Initialize scaling to be 1 and shifting be 0.
	}
	
	public ArrayList<_Review> getReviews(){
		return m_user.getReviews();
	}
	
	public void setPersonalizedModel(double[] pWeight, int classNo, int featureSize) {
		m_user.setModel(pWeight, classNo, featureSize);
	}
	
	public int predict(_Doc doc) {
		return m_user.predict(doc);
	}
	
	public double[] getA() {
		return m_A;
	}
	
	//get the shifting operation for this group
	public double getShifting(int gid) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the scope of feature grouping!\n", gid);
			return Double.NaN;
		}
		return m_A[m_dim+gid];
	}
	
	//get the shifting operation for this group
	public double getScaling(int gid) {
		if (gid<0 || gid>m_dim) {
			System.err.format("[Error]%d is beyond the scope of feature grouping!\n", gid);
			return Double.NaN;
		}
		return m_A[gid];
	}
}
