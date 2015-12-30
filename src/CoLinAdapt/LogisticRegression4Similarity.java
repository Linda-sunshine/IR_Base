package CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

import structures._User;
import utils.Utils;

public class LogisticRegression4Similarity {
	ArrayList<_User> m_users;
	double[][] m_Diffs;

	//Parameters used in LBFGS.
	double[] m_diag;
	double[] m_gs;
	double[] m_ws; //Ws is weights for all users, what we want to optimize.
				   //Currently, we only have one feature for each pair, we will introduce more later.
	int m_dim; //Dimension of the pairwise features.
	
	public LogisticRegression4Similarity(ArrayList<_User> users, double[][] Diffs){
		m_users = users;
		m_Diffs = Diffs;
		m_dim = 2;//Currently, we only have one BoW similarity and bias.
		m_diag = new double[m_dim * m_users.size()];
		m_gs = new double[m_diag.length];
		m_ws = new double[m_diag.length];
	}
	
	public void initLBFGS(){
		Arrays.fill(m_diag, 0);
		Arrays.fill(m_gs, 0);
	}
	
	public double calculateFValueGradients(){
		int[] neighborIndexes;
		_User user, neighbor;
		double fValue = 0, xij = 0, exp = 0, bias = 1; //Xij is the training instances, currently, we only have one 
		double[] Diff;
		
		for(int i=0; i<m_users.size(); i++){
			Diff = m_Diffs[i];
			user = m_users.get(i);
			neighborIndexes = user.getNeighborIndexes();
			for(int j=0; j<neighborIndexes.length; j++){
				neighbor = m_users.get(neighborIndexes[j]);
				xij = Utils.cosine(user.getSparse(), neighbor.getSparse()); // Xij is the instance(bow, ..) currently.
				exp = Math.exp(-(m_ws[i * m_dim] * bias + m_ws[i * m_dim + 1] * xij)); // Current w^T*x is 2x1 vector, so we do multi directly.
				fValue += Diff[j] * Math.log(1 + exp); //DiffAij * log(1 + exp(-w^T*xij))
				//Bias term for the user.
				m_gs[i * m_dim] += Diff[j] * exp * (-bias); 
				m_gs[i * m_dim + 1] += Diff[j] * exp * (-xij); 
			}
		}
		// We have "-" for fValue and we want to maxmize the loglikelihood.
		// LBFGS is minization, thus, we take off the negative sign in calculation.
		return fValue;
	}
	
	public void train(){
		int[] iflag = { 0 }, iprint = { -1, 3 };
		double fValue;
		int fSize =  m_users.size() * m_dim;
		initLBFGS();
		
		try {
			do {
				fValue = calculateFValueGradients();
				LBFGS.lbfgs(fSize, 6, m_ws, fValue, m_gs, false, m_diag, iprint, 1e-4, 1e-10, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e) {
			e.printStackTrace();
		}
	}
	
	// Return the trained wights.
	public double[] getWeights(){
		return m_ws;
	}
}
