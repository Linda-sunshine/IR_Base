package CoLinAdapt;

import java.util.ArrayList;
import java.util.Arrays;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

import structures._User;
import utils.Utils;

public class LogisticRegressionIndividual extends LogisticRegressionGlobal {
	
	public LogisticRegressionIndividual(ArrayList<_User> users, double[][] sims, int dim){
		super(users, sims, dim);	
	}
	
	public void init(){
		m_diag = new double[m_dim * m_users.size()];
		m_gs = new double[m_diag.length];
		m_ws = new double[m_diag.length];
	}
	
	public double calculateFValueGradients(){
		int[] neighborIndexes;
		_User user, neighbor;
		double fValue = 0, xij = 0, exp = 0, bias = 1; //Xij is the training instances, currently, we only have one 
		double[] sim;
//		double mag = 0;
		
		//Add the L2 Regularization.
		double L2 = 0, b = 0;
		for(int i=0; i<m_ws.length; i++){
			b = m_ws[i];
			m_gs[i] = 2*m_lambda*b;
			L2 += b*b;
		}
		
		for(int i=0; i<m_users.size(); i++){
			sim = m_sims[i];
			user = m_users.get(i);
			neighborIndexes = user.getNeighborIndexes();
			for(int j=0; j<neighborIndexes.length; j++){
				neighbor = m_users.get(neighborIndexes[j]);
				xij = Utils.cosine(user.getSparse(), neighbor.getSparse()); // Xij is the instance(bow, ..) currently.
				exp = Math.exp(-(m_ws[i * m_dim] * bias + m_ws[i * m_dim + 1] * xij)); // Current w^T*x is 2x1 vector, so we do multi directly.
				fValue += sim[j] * Math.log(1 + exp); // sim(i, j) * log(1 + exp(-w^T*xij))
				//Bias term for the user.
				m_gs[i * m_dim] += exp * (-bias) * sim[j] / (1 + exp);
				m_gs[i * m_dim + 1] += exp * (-xij) * sim[j] / (1 + exp); 
			}
		}
//		for(double g: m_gs)
//			mag += g * g;
//		// We have "-" for fValue and we want to maxmize the loglikelihood.
//		// LBFGS is minization, thus, we take off the negative sign in calculation.
//		System.out.format("Fvalue: %.4f\tGradient: %.4f\n", fValue, mag);
		return fValue + m_lambda*L2;
	}
	
	
	// Training process.
	public void train(){
		init();
		int[] iflag = { 0 }, iprint = { -1, 3 };
		double fValue;
		int fSize = m_dim * m_users.size();
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
	
	//UpdateSimialrity
	public void updateSimilarity(double[] similarity){
		int index, dim = 2;
		int[] neighborIndexes;
		double sim = 0, bias = 1, vsum = 0;
		double[] wi;
		for(int i=0; i<m_users.size(); i++){
			neighborIndexes = m_users.get(i).getNeighborIndexes();
			for(int j=0; j<neighborIndexes.length; j++){
				index = neighborIndexes[j];
				//Currently, the feature vector is two-dimension.
				wi = Arrays.copyOfRange(m_ws, i*dim, (i+1)*dim);
				vsum = wi[0] * bias + wi[1] * Utils.cosine(m_users.get(i).getSparse(), m_users.get(index).getSparse());
				sim = 1/(1 + Math.exp(-vsum)); //logit function.
				similarity[Utils.getIndex(i, index)] = sim;
			}
		}
	}
}
