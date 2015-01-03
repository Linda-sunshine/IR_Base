package topicmodels;

import java.util.Arrays;

import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;
import markovmodel.LRFastRestrictedHMM;
import structures._Corpus;
import structures._Doc;
import utils.Utils;

// HTMM parameter both in log space
public class LRHTMM extends HTMM {	
	//feature weight vector
	double[] m_omega;
	double[] m_g_omega, m_diag_omega;//gradient and diagnoal for omega estimation
	
	//L2 regularization for omega
    double m_lambda;
    
	public LRHTMM(int number_of_topics, double d_alpha, double d_beta, int number_of_iteration, _Corpus c) {
		super(number_of_topics, d_alpha, d_beta, number_of_iteration, c);
		
		//variable related to LR
		m_omega = new double [_Doc.stn_fv_size + 1];//bias + stn_transition_features
		m_g_omega = new double[m_omega.length];
		m_diag_omega = new double[m_omega.length];
		m_lambda = 0.05;
	}
	
	//convert them to log-space (pLSA is not running in log-space!!!)
	@Override
	protected void initialize_probability() {	
		super.initialize_probability();
		
		for(_Doc d:m_corpus.getCollection())
			d.setSentenceFeatureVector(); // sentence feature vector only in LRHTMM
	}
	
	@Override
	public void calculate_E_step(_Doc d) {
		//Step 1: pre-compute emission probability
		ComputeEmissionProbsForDoc(d);
		
		//Step 2: use forword/backword algorithm to compute the posterior
		LRFastRestrictedHMM f = new LRFastRestrictedHMM(m_omega); 
		loglik += f.ForwardBackward(d, emission);
		
		//Step 3: collection expectations from the posterior distribution
		f.collectExpectations(p_dwzpsi);//expectations will be in the original space		
		accEpsilonStat(d);
		accPhiStat(d);
		estThetaInDoc(d);
	}
	
	@Override
	public int[] get_MAP_topic_assignment(_Doc d) {
		int path [] = new int [d.getSenetenceSize()];
		LRFastRestrictedHMM v = new LRFastRestrictedHMM(m_omega);
		v.BackTrackBestPath(d, emission, path);
		return path;
	}	
	
	//accumulate sufficient statistics for epsilon, according to Eq(15) in HTMM note
	@Override
	void accEpsilonStat(_Doc d) {
		double[] label = d.m_sentence_labels;
		for(int t=1; t<d.getSenetenceSize(); t++) {
			label[t-1] = 0; 
			for(int i=0; i<this.number_of_topics; i++)
				label[t-1] += this.p_dwzpsi[t][i];
			this.lot += label[t-1];//we do not need this actually
			this.total ++;
		}
	}
	
	@Override
	public void calculate_M_step() {
		super.calculate_M_step();
		estimateOmega();//maximum likelihood estimation for w
	}
	
	void estimateOmega() {
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue;
		int fSize = m_omega.length;
		
		Arrays.fill(m_diag_omega, 0);//since we are reusing this space
		try{
			do {
				fValue = calcFuncGradient();
				LBFGS.lbfgs(fSize, 4, m_omega, fValue, m_g_omega, false, m_diag_omega, iprint, 1e-2, 1e-20, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
	}
	
	//log-likelihood: 0.5\lambda * w^2 + \sum_x [q(y=1|x) logp(y=1|x,w) + (1-q(y=1|x)) log(1-p(y=1|x,w))]
	//NOTE: L-BFGS code is for minimizing a problem
	double calcFuncGradient() {
		double p, q, g, loglikelihood = 0;
		
		//L2 normalization for omega
		for(int i=0; i<m_omega.length; i++) {
			m_g_omega[i] = m_lambda * m_omega[i];
			loglikelihood += m_omega[i] * m_omega[i];
		}
		loglikelihood *= m_lambda/2;
		
		for(_Doc d:m_corpus.getCollection()) {			
			for(int i=1; i<d.getSenetenceSize(); i++) {//start from the second sentence
				p = Utils.logistic(d.m_sentence_features[i-1], m_omega); // p(\epsilon=1|x, w)
				q = d.m_sentence_labels[i-1]; // posterior of p(\epsilon=1|x, w)
				
				if (p==1 || p==0)//temporary code for debugging purpose
					System.err.println("Error!");
				loglikelihood -= q * Math.log(p) + (1-q) * Math.log(1-p); // this is actually cross-entropy
				
				//collect gradient
				g = p - q;
				m_g_omega[0] += g;//for bias term
				for(int n=0; n<_Doc.stn_fv_size; n++)
					m_g_omega[1+n] += g * d.m_sentence_features[i-1][n];
			}
		}
		
		return loglikelihood;
	}
}
