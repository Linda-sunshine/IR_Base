package topicmodels;

import java.util.Arrays;
import java.util.Collection;

import markovmodel.LRFastRestrictedHMM_sentiment;
import structures._Corpus;
import structures._Doc;
import utils.Utils;
import LBFGS.LBFGS;
import LBFGS.LBFGS.ExceptionWithIflag;

// HTMM parameter both in log space
public class LRHTSM extends HTSM {	
	//feature weight vector for topic 
	double[] m_omega;
	double[] m_g_omega, m_diag_omega;//gradient and diagnoal for omega estimation
	
	//feature weight vector for sentiment
	double[] m_delta;
	double[] m_g_delta, m_diag_delta;//gradient and diagnoal for delta estimation
	
	//L2 regularization for omega
    //double m_lambda;    
    
	public LRHTSM(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			int number_of_topics, double alpha, //arguments for pLSA	
			double lambda) {//arguments for LR-HTMM		
		super(number_of_iteration, converge, beta, c, number_of_topics, alpha);

		//variable related to LR topic
		m_omega = new double [_Doc.stn_fv_size + 1];//bias + stn_transition_features
		m_g_omega = new double[m_omega.length];
		m_diag_omega = new double[m_omega.length];
		m_lambda = lambda;
		
		//variable related to LR sentiment
		m_delta = new double [_Doc.stn_senti_fv_size + 1];//bias + stn_senti_transition_features
		m_g_delta = new double[m_delta.length];
		m_diag_delta = new double[m_delta.length];
	
		m_hmm = new LRFastRestrictedHMM_sentiment(m_omega, m_delta, c.getLargestSentenceSize(), number_of_topics); 
	}
	
	@Override
	public String toString() {
		return String.format("LR-HTSM[k:%d, alpha:%.3f, beta:%.3f, lambda:%.2f]", number_of_topics, d_alpha, d_beta, m_lambda);
	}
	
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {
		super.initialize_probability(collection);
		Arrays.fill(m_omega, 0);
		Arrays.fill(m_delta, 0);
	}
	
	//accumulate sufficient statistics for epsilon, according to Eq(15) in HTMM note
	@Override
	void accEpsilonStat(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++) {
			double transit = 0;
			for(int i=0; i<(this.constant-1)*this.number_of_topics; i++) 
				transit += this.p_dwzpsi[t][i];
			d.getSentence(t-1).setTransit(transit);
			this.lot += transit;//we do not need this actually
			this.total ++;
		}
	}
	
	@Override
	// run upto number_of_topic since the first chunk is for sentiment switching
	void accSigmaStat(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++) {
			double transit = 0;
			for(int i=0; i<this.number_of_topics; i++) 
				transit += this.p_dwzpsi[t][i];
			d.getSentence(t-1).setSentiTransit(transit); 
			this.sigma_lot += transit;
		}
	}
	
	@Override
	public void calculate_M_step(int iter) {
		super.calculate_M_step(iter);
		estimateOmega();//maximum likelihood estimation for w
		estimateDelta();
	}
	
	void estimateOmega() {
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue;
		int fSize = m_omega.length;
		
		Arrays.fill(m_diag_omega, 0);//since we are reusing this space
		try{
			do {
				fValue = calcOmegaFuncGradient();
				LBFGS.lbfgs(fSize, 4, m_omega, fValue, m_g_omega, false, m_diag_omega, iprint, 1e-2, 1e-32, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
	}
	
	//log-likelihood: 0.5\lambda * w^2 + \sum_x [q(y=1|x) logp(y=1|x,w) + (1-q(y=1|x)) log(1-p(y=1|x,w))]
	//NOTE: L-BFGS code is for minimizing a problem
	double calcOmegaFuncGradient() {
		double p, q, g, loglikelihood = 0;
		
		//L2 normalization for omega
		for(int i=0; i<m_omega.length; i++) {
			m_g_omega[i] = m_lambda * m_omega[i];
			loglikelihood += m_omega[i] * m_omega[i];
		}
		loglikelihood *= m_lambda/2;
		
		double[] transitFv;
		for(_Doc d:m_corpus.getCollection()) {			
			for(int i=1; i<d.getSenetenceSize(); i++) {//start from the second sentence
				p = Utils.logistic(d.getSentence(i-1).getTransitFvs(), m_omega); // p(\epsilon=1|x, w)
				q = d.getSentence(i-1).getTransit(); // posterior of p(\epsilon=1|x, w)
				
				loglikelihood -= q * Math.log(p) + (1-q) * Math.log(1-p); // this is actually cross-entropy
				
				//collect gradient
				g = p - q;
				m_g_omega[0] += g;//for bias term
				transitFv = d.getSentence(i-1).getTransitFvs();
				for(int n=0; n<_Doc.stn_fv_size; n++)
					m_g_omega[1+n] += g * transitFv[n];
			}
		}
		
		return loglikelihood;
	}
	
	void estimateDelta() {
		int[] iflag = {0}, iprint = { -1, 3 };
		double fValue;
		int fSize = m_delta.length;
		
		Arrays.fill(m_diag_delta, 0);//since we are reusing this space
		try{
			do {
				fValue = calcDeltaFuncGradient();
				LBFGS.lbfgs(fSize, 4, m_delta, fValue, m_g_delta, false, m_diag_delta, iprint, 1e-2, 1e-32, iflag);
			} while (iflag[0] != 0);
		} catch (ExceptionWithIflag e){
			e.printStackTrace();
		}
	}
	
	
	//log-likelihood: 0.5\lambda * w^2 + \sum_x [q(y=1|x) logp(y=1|x,w) + (1-q(y=1|x)) log(1-p(y=1|x,w))]
	//NOTE: L-BFGS code is for minimizing a problem
	double calcDeltaFuncGradient() {
		double p, q, g, loglikelihood = 0;
		
		//L2 normalization for delta
		for(int i=0; i<m_delta.length; i++) {
			m_g_delta[i] = m_lambda * m_delta[i];
			loglikelihood += m_delta[i] * m_delta[i];
		}
		loglikelihood *= m_lambda/2;
		
		double[] transitFv;
		for(_Doc d:m_corpus.getCollection()) {			
			for(int i=1; i<d.getSenetenceSize(); i++) {//start from the second sentence
				p = Utils.logistic(d.getSentence(i-1).getSentiTransitFvs(), m_delta); // p(\epsilon=1|x, w)
				q = d.getSentence(i-1).getSentiTransit(); // posterior of p(\epsilon=1|x, w)
				
				loglikelihood -= q * Math.log(p) + (1-q) * Math.log(1-p); // this is actually cross-entropy
				
				//collect gradient
				g = p - q;
				m_g_delta[0] += g;//for bias term
				transitFv = d.getSentence(i-1).getSentiTransitFvs();
				for(int n=0; n<_Doc.stn_senti_fv_size; n++)
					m_g_delta[1+n] += g * transitFv[n];
			}
		}
		
		return loglikelihood;
	}
	
}
