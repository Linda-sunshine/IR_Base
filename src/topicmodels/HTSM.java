package topicmodels;

import java.util.Random;
import markovmodel.FastRestrictedHMM_sentiment;
import structures._Corpus;
import structures._Doc;
import utils.Utils;

public class HTSM extends HTMM{
	
	double sigma;
	int sigma_total;
	double sigma_lot;
	
	public FastRestrictedHMM_sentiment m_htsm;
	public HTSM(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			int number_of_topics, double alpha) {//arguments for HTMM	
		super(number_of_iteration, converge, beta, c, number_of_topics, alpha, 3); // 3 is constant
		
		Random r = new Random();
		this.sigma = r.nextDouble();
		int maxSeqSize = c.getLargestSentenceSize();	
		m_htsm = new FastRestrictedHMM_sentiment(epsilon, sigma, maxSeqSize, this.number_of_topics); 
	}
	
	@Override
	public double calculate_E_step(_Doc d) {
		//Step 1: pre-compute emission probability
		ComputeEmissionProbsForDoc(d);
		
		//Step 2: use forword/backword algorithm to compute the posterior
		double logLikelihood = m_htsm.ForwardBackward(d, emission) + docThetaLikelihood(d);
		loglik += logLikelihood;
		
		//Step 3: collection expectations from the posterior distribution
		m_htsm.collectExpectations(p_dwzpsi);//expectations will be in the original space	
		accTheta(d);
		
		if (m_collectCorpusStats) {
			accEpsilonStat(d);
			accSigmaStat(d);
			accPhiStat(d);
		}
		
		return logLikelihood;
	}
	
	
	// run upto number_of_topic since the first chunk is for sentiment switching
	void accSigmaStat(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<this.number_of_topics; i++) 
				this.sigma_lot += this.p_dwzpsi[t][i];
			this.sigma_total ++;
		}
	} 
	
	//accumulate sufficient statistics for epsilon, according to Eq(15) in HTMM note
	// first two chunk are for topic switch, that is why (this.constant - 1)
	@Override
	void accEpsilonStat(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<(this.constant-1)*this.number_of_topics; i++) 
				this.lot += this.p_dwzpsi[t][i];
			this.total ++;
		}
	}

	
	@Override
	public void calculate_M_step(int iter) {
		this.epsilon = this.lot/this.total; // to make the code structure concise and consistent, keep epsilon in real space!!
		this.sigma = this.sigma_lot/this.sigma_total;
		
		for(int i=0; i<this.number_of_topics; i++) {
			double sum = Math.log(Utils.sumOfArray(word_topic_sstat[i]));
			for(int v=0; v<this.vocabulary_size; v++)
				topic_term_probabilty[i][v] = Math.log(word_topic_sstat[i][v]) - sum;
		}
		
		for(_Doc d:m_trainSet)
			estThetaInDoc(d);
		
		m_htsm.setEpsilon(this.epsilon);
		m_htsm.setSigma(this.sigma);
	}
	
	protected void init() {
	
		super.init();
		this.sigma_lot = 0.0; // sufficient statistics for sigma	
		this.sigma_total = 0;
		
	}
	
	@Override
	public String toString() {
		return String.format("HTSM[k:%d, alpha:%.3f, beta:%.3f]", number_of_topics, d_alpha, d_beta);
	}
	
	@Override
	public int[] get_MAP_topic_assignment(_Doc d) {
		int path [] = new int [d.getSenetenceSize()];
		m_htsm.BackTrackBestPath(d, emission, path);
		return path;
	}
	
	
	@Override
	public double calculate_log_likelihood(_Doc d) {//it is very expensive to re-compute this
		//Step 1: pre-compute emission probability
		ComputeEmissionProbsForDoc(d);		
		
		double logLikelihood = 0;
		for(int i=0; i<this.number_of_topics; i++) 
			logLikelihood += (d_alpha-1)*d.m_topics[i];
		return logLikelihood + m_htsm.ForwardBackward(d, emission);
	}
	
}
