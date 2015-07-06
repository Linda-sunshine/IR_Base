package topicmodels;

import markovmodel.FastRestrictedHMM_sentiment;
import markovmodel.FastRestrictedHMM_sentiment_normalized;
import structures._Corpus;
import structures._Doc;

public class HTSM extends HTMM {
	
	double sigma;
	double sigma_lot; // sufficient statistic about sigma
	
	public HTSM(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			int number_of_topics, double alpha) {//arguments for HTMM	
		super(number_of_iteration, converge, beta, c, number_of_topics, alpha, 3); // 3 is constant
		
		if (number_of_topics%2!=0) {
			System.err.println("[Error]In HTSM the number of topics has to be even!");
			System.exit(-1);
		}
		
		this.sigma = Math.random();
		m_hmm = new FastRestrictedHMM_sentiment_normalized(epsilon, sigma, c.getLargestSentenceSize(), this.number_of_topics); 
	}
	
	@Override
	public double calculate_E_step(_Doc d) {
		double logLikelihood = super.calculate_E_step(d);
		
		if (m_collectCorpusStats)
			accSigmaStat(d);
		
		return logLikelihood;
	}
	
	// run upto number_of_topic since the first chunk is for sentiment switching
	void accSigmaStat(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<this.number_of_topics; i++) 
				this.sigma_lot += this.p_dwzpsi[t][i];
		}
	}
	
	@Override
	public void calculate_M_step(int iter) {
		super.calculate_M_step(iter);
		
		if (iter>0) {
			this.sigma = this.sigma_lot / this.total;
			((FastRestrictedHMM_sentiment_normalized)m_hmm).setSigma(this.sigma);
		}
	}
	
	protected void init() {
		super.init();
		this.sigma_lot = 0.0; // sufficient statistics for sigma	
	}
	
	@Override
	public String toString() {
		return String.format("HTSM[k:%d, alpha:%.3f, beta:%.3f]", number_of_topics, d_alpha, d_beta);
	}
	
	@Override
	public int[] get_MAP_topic_assignment(_Doc d) {
		int path [] = new int [d.getSenetenceSize()];
		((FastRestrictedHMM_sentiment_normalized)m_hmm).BackTrackBestPath(d, emission, path);
		return path;
	}
	
	@Override
	public double calculate_log_likelihood(_Doc d) {//it is very expensive to re-compute this
		//Step 1: pre-compute emission probability
		ComputeEmissionProbsForDoc(d);		
		
		double logLikelihood = 0;
		for(int i=0; i<this.number_of_topics; i++) 
			logLikelihood += (d_alpha-1)*d.m_topics[i];
		return logLikelihood + m_hmm.ForwardBackward(d, emission);
	}
	
	// perform inference of topic distribution in the document
	@Override
	public double inference(_Doc d) {
		initTestDoc(d);//this is not a corpus level estimation
		
		double delta, last = 1, current;
		int  i = 0;
		do {
			init();
			current = calculate_E_step(d);
			estThetaInDoc(d);			
			delta = (last - current)/last;
			last = current;
		} while (Math.abs(delta)>m_converge && ++i<this.number_of_iteration);
		int path[] = get_MAP_topic_assignment(d);
		System.out.println("Doc No: "+d.getID());
		for(i=0; i<path.length;i++)
			System.out.print(path[i]+",");
		System.out.println();
		
		return current;
	}
	
}
