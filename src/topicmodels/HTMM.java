package topicmodels;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

import markovmodel.FastRestrictedHMM;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;

public class HTMM extends pLSA {
	// HTMM parameter both in log space
	double epsilon;   // estimated epsilon
	
	// cache structure
	double[][] p_dwzpsi;  // The state probabilities that is Pr(z,psi | d,w)
 	double[][] emission;  // emission probability of p(s|z)
	
 	// HMM-style inferencer
 	FastRestrictedHMM m_hmm; 
 	
	// sufficient statistics for p(w|\phi_z) and p(\epsilon)
	int total; // used for epsilion
	double lot; // used for epsilion
	
	double loglik;
	final int constant = 2;

	public HTMM(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			int number_of_topics, double alpha) {//arguments for pLSA	
		super(number_of_iteration, converge, beta, c,
				0.5, null, //HTMM does not have a background setting
				number_of_topics, alpha);
		
		Random r = new Random();
		this.epsilon = r.nextDouble();
		
		int maxSeqSize = c.getLargestSentenceSize();		
		m_hmm = new FastRestrictedHMM(epsilon, maxSeqSize, this.number_of_topics); 
		
		//cache in order to avoid frequently allocating new space
		p_dwzpsi = new double[maxSeqSize][constant * this.number_of_topics]; // max|S_d| * (2*K)
		emission = new double[p_dwzpsi.length][this.number_of_topics]; // max|S_d| * K
	}
	
	public HTMM(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			int number_of_topics, double alpha, //arguments for pLSA
			boolean setHMM) { //just to indicate we don't need initiate hmm inferencer
		super(number_of_iteration, converge, beta, c,
				0.5, null, //HTMM does not have a background setting
				number_of_topics, alpha);
		
		Random r = new Random();
		this.epsilon = r.nextDouble();
		
		int maxSeqSize = c.getLargestSentenceSize();		
		if (setHMM)
			m_hmm = new FastRestrictedHMM(epsilon, maxSeqSize, this.number_of_topics); 
		else
			m_hmm = null;
		
		//cache in order to avoid frequently allocating new space
		p_dwzpsi = new double[maxSeqSize][constant * this.number_of_topics]; // max|S_d| * (2*K)
		emission = new double[p_dwzpsi.length][this.number_of_topics]; // max|S_d| * K
	}
	
	@Override
	public String toString() {
		return String.format("HTMM[k:%d, alpha:%.3f, beta:%.3f]", number_of_topics, d_alpha, d_beta);
	}
	
	//convert them to log-space (pLSA is not running in log-space!!!)
	@Override
	protected void initialize_probability(Collection<_Doc> collection) {	
		super.initialize_probability(collection);
		
		//need to convert into log-space
		for(_Doc d:collection)
			for(int i=0; i<d.m_topics.length; i++)
				d.m_topics[i] = Math.log(d.m_topics[i]);
		
		for(int i=0;i<number_of_topics;i++)
			for(int v=0; v<this.vocabulary_size; v++)
				topic_term_probabilty[i][v] = Math.log(topic_term_probabilty[i][v]);
	}
	
	// Construct the emission probabilities for sentences under different topics in a particular document.
	void ComputeEmissionProbsForDoc(_Doc d) {
		for(int i=0; i<d.getSenetenceSize(); i++) {
			_SparseFeature[] stn = d.getSentences(i);
			Arrays.fill(emission[i], 0);
			for(int k=0; k<this.number_of_topics; k++) {
				for(_SparseFeature w:stn) {
					emission[i][k] += w.getValue() * topic_term_probabilty[k][w.getIndex()];//all in log-space
				}
			}
		}
	}
	
	@Override
	public void calculate_E_step(_Doc d) {
		//Step 0: initiate sufficient statistic collector
		initStatInDoc(d);
		
		//Step 1: pre-compute emission probability
		ComputeEmissionProbsForDoc(d);
		
		//Step 2: use forword/backword algorithm to compute the posterior
		loglik += m_hmm.ForwardBackward(d, emission);
		
		//Step 3: collection expectations from the posterior distribution
		m_hmm.collectExpectations(p_dwzpsi);//expectations will be in the original space	
		accTheta(d);
		
		if (m_collectCorpusStats) {
			accEpsilonStat(d);
			accPhiStat(d);
		}
	}
	
	public int[] get_MAP_topic_assignment(_Doc d) {
		int path [] = new int [d.getSenetenceSize()];
		m_hmm.BackTrackBestPath(d, emission, path);
		return path;
	}
	
	//accumulate sufficient statistics for epsilon, according to Eq(15) in HTMM note
	void accEpsilonStat(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<this.number_of_topics; i++) 
				this.lot += this.p_dwzpsi[t][i];
			this.total ++;
		}
	}
	
	void accPhiStat(_Doc d) {
		for(int t=0; t<d.getSenetenceSize(); t++) {
			for(_SparseFeature s:d.getSentences(t)) {
				int wid = s.getIndex();
				double v = s.getValue();//frequency
				for(int i=0; i<this.number_of_topics; i++) {
					this.word_topic_sstat[i][wid] += v * (this.p_dwzpsi[t][i] + this.p_dwzpsi[t][i+this.number_of_topics]);
				}
			}
		}
	}
	
	void accTheta(_Doc d) {
		for(int t=0; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<this.number_of_topics; i++) 
				d.m_sstat[i] += this.p_dwzpsi[t][i];
		}
	}
	
	//accumulate sufficient statistics for theta, according to Eq(21) in HTMM note
	@Override
	protected void estThetaInDoc(_Doc d) {
		double sum = Math.log(Utils.sumOfArray(d.m_sstat));//prior has already been incorporated when initialize m_sstat
		for(int i=0; i<this.number_of_topics; i++) 
			d.m_topics[i] = Math.log(d.m_sstat[i]) - sum;//ensure in log-space
	}
	
	@Override
	public void calculate_M_step() {
		this.epsilon = this.lot/this.total; // to make the code structure concise and consistent, keep epsilon in real space!!
		
		for(int i=0; i<this.number_of_topics; i++) {
			double sum = Math.log(Utils.sumOfArray(word_topic_sstat[i]));
			for(int v=0; v<this.vocabulary_size; v++)
				topic_term_probabilty[i][v] = Math.log(word_topic_sstat[i][v]) - sum;
		}
		
		for(_Doc d:m_trainSet)
			estThetaInDoc(d);
		
		m_hmm.setEpsilon(this.epsilon);
	}
	
	@Override
	protected double calculate_log_likelihood() {
		//prior from Dirichlet distributions
		double logLikelihood = 0;
		for(_Doc d:m_trainSet) {
			for(int i=0; i<this.number_of_topics; i++) {
				logLikelihood += (d_alpha-1)*d.m_topics[i];
			}
		}
		
		for(int i=0; i<this.number_of_topics; i++) {
			for(int v=0; v<this.vocabulary_size; v++) {
				logLikelihood += (d_beta-1)*topic_term_probabilty[i][v];
			}
		}
		
		return this.loglik + logLikelihood;
	}
		
	protected void init() {
		this.loglik = 0;
		this.total = 0;
		this.lot = 0.0;// sufficient statistics for epsilon	
		
		super.init();
	}
	
	@Override
	protected void initTestDoc(_Doc d) {
		super.initTestDoc(d);
		for(int i=0; i<d.m_topics.length; i++)//convert to log-space
			d.m_topics[i] = Math.log(d.m_topics[i]);
	}
	
	//for HTMM, this function will be only called in testing phase to avoid duplicated computation
	@Override
	public double calculate_log_likelihood(_Doc d) {//it is very expensive to re-compute this
		//Step 1: pre-compute emission probability
		ComputeEmissionProbsForDoc(d);		
		
		double logLikelihood = 0;
		for(int i=0; i<this.number_of_topics; i++) 
			logLikelihood += (d_alpha-1)*d.m_topics[i];
		return logLikelihood + m_hmm.ForwardBackward(d, emission);
	}
}
