package topicmodels;

import java.util.Arrays;
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
	
	// sufficient statistics for p(w|\phi_z) and p(\epsilon)
	int total; // used for epsilion
	double lot; // used for epsilion
	
	double loglik;
	final int constant = 2;

	public HTMM(int number_of_topics, double d_alpha, double d_beta, int number_of_iteration, _Corpus c) {
		super(number_of_topics, number_of_iteration, 0.5, d_beta, d_alpha, null, c);
		
		Random r = new Random();
		this.epsilon = r.nextDouble(); //Hongning: how to make sure this is in the range of (0,1)
		
		//cache in order to avoid frequently allocating new space
		p_dwzpsi = new double[c.getLargestSentenceSize()][constant * this.number_of_topics]; // max|S_d| * (2*K)
		emission = new double[p_dwzpsi.length][this.number_of_topics]; // max|S_d| * K
	}
	
	//convert them to log-space (pLSA is not running in log-space!!!)
	@Override
	protected void initialize_probability()
	{	
		super.initialize_probability();
		
		for(_Doc d:m_corpus.getCollection())
			for(int i=0; i<d.m_topics.length; i++)
				d.m_topics[i] = Math.log(d.m_topics[i]);
		
		for(int i=0;i<number_of_topics;i++)
			for(int v=0; v<this.vocabulary_size; v++)
				topic_term_probabilty[i][v] = Math.log(topic_term_probabilty[i][v]);
	}
	
	// Construct the emission probabilities for sentences under different topics in a particular document.
	private void ComputeEmissionProbsForDoc(_Doc d) {
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
		//Step 1: pre-compute emission probability
		ComputeEmissionProbsForDoc(d);
		
		//Step 2: use forword/backword algorithm to compute the posterior
		FastRestrictedHMM f = new FastRestrictedHMM(); 
		loglik += f.ForwardBackward(d, epsilon, emission);
		
		//Step 3: collection expectations from the posterior distribution
		f.collectExpectations(p_dwzpsi);//expectations will be in the original space		
		accEpsilonStat(d);
		accPhiStat(d);
		estThetaInDoc(d);		
	}
	
	//accumulate sufficient statistics for epsilon, according to Eq(15) in HTMM note
	private void accEpsilonStat(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<this.number_of_topics; i++) 
				this.lot += this.p_dwzpsi[t][i];
			this.total ++;
		}
	}
	
	private void accPhiStat(_Doc d) {
		for(int t=0; t<d.getSenetenceSize(); t++) {
			for(_SparseFeature w:d.getSentences(t)) {
				int wid = w.getIndex();
				double v = w.getValue();//frequency
				for(int i=0; i<this.number_of_topics; i++) {
					this.word_topic_sstat[i][wid] += v * (this.p_dwzpsi[t][i] + this.p_dwzpsi[t][i+this.number_of_topics]);
				}
			}
		}
	}
	
	//accumulate sufficient statistics for theta, according to Eq(21) in HTMM note
	private void estThetaInDoc(_Doc d) {
		for(int t=0; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<this.number_of_topics; i++) 
				d.m_sstat[i] += this.p_dwzpsi[t][i];
		}
		
		double sum = Math.log(Utils.sumOfArray(d.m_sstat));
		for(int i=0; i<this.number_of_topics; i++) 
			d.m_topics[i] = Math.log(d.m_sstat[i]) - sum;//ensure in log-space
	}
	
	@Override
	public void calculate_M_step() {
		this.epsilon = Math.log(this.lot/this.total);
		
		for(int i=0; i<this.number_of_topics; i++) {
			double sum = Math.log(Utils.sumOfArray(word_topic_sstat[i]));
			for(int v=0; v<this.vocabulary_size; v++)
				topic_term_probabilty[i][v] = Math.log(word_topic_sstat[i][v]+d_beta-1) - sum;
		}
	}
	
	@Override
	protected double calculate_log_likelihood() {
		double logLikelihood = 0;
		for(_Doc d:m_corpus.getCollection()) {
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
	public double calculate_log_likelihood(_Doc d) {//it is very expensive to re-compute this
		return 0;
	}
}
