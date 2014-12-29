package topicmodels;

import java.util.Arrays;
import java.util.Random;

import markovmodel.FastRestrictedHMM;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;

public class HTMM extends TopicModel {
	// Dirichlet prior for p(\theta|d)
	double d_alpha; 
	
	// HTMM parameter both in log space
	private double epsilon;   // estimated epsilon
	private double[][] topic_term_probabilty ; /* p(w|z) phi */ 
	
	// cache structure
	private double[][] p_dwzpsi;  // The state probabilities that is Pr(z,psi | d,w)
 	private double[][] emission;  // emission probability of p(s|z)
	private double[] pi; // initial state probability of p(z_i|d)
	
	// sufficient statistics for p(w|\phi_z) and p(\epsilon)
	private double[][] word_topic_sstat; // Czw as in HTMM
	private int total; // used for epsilion
	private double lot; // used for epsilion
	
	private double loglik;
	final int constant = 2;
	
	public HTMM(int number_of_topics,double d_alpha, double d_beta, double beta, int number_of_iteration, _Corpus c) 
	{
		super(number_of_iteration, beta, c);
		this.d_alpha = d_alpha;
		this.d_beta = d_beta;
		this.number_of_topics = number_of_topics;
		
		Random r = new Random();
		this.epsilon = beta + r.nextDouble(); //Hongning: how to make sure this is in the range of (0,1)
		
		word_topic_sstat = new double [this.number_of_topics][this.vocabulary_size];
		topic_term_probabilty = new double[this.number_of_topics][this.vocabulary_size];
		
		//cache in order to avoid frequently allocating new space
		p_dwzpsi = new double[c.getLargestSentenceSize()][constant * this.number_of_topics]; // max|S_d| * (2*K)
		emission = new double[p_dwzpsi.length][this.number_of_topics]; // max|S_d| * K
		pi = new double[constant * this.number_of_topics];
	}

	@Override
	protected void initialize_probability() {
		// theta is doc_topic probability
		for(_Doc d:m_corpus.getCollection())
			d.setTopics(number_of_topics, d_alpha);//allocate memory and randomize it
		
		// phi is topic_term probability
		for(int i=0;i<number_of_topics;i++)
			Utils.randomize(this.topic_term_probabilty[i], d_beta);
	}
	
	// Construct the emission probabilities for sentences under different topics in a particular document.
	private void ComputeLocalProbsForDoc(_Doc d)
	{
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
		ComputeLocalProbsForDoc(d);
	
		//Step 2: compute initial state probability 
		for(int i=0; i<this.number_of_topics; i++) {
			pi[i] = d.m_topics[i];//already in log-space 
		    pi[i+this.number_of_topics] = Double.NEGATIVE_INFINITY;  // Document must begin with a topic transition.
		}
		
		//Step 3: use forword/backword algorithm to compute the posterior
		FastRestrictedHMM f = new FastRestrictedHMM(); 
		loglik += f.ForwardBackward(d, epsilon, emission, pi);
		
		//Step 4: collection expectations from the posterior distribution
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
		Arrays.fill(d.m_sstat, d_alpha-1.0);
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
		this.epsilon = Math.log(this.lot / this.total);
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
		
		//sufficient statistics for phi
		for(int z=0; z<this.number_of_topics; z++)
			Arrays.fill(word_topic_sstat[z], d_beta-1.0);
	}
	
	@Override
	public void printTopWords(int k) {
		for(int i=0; i<topic_term_probabilty.length; i++) {
			MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
			for(int j = 0; j < vocabulary_size; j++)
				fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));
			for(_RankItem it:fVector)
				System.out.format("%s(%.3f)\t", it.m_name, it.m_value);
			System.out.println();
		}
	}
	
	@Override
	protected double calculate_log_likelihood(_Doc d) {
		return 0;
	}

	@Override
	public double[] get_topic_probability(_Doc d) {
		return d.m_topics;
	}
}
