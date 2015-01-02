package topicmodels;

import java.util.Arrays;
import java.util.Random;

import markovmodel.LRFastRestrictedHMM;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;

public class LRHTMM extends pLSA {
	// HTMM parameter both in log space
	double epsilon;   // estimated epsilon
	
	// cache structure
	double[][] p_dwzpsi;  // The state probabilities that is Pr(z,psi | d,w)
 	double[][] emission;  // emission probability of p(s|z)
	
	// sufficient statistics for p(w|\phi_z) and p(\epsilon)
	int total; // used for epsilion
	double lot; // used for epsilion
	
	//feature weight vector
	double [] sentence_feature_weight;
	//the learning rate for LR
    private double rate;
    //epsilon array for sentence;
    private double[] epsilon_array;

	
	double loglik;
	final int constant = 2;

	public LRHTMM(int number_of_topics, double d_alpha, double d_beta, int number_of_iteration, _Corpus c) {
		super(number_of_topics, number_of_iteration, 0.5, d_beta, d_alpha, null, c);
		
		Random r = new Random();
		this.epsilon = Math.log(r.nextDouble()); //Hongning: how to make sure this is in the range of (0,1)
		
		//cache in order to avoid frequently allocating new space
		p_dwzpsi = new double[c.getLargestSentenceSize()][constant * this.number_of_topics]; // max|S_d| * (2*K)
		emission = new double[p_dwzpsi.length][this.number_of_topics]; // max|S_d| * K
		
		//variable related to LR
		sentence_feature_weight = new double [2];
		this.rate = 0.0001;
		epsilon_array = new double[c.getLargestSentenceSize()];
	}
	
	//convert them to log-space (pLSA is not running in log-space!!!)
	@Override
	protected void initialize_probability()
	{	
		super.initialize_probability();
		
		for(_Doc d:m_corpus.getCollection()){
			d.setSentenceFeatureVector(); // sentence feature vector only in LRHTMM
			for(int i=0; i<d.m_topics.length; i++)
				d.m_topics[i] = Math.log(d.m_topics[i]);
		}
		
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
	
	private void calculate_epsilions(_Doc d)
	{
		//System.out.println("In E-step");
		//System.out.println("Feature 0:"+this.sentence_feature_weight[0] + " Feature 1:"+this.sentence_feature_weight[1]);
		for (int i=0; i<d.getSenetenceSize(); i++) {
			 this.epsilon_array[i] = Math.log(classify(d.m_sentence_features[i]));
			 //System.out.println("Doc:"+ d.getID() +" value:" + this.epsilon_array[i]);
		 }
	}
	
	@Override
	public void calculate_E_step(_Doc d) {
		//Step 1: pre-compute emission probability
		ComputeEmissionProbsForDoc(d);
		
		calculate_epsilions(d);
		//Step 2: use forword/backword algorithm to compute the posterior
		LRFastRestrictedHMM f = new LRFastRestrictedHMM(); 
		loglik += f.ForwardBackward(d, epsilon_array, emission);
		
		//Step 3: collection expectations from the posterior distribution
		f.collectExpectations(p_dwzpsi);//expectations will be in the original space		
		accEpsilonStat(d);  //accumulate both the epsilon and label for the LR and finallly call trainLR to update feature weight
		
		accPhiStat(d);
		estThetaInDoc(d);
		
		
	}
	
	public int[] get_MAP_topic_assignment(_Doc d) {
		int path [] = new int [d.getSenetenceSize()];
		LRFastRestrictedHMM v = new LRFastRestrictedHMM();
		v.BackTrackBestPath(d, epsilon_array, emission, path);
		return path;
	}
	
	
	// ----------------LR part Start-------------
	private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }
	
    private double classify(double[] x) {
        double logit = .0;
        for (int i=0; i<sentence_feature_weight.length;i++)  {
            logit += sentence_feature_weight[i] * x[i];
        }
        return sigmoid(logit);
    }
	
	public void train(int number_of_itearion, _Doc d, double [] sentence_label) {
        for (int n=0; n<number_of_itearion; n++) {
            double lik = 0.0;
            for (int i=0; i<d.getSenetenceSize(); i++) {
                double[] x = d.m_sentence_features[i]; // return the feature Vector of ith sentence
                double predicted = classify(x);
                double label = sentence_label[i]; // return the label of ith sentence
                for (int j=0; j<sentence_feature_weight.length; j++) {
                	sentence_feature_weight[j] = sentence_feature_weight[j] + rate * (label - predicted) * x[j];
                }
                // not necessary for learning
                //lik += label * Math.log(classify(x)) + (1-label) * Math.log(1- classify(x));
            }
           // System.out.println("iteration: " + n + " " + Arrays.toString(sentence_feature_weight) + " mle: " + lik);
        }
        
       //System.out.print("After M Step\n");
       //System.out.println("Feature 0:"+this.sentence_feature_weight[0] + " Feature 1:"+this.sentence_feature_weight[1]);
		
    }
	
	// ----------------LR part Fininsh-------------
	
	//accumulate sufficient statistics for epsilon, according to Eq(15) in HTMM note
	private void accEpsilonStat(_Doc d) {
		double label [] = new double[d.getSenetenceSize()]; // label from E-step of HTMM
		for(int t=1; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<this.number_of_topics; i++){ 
				label[t] += this.p_dwzpsi[t][i];
				this.lot += this.p_dwzpsi[t][i];
			}
			this.total ++;
			//System.out.println("label:"+ label[t]);
		}
		train(30,d,label); // train LR and update weight	
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
