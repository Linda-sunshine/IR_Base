package topicmodels;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import Analyzer.jsonAnalyzer;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import markovmodel.FastRestrictedHMM;

public class HTMM extends TopicModel {

	private double d_alpha; //the symmetric dirichlet prior
	private double d_beta; // the symmetric dirichlet prior
	private int number_of_topics;
	private double epsilon;   // estimated epsilon
	private double[][] p_dwzpsi;  // The state probabilities that is Pr(z,psi | d,w). 
	double[][] topic_term_probabilty ; /* p(w|z) phi */ 
	private int number_of_docs;
	private double loglik;
	final int constant = 2;
	private double[][] word_topic_sstat; // Czw as in HTMM
	// For Cdz we use here _Doc.m_sstat
	
	private int total; // used for epsilion
	private double lot; // used for epsilion
	
	public HTMM(int number_of_topics,double d_alpha, double d_beta, double beta, int number_of_iteration, _Corpus c) 
	{
		super(number_of_iteration, beta, c);
		loglik = 0.0;
		this.d_alpha = d_alpha;
		this.d_beta = d_beta;
		Random r = new Random();
		this.epsilon = beta + r.nextDouble();
		this.number_of_topics = number_of_topics;
		this.number_of_docs = c.getSize();
		
		word_topic_sstat = new double [this.number_of_topics][this.vocabulary_size];
		topic_term_probabilty = new double[this.number_of_topics][this.vocabulary_size];
		p_dwzpsi = new double[c.getLargestSentenceLenght()][]; // largest number of sentence of a Doc in corpus
		for(int s=0; s<c.getLargestSentenceLenght(); s++)
		{
			p_dwzpsi[s] = new double [constant * this.number_of_topics];
			
		}
		
	}

	@Override
	protected void initialize_probability() {
		
		// theta is doc_topic probability
		for(_Doc d:m_corpus.getCollection())
			d.setTopics(number_of_topics, beta);//allocate memory and randomize it
		
		// phi is topic_term probability
		for(int i=0;i<number_of_topics;i++)
			Utils.randomize(this.topic_term_probabilty[i], beta);
	}
	
	
	// This method is used to compute local probabilities for a word or for a
	// sentence.
	private double ComputeLocalProbsForItem(_Doc d, _SparseFeature[] sentence, double local [])
	{
		double likelihood = 0.0;
		Arrays.fill(local,1.0/this.number_of_topics);
		likelihood += Math.log(this.number_of_topics);
		for (int i = 0; i < sentence.length; i++) {
		    double norm = 0;
		    int word = sentence[i].getIndex();
		    double frequency = sentence[i].getValue();
		    for(double count = 0; count < frequency; count = count + 1.0) {
			    for (int z = 0; z < this.number_of_topics; z++) {
			      local[z] *= topic_term_probabilty[z][word];//Hongning: why do we multiply word count (sentence[i].getValue()) as well?
			      norm += local[z];
			    }
		    }
		    Utils.scaleArray(local, 1.0/norm);//Hongning: please use the shared implementation
		    likelihood += Math.log(norm);
		}
		
		return likelihood;
	}
	
	// Computes the local probabilities for all the sentences of a particular
	// document.
	private double ComputeLocalProbsForDoc(_Doc d, double local [][])
	{
		double likelihood = 0.0;
		
		for(int i=0; i<d.getTotalSenetences(); i++)
		{
			likelihood += ComputeLocalProbsForItem(d, d.getSentences(i), local[i]);
		}
		return likelihood;
	}
	
	@Override
	public void calculate_E_step(_Doc d) {
		
		double local [][] = new double [d.getTotalSenetences()][this.number_of_topics];
		double loca_ll = 0.0; // local likelihood
		loca_ll = ComputeLocalProbsForDoc(d, local);
	
		double init_probs [] = new double [constant*this.number_of_topics];
		for(int i=0; i<this.number_of_topics; i++)
		{
			init_probs[i] = d.m_topics[i];
		    init_probs[i+this.number_of_topics] = 0;  // Document must begin with a topic transition.
		}
		
		// For each doc we use the same this.p_dwzpsi[s] so we clear it 0.0 before use
		for(int s=0; s<this.m_corpus.getLargestSentenceLenght(); s++)
		{
			Arrays.fill(this.p_dwzpsi[s], 0.0);
		}
		
		FastRestrictedHMM f = new FastRestrictedHMM(); 
		double ll = f.ForwardBackward(this.epsilon, d.m_topics, local, init_probs, this.p_dwzpsi);
		
		//-------fractional Epsilon as in HTMM paper------------------
		FindSingleEpsilon(d);
		//-------fractional phi or beta_z_w as in HTMM paper-----------
		// For per_doc_phi calculation 
		// FindPhi(total phi) will be called later from M_step
		CountTopicWord(d);   
		//-------fractional theta_d_z as in HTMM paper-----------------
		FindSingleTheta(d);
		this.loglik += ll + loca_ll;
	}

	
	// We count only the number of times when a new topic was drawn according to
	// theta, i.e. when psi=1 (this includes the beginning of a document).
	private void CountTopicsInDoc(_Doc d) 
	{
	  for (int i = 0; i < d.getTotalSenetences() ; i++) {
	    for (int z = 0; z < this.number_of_topics; z++) {
	      // only psi=1
	      d.m_sstat[z] += this.p_dwzpsi[i][z]; 
	    }
	  }
	  
	}
	
	// Finds the MAP estimator for theta_d
	private void FindSingleTheta(_Doc d) {
	  double norm = 0;
	  CountTopicsInDoc(d); 
	  for (int z = 0; z < this.number_of_topics; z++) {
	    d.m_topics[z] = d.m_sstat[z] + d_alpha - 1;
	    d.m_sstat[z] = 0.0; // cleared for next iteration
	    norm += d.m_topics[z];
	  }
	  Utils.scaleArray(d.m_topics, 1.0/norm);
	  
	}

	// Counts how many times the pair (z,w) for a certain topic z and a certain
	// word w appears in a certain sentence,
	private void CountTopicWordInSentence(_SparseFeature[] sen, double [] topic_probs) 
	{
	  // Iterate over all the words in a sentence
	  for (int n = 0; n < sen.length; n++) {
	    int w = sen[n].getIndex();
	    double frequency = sen[n].getValue();
	    for(double count = 0.0; count <frequency; count = count + 1.0){
		    for (int z = 0; z < this.number_of_topics; z++) {
		      // both psi=1 and psi=0
		      word_topic_sstat[z][w] += topic_probs[z]+topic_probs[z+this.number_of_topics];
		    }
	    }
	  }
	}

	private void CountTopicWord(_Doc d) {
	  for (int i = 0; i < d.getTotalSenetences() ; i++) {
	      CountTopicWordInSentence(d.getSentences(i), this.p_dwzpsi[i]);
	    }
	  }
	
	
	// Finds the MAP estimator for phi
	private void FindPhi() 
	{
		  for (int z = 0; z < this.number_of_topics; z++) {
		    double norm = Utils.sumOfArray(word_topic_sstat[z]) + this.vocabulary_size*(this.d_beta - 1);
			  for (int w = 0; w < this.vocabulary_size; w++) {
		      topic_term_probabilty[z][w] = word_topic_sstat[z][w] + this.d_beta - 1; 
		    }
		    Utils.scaleArray(topic_term_probabilty[z], 1.0/norm); 
		}
	}
	
	// Finds epsilon for single doc
	private void FindSingleEpsilon(_Doc d)
	{
		 //we start counting from the second item in the document
		for (int i = 1; i < d.getTotalSenetences(); i++) {
	      for (int z = 0; z < this.number_of_topics; z++) {
	        // only psi=1
	        this.lot += this.p_dwzpsi[i][z];
	      }
	    }
	    this.total += d.getTotalSenetences() - 1; // Psi is always 1 for the first  word/sentence
	}
	
	// Finds the MAP estimator for epsilon.
	private void FindEpsilon() {
		
		this.epsilon = (double) this.lot/this.total;
	}
	
	@Override
	public void calculate_M_step() {
		
		FindEpsilon();
		FindPhi();
		// word_topic_sstat is allocated and initialized to 0 for next iteration
		for(int z=0; z<this.number_of_topics; z++)
			 Arrays.fill(word_topic_sstat[z], 0.0);
		
	}

	protected void IncorporatePriorsIntoLikelihood()
	{
		// The prior on theta, assuming a symmetric Dirichlet distirubiton
		  for (int d = 0; d < this.number_of_docs; d++) {
			  _Doc doc = m_corpus.getCollection().get(d);
		    for (int z = 0; z < this.number_of_topics; z++) {
		      this.loglik += (this.d_alpha-1)*Math.log(doc.m_topics[z]);
		    }
		  }
		  
		// The prior on phi, assuming a symmetric Dirichlet distirubiton
		  for (int z = 0; z < this.number_of_topics; z++) {
		    for (int w = 0; w < this.vocabulary_size; w++) {
		      this.loglik += (this.d_beta-1)*Math.log(topic_term_probabilty[z][w]);
		    }
		  }  
	}
		
	
	@Override
	public void EM(double converge)
	{	
		initialize_probability();
		
		int  i = 0;
		do
		{
			this.loglik = 0;
			this.total = 0; // used for epsilion
			this.lot = 0.0;// used for epsilion
			for(_Doc d:m_corpus.getCollection())
				calculate_E_step(d);
			
			IncorporatePriorsIntoLikelihood();
			calculate_M_step();
			
			System.out.format("Likelihood %.3f at step %s\n", this.loglik, i);
			i++;
			
		} while (i<this.number_of_iteration);
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
		
		return null;
	}
	
	public static void main(String[] args) throws IOException
	{
		
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int lengthThreshold = 5; //Document length threshold
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/test";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String stnModel = "./data/Model/en-sent.bin"; //Sentence detection model.
		
		String featureLocation = "./data/Features/selected_fv.txt";
		String finalLocation = "./data/Features/selected_fv_stat.txt";

		System.out.println("Creating feature vectors, wait...");
		
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold, stnModel);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(finalLocation); // Get the collection of all the documents.
		
		
		int number_of_topics = 4;
		double alpha = (1 + 50.0 / number_of_topics ); 
		double beta = 1.01;
		int number_of_iteration = 50;
		
		HTMM htmm = new HTMM(number_of_topics,alpha,beta,.00001,number_of_iteration ,c);
		htmm.EM(0.0);
		htmm.printTopWords(10);
	}
	
}
