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

public class HTMM extends TopicModel{

	private double d_alpha; //the symmetric dirichlet prior
	private double d_beta; // the symmetric dirichlet prior
	private int number_of_topics;
	private double epsilon;   // estimated epsilon
	private double[][][] p_dwzpsi;  // The state probabilities that is Pr(z,psi | d,w). 
	double[][] topic_term_probabilty ; /* p(w|z) phi */ 
	private int number_of_docs;
	private double loglik;
	private int constant;
	
	private double Czw [][];
	
	HTMM(int number_of_topics,double d_alpha, double d_beta, double beta, int number_of_iteration, _Corpus c) 
	{
		super(number_of_iteration, beta, c);
		loglik = 0.0;
		constant = 2;
		this.d_alpha = d_alpha;
		this.d_beta = d_beta;
		Random r = new Random();
		this.epsilon = beta + r.nextDouble();
		this.number_of_topics = number_of_topics;
		topic_term_probabilty = new double[this.number_of_topics][this.vocabulary_size];
		this.number_of_docs = c.getSize();
		p_dwzpsi = new double[number_of_docs][][];
		for(int d=0; d<number_of_docs; d++)
		{
			_Doc dc = c.getCollection().get(d);
			int number_of_sentecne_in_doc_d = dc.getTotalSenetences();
			p_dwzpsi[dc.getID()] = new double [number_of_sentecne_in_doc_d][];
			for(int w=0; w<number_of_sentecne_in_doc_d;w++)
			{
				p_dwzpsi[dc.getID()][w] = new double [constant*this.number_of_topics];
			}
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
			//Arrays.fill(this.topic_term_probabilty[i], 1.0/this.vocabulary_size);
		
	}
	
	public void IncorporatePriorsIntoLikelihood()
	{
		// The prior on theta, assuming a symmetric Dirichlet distirubiton
		  for (int d = 0; d < this.number_of_docs; d++) {
		    for (int z = 0; z < this.number_of_topics; z++) {
		      this.loglik += (this.d_alpha-1)*Math.log(this.m_corpus.getCollection().get(d).m_topics[z]);
		    }
		  }
		  
		// The prior on phi, assuming a symmetric Dirichlet distirubiton
		  for (int z = 0; z < this.number_of_topics; z++) {
		    for (int w = 0; w < this.vocabulary_size; w++) {
		      this.loglik += (this.d_beta-1)*Math.log(topic_term_probabilty[z][w]);
		    }
		  }  
	}
	
	public void Normalize(double norm, double [] v) {
		  double invnorm = 1.0 / norm;
		  for (int i = 0; i < v.length; i++) {
		    v[i] *= invnorm;
		  }
		}
	
	
	// This method is used to compute local probabilities for a word or for a
	// sentece.
	public double ComputeLocalProbsForItem(_Doc d, _SparseFeature[] sentence, double local [])
	{
		double likelihood = 0.0;
		for (int z = 0; z < this.number_of_topics; z++) {
		    local[z] = 1;
		  }
		
		Normalize(this.number_of_topics, local);
		likelihood += Math.log(this.number_of_topics);
		for (int i = 0; i < sentence.length; i++) {
		    double norm = 0;
		    int word = sentence[i].getIndex();
		    for (int z = 0; z < this.number_of_topics; z++) {
		      local[z] *= topic_term_probabilty[z][word];
		      norm += local[z];
		    }
		    Normalize(norm, local);
		    likelihood += Math.log(norm);
		}
		
		return likelihood;
	}
	
	// Computes the local probabilites for all the sentences of a particular
	// document.
	public double ComputeLocalProbsForDoc(_Doc d, double local [][])
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
		
		/*System.out.print("\nlocal\n");
		for(int i=0; i< d.getTotalSenetences(); i++)
		{
			for(int j=0; j<this.number_of_topics; j++)
			{
				System.out.print(local[i][j]+" ");
				
			}
			System.out.print("\n");
		}*/
		
		
		double init_probs [] = new double [constant*this.number_of_topics];
		for(int i=0; i<this.number_of_topics; i++)
		{
			init_probs[i] = d.m_topics[i];
		    init_probs[i+this.number_of_topics] = 0;  // Document must begin with a topic transition.
		}
		
		FastRestrictedHMM f = new FastRestrictedHMM();
		//System.out.println("Doc ID:"+ d.getID());
		double ll = f.ForwardBackward(this.epsilon, d.m_topics, local, init_probs, this.p_dwzpsi[d.getID()]);
		
		this.loglik += ll+loca_ll;
	}

	
	// We count only the number of times when a new topic was drawn according to
	// theta, i.e. when psi=1 (this includes the beginning of a document).
	public double [] CountTopicsInDoc(_Doc d) 
	{
	  double[] Cdz = new double [this.number_of_topics];
	  for (int z = 0; z < this.number_of_topics; z++) {
	    Cdz[z] = 0;
	  }
	  for (int i = 0; i < d.getTotalSenetences() ; i++) {
	    for (int z = 0; z < this.number_of_topics; z++) {
	      // only psi=1
	      Cdz[z] += p_dwzpsi[d.getID()][i][z];
	    }
	  }
	  
	  return Cdz;
	}

	
	// Finds the MAP estimator for theta_d
	void FindSingleTheta(_Doc d) {
	  double norm = 0;
	  double Cdz [] = CountTopicsInDoc(d);
	  for (int z = 0; z < this.number_of_topics; z++) {
	    d.m_topics[z] = Cdz[z] + d_alpha - 1;
	    norm += d.m_topics[z];
	  }
	  Normalize(norm, d.m_topics);
	}
	
	// Finds the theta for all documents in the train set.
	void FindTheta() {
	  for (int d = 0; d < this.number_of_docs; d++) {
	    FindSingleTheta(this.m_corpus.getCollection().get(d));
	  }
	}
	
	
	// Counts how many times the pair (z,w) for a certain topic z and a certain
	// word w appears in a certain sentence,
	void CountTopicWordInSentence(_SparseFeature[] sen, double [] topic_probs) 
	{
	  // Iterate over all the words in a sentence
	  for (int n = 0; n < sen.length; n++) {
	    int w = sen[n].getIndex();
	    for (int z = 0; z < this.number_of_topics; z++) {
	      // both psi=1 and psi=0
	      Czw[z][w] += topic_probs[z]+topic_probs[z+this.number_of_topics];
	    }
	  }
	}

	public void CountTopicWord() {
	  // iterate over all sentences in corpus
	  for (int d = 0; d < this.number_of_docs; d++) {
		  _Doc dc = this.m_corpus.getCollection().get(d);
	    for (int i = 0; i < dc.getTotalSenetences() ; i++) {
	      CountTopicWordInSentence(dc.getSentences(i), this.p_dwzpsi[dc.getID()][i]);
	    }
	  }
	}
	
	// Finds the MAP estimator for phi
	public void FindPhi() 
	{
		this.Czw = new double [this.number_of_topics][this.vocabulary_size];
		for(int i=0; i<this.number_of_topics; i++)
			Arrays.fill(Czw[i], 0.0);
	  	CountTopicWord();   // Czw is allocated and initialized to 0
		  for (int z = 0; z < this.number_of_topics; z++) {
		    double norm = 0;
		    for (int w = 0; w < this.vocabulary_size; w++) {
		      topic_term_probabilty[z][w] = Czw[z][w] + this.d_beta - 1;
		      norm += topic_term_probabilty[z][w];
		    }
		    Normalize(norm, topic_term_probabilty[z]);
		  }
	}
	
	
	// Finds the MAP estimator for epsilon.
	public void FindEpsilon() {
	  int total = 0;
	  double lot = 0;
	  for (int d = 0; d < this.number_of_docs; d++) {
	    //  we start counting from the second item in the document
		  _Doc dc = this.m_corpus.getCollection().get(d);
	    for (int i = 1; i < dc.getTotalSenetences(); i++) {
	      for (int z = 0; z < this.number_of_topics; z++) {
	        // only psi=1
	        lot += p_dwzpsi[dc.getID()][i][z];
	      }
	    }
	    total += dc.getTotalSenetences() - 1;      // Psi is always 1 for the first
	                                      // word/sentence
	  }
	  this.epsilon = lot/total;
	}
	
	@Override
	public void calculate_M_step() {
		
		FindEpsilon();
		FindPhi();
		FindTheta();
	}

	
	@Override
	public void EM(double converge)
	{	
		initialize_probability();
		
		
		int  i = 0;
		do
		{
			this.loglik = 0;
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
		
		String featureLocation = "./data/Features/selected_fv.txt";
		String finalLocation = "./data/Features/selected_fv_stat.txt";

		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(finalLocation); // Get the collection of all the documents.
		
		
		int number_of_topics = 10;
		double alpha = 1.5;
		double beta = 1.01;
		int number_of_iteration = 2;
		
		HTMM htmm = new HTMM(number_of_topics,alpha,beta,.00001,number_of_iteration ,c);
		htmm.EM(0.0);
		htmm.printTopWords(10);
	}
	
}
