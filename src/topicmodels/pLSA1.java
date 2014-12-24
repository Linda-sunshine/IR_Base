package topicmodels;


/**
 * @author Md. Mustafizur Rahman (mr4xb@virginia.edu)
 * Probabilistic Latent Semantic Analysis Topic Modeling 
 */

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;

import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures._RankItem;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import Analyzer.jsonAnalyzer;


public class pLSA1 extends TopicModel {

	private int number_of_docs;
	private int number_of_topics;
	protected double[] background_probability;
	
	double document_topic_probabilty [][]; /* p(z|d) */
	double topic_term_probabilty [][]; /* p(w|z) */
	double document_word_topic_probabilty[][][]; /*p(z|d,w) */
	double document_word_background_probabilty[][];
	
	
	public pLSA1(int number_of_docs, int number_of_topics, int number_of_iteration, int vocabulary_size, double lambda, double beta, double back_ground [], _Corpus c)
	{	
		
		super(vocabulary_size, number_of_iteration, lambda, beta,c);
		
		this.number_of_docs = number_of_docs;
		this.number_of_topics = number_of_topics;
		this.background_probability = back_ground;
		
	    document_topic_probabilty = new double [this.number_of_docs][this.number_of_topics];
		topic_term_probabilty = new double [this.number_of_topics][this.vocabulary_size];
		document_word_topic_probabilty = new double [this.number_of_docs][this.vocabulary_size][this.number_of_topics];
		document_word_background_probabilty = new double [this.number_of_docs][this.vocabulary_size];
		
	}

	
	public void initialize_probability()
	{
		//initialize document-topic matrix
		// random is better than uniform
		for(int i=0;i<this.number_of_docs;i++)
		{
			this.document_topic_probabilty[i] = new double [this.number_of_topics];
			Utils.randomize(this.document_topic_probabilty[i], beta);
		}
		
		// initialize term topic matrix
		// uniform is better than random
		for(int i=0;i<number_of_topics;i++)
		{ 
			this.topic_term_probabilty[i] = new double [this.vocabulary_size];
			Arrays.fill(this.topic_term_probabilty[i], 1.0/this.vocabulary_size);
		}
		
		
	}
	

	public void calculate_E_step()
	{
		
		for (int i = 0; i < this.number_of_docs ; i++)
		{
			_Doc d = this.m_corpus.getCollection().get(i);
		//-----------------other topics----------- 
			for(_SparseFeature fv:d.getSparse()) 
			{
				int j = fv.getIndex(); // jth word in doc i
				double sum = 0.0;
				for(int k=0;k<this.number_of_topics;k++)
				{
					sum = sum + document_topic_probabilty[i][k] * topic_term_probabilty[k][j];
				}
				double denumerator = sum;
				
				for(int k=0;k<this.number_of_topics;k++)
				{
					double numerator = document_topic_probabilty[i][k] * topic_term_probabilty[k][j];
					document_word_topic_probabilty[i][j][k] = (double ) numerator / denumerator;
					document_word_topic_probabilty[i][j][k] = fv.getValue() * document_word_topic_probabilty[i][j][k]; 
					
				}
			
				//-----------------background topics----------- 
				double numerator = this.lambda * background_probability[j];
				denumerator = numerator + (1 - this.lambda) * sum;
				document_word_background_probabilty[i][j] = (double) numerator / denumerator;
			
			}
			}
	}
		
	
	
	@Override
	public void calculate_M_step()
	{
		
		// update document-topic matrix -------------
		for(int i=0;i<this.number_of_docs;i++)
		{
	
			double total_denumerator = 0.0;
			for(int j=0;j<this.number_of_topics;j++)
			{
				double numerator = 0.0;
				for(int k=0; k<this.vocabulary_size; k++)
				{
					numerator = numerator + (1 - document_word_background_probabilty[i][k])*document_word_topic_probabilty[i][k][j];
				}
				document_topic_probabilty [i][j] = numerator;
				total_denumerator = total_denumerator + numerator;
				
			}
			
	    for(int j=0;j<this.number_of_topics;j++)
			{
				document_topic_probabilty [i][j] = (double) document_topic_probabilty [i][j] / total_denumerator;
			}
		}
		
		// update topic-term matrix -------------
		for(int k=0;k<this.number_of_topics;k++)
		{
			double total_denumerator = 0.0;
			for(int i=0;i<this.vocabulary_size;i++)
			{
				double numerator = 0.0;
				for(int j=0;j<this.number_of_docs;j++)
				{
					numerator = numerator + (1 - document_word_background_probabilty[j][i])*document_word_topic_probabilty[j][i][k]; 
				}
				
				topic_term_probabilty[k][i] = numerator;
				total_denumerator = total_denumerator + numerator;
			}
			
			for(int i=0;i<this.vocabulary_size;i++)
			{
				topic_term_probabilty[k][i] = (double) topic_term_probabilty[k][i] / total_denumerator;
			}
		}
		
		
	}
	
	
	/*likelihod calculation */
	/* M is number of doc
	 * N is number of word in corpus
	 */
	/* p(w,d) = sum_1_M sum_1_N count(d_i, w_j) * log[ lambda*p(w|theta_B) + [lambda * sum_1_k (p(w|z) * p(z|d)) */ 
	@Override
	public double calculate_log_likelihood()
	{
		//print(topic_term_probabilty, number_of_topics,vocabulary_size);
		//print(document_topic_probabilty, number_of_docs, number_of_topics);
		
		double likelihood = 0.0;
		for (int i = 0; i < this.number_of_docs ; i++)
		{
			_Doc d = this.m_corpus.getCollection().get(i);
			for(_SparseFeature fv:d.getSparse()) {
				int j = fv.getIndex();	
				double sum = 0.0;
				for(int k=0;k<this.number_of_topics;k++)
				{
					sum = sum + (document_topic_probabilty[i][k] * topic_term_probabilty[k][j]);
				}
				double parameter = sum * (1 - lambda) + this.background_probability[j]*lambda;
				likelihood = likelihood + (fv.getValue() * Math.log(parameter));
			}
		}
			return likelihood;
	}
	
	
	public void print(double [][] array, int m, int n)
	{
		for(int i=0;i<m;i++)
		{
			for(int j=0;j<n;j++)
			{
				System.out.print("array["+i+"]["+j+"]=" + array[i][j] + " ");
			}
			System.out.print("\n");
			
		}
	}
	
	@Override
	public void printTopWords(int k) {
		//we only have one topic to show
		for(int i=0; i<topic_term_probabilty.length; i++)
		{
			MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
			for(int j = 0; j < vocabulary_size; j++)
			{
				fVector.add(new _RankItem(m_corpus.getFeature(j), topic_term_probabilty[i][j]));
			}
			for(_RankItem it:fVector)
				System.out.format("%s(%.3f)\t", it.m_name, it.m_value);
			System.out.println();
		}
		
		
	}
	
	
	public void get_topic_probability()
	{
		initialize_probability();
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do
		{
			calculate_E_step();
			calculate_M_step();
			
			current = calculate_log_likelihood();
			delta = Math.abs((current - last)/last);
			last = current;
			System.out.format("Likelihood %.9f at step %s converge to %f...\n", last, i, delta);
			
			i++;
		} while (delta>1e-4 && i<this.number_of_iteration);
		
		double perplexity = Math.exp(-current/m_corpus.getCorpusTotalLenght());
		System.out.format("Likelihood converges to %.4f after %d steps...\n", last, i);
		
	}
	
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException
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
		
		
		/* Variable related to PLSA */
		
		int number_of_topics = 3;
		double beta = 1e-3;
		double lambda = 0.9;
		int topK = 10;
	    int number_of_iteration = 50;
	    int number_of_docs = c.getSize();
		int vocabulary_size = analyzer.getFeatureSize();
		
		pLSA1 model = new pLSA1(number_of_docs, number_of_topics, number_of_iteration, vocabulary_size, lambda,  beta, analyzer.get_back_ground_probabilty(),c);
		model.get_topic_probability();
		model.printTopWords(topK);
		
		
		
	}

	@Override
	public double[] get_topic_probability(_Doc d) {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public void calculate_E_step(_Doc d) {
		// TODO Auto-generated method stub
		
	}


	@Override
	protected double calculate_log_likelihood(_Doc d) {
		// TODO Auto-generated method stub
		return 0;
	}
	

}
