package topicModel;


/**
 * @author Md. Mustafizur Rahman (mr4xb@virginia.edu)
 * Probabilistic Latent Semantic Analysis Topic Modeling 
 */

import java.io.*;
import java.util.ArrayList;
import java.util.Random;

import opennlp.tools.util.InvalidFormatException;
import Analyzer.jsonAnalyzer;
import structures._Doc;


public class plsa extends TopicModel{

	int number_of_docs;
	int number_of_topics;
	
	double document_term_count [][];
	
	/* p(z|d) */
	double document_topic_probabilty [][];
	
	/* p(w|z) */
	double topic_term_probabilty [][];
	
	/*p(z|d,w) */
	
	double document_word_topic_probabilty[][][];
	double document_word_background_probabilty[][];
	double likelihood[];
	
	plsa()
	{
		
		lambda = 0.9;
		vocabulary_size = 3;
		number_of_iteration = 100;
		number_of_docs = 2;
		number_of_topics = 2;
		
		//initialize background probability
		background_probability = new double [vocabulary_size];
		background_probability[0] = 0.5;
		background_probability[1] = 0.3;
		background_probability[2] = 0.2;
		
		
		//initialize document_term_count
		document_term_count = new double [this.number_of_docs][this.vocabulary_size];
				
		document_term_count[0][0] = 7;
		document_term_count[0][1] = 5;
		document_term_count[0][2] = 6;
				
		document_term_count[1][0] = 8;
		document_term_count[1][1] = 7;
		document_term_count[1][2] = 5;
		
		document_topic_probabilty = new double [this.number_of_docs][this.number_of_topics];
		topic_term_probabilty = new double [this.number_of_topics][this.vocabulary_size];
		
		document_word_topic_probabilty = new double [this.number_of_docs][this.vocabulary_size][this.number_of_topics];
		document_word_background_probabilty = new double [this.number_of_docs][this.vocabulary_size];
				
		initialize_probability();
		EM();			
		
	}
	
	plsa(int number_of_docs, int number_of_topics, int number_of_iteration, int vocabulary_size, double lambda, double back_ground [], double document_term_count [][])
	{
		
		this.number_of_iteration = number_of_iteration;
		this.lambda = lambda;
		this.background_probability = back_ground;
		this.document_term_count = document_term_count;
		this.vocabulary_size = vocabulary_size;
		this.number_of_docs = number_of_docs;
		this.number_of_topics = number_of_topics;
		
		
		document_topic_probabilty = new double [this.number_of_docs][this.number_of_topics];
		topic_term_probabilty = new double [this.number_of_topics][this.vocabulary_size];
		
		document_word_topic_probabilty = new double [this.number_of_docs][this.vocabulary_size][this.number_of_topics];
		document_word_background_probabilty = new double [this.number_of_docs][this.vocabulary_size];
		
		
		likelihood = new double [this.number_of_iteration];
		
		initialize_probability();
		EM();		
		
	}
	

	
	public void initialize_probability()
	{
		
		//initialize document-topic matrics
		// random is better than uniform
		for(int i=0;i<this.number_of_docs;i++)
		{
			for(int j=0;j<number_of_topics;j++)
			this.document_topic_probabilty[i][j] = 1.0/this.number_of_topics;
		}
		
		// initialize term topic metrics
		// uniform is better than random
		

		for(int i=0;i<number_of_topics;i++)
		{ 
			for(int j=0;j<this.vocabulary_size;j++)
			{
		
				//this.term_topic_probabilty[i][j] = 0.5; // change later since 2 topic so 0.5
				this.topic_term_probabilty[i][j] = 1.0 / this.vocabulary_size;
				
			}
		}
		
		
	}
	
	public void calculate_E_step()
	{
		for(int i=0;i<this.number_of_docs;i++)
		{
			
			//-----------------other topics----------- 
		
			for(int j=0;j<this.vocabulary_size;j++)
			{
				double sum = 0.0;
				for(int k=0;k<this.number_of_topics;k++)
				{
					sum = sum + document_topic_probabilty[i][k] * topic_term_probabilty[k][j];
					
				}
			
			
				double denumerator = sum;
			
				for(int k=0;k<this.number_of_topics;k++)
				{
					double numerator = document_topic_probabilty[i][k] * topic_term_probabilty[k][j];
					document_word_topic_probabilty[i][j][k] = (double )numerator / denumerator;
					
				}
			
				//-----------------background topics----------- 
				double numerator = this.lambda * background_probability[j];
				denumerator = numerator + (1 - this.lambda) * sum;
				document_word_background_probabilty[i][j] = (double) numerator / denumerator;
			
			
			}
		}
		
	}
	
	
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
					numerator = numerator + document_term_count[i][k]*(1 - document_word_background_probabilty[i][k])*document_word_topic_probabilty[i][k][j];
				}
				
				document_topic_probabilty [i][j] = numerator;
				total_denumerator = total_denumerator + numerator;
				
			}
			
			for(int j=0;j<this.number_of_topics;j++)
			{
				document_topic_probabilty [i][j] = document_topic_probabilty [i][j] / total_denumerator;
			}
			
			
		}
		
		// update term-topic matrix -------------
		
		
		for(int k=0;k<this.number_of_topics;k++)
		{
			double total_denumerator = 0.0;
			for(int i=0;i<this.vocabulary_size;i++)
			{
				double numerator = 0.0;
				for(int j=0;j<this.number_of_docs;j++)
				{
					numerator = numerator + document_term_count[j][i]*(1 - document_word_background_probabilty[j][i])*document_word_topic_probabilty[j][i][k]; 
				}
				
				topic_term_probabilty[k][i] = numerator;
				total_denumerator = total_denumerator + numerator;
			}
			
			for(int i=0;i<this.vocabulary_size;i++)
			{
				topic_term_probabilty[k][i] = topic_term_probabilty[k][i] / total_denumerator;
			}
		}
		
		
	}
	
	
	/*likelihod calculation */
	/* M is number of doc
	 * N is number of word in corpus
	 */
	/* p(w,d) = sum_1_M sum_1_N count(d_i, w_j) * log[ lambda*p(w|theta_B) + [lambda * sum_1_k (p(w|z) * p(z|d)) */ 
	
	public double calculate_log_likelihood()
	{
		
		double likelihood = 0.0;
	
		for(int i=0;i<this.number_of_docs;i++)
		{
			for(int j=0;j<this.vocabulary_size;j++)
			{
				
				double sum = 0.0;
				for(int k=0;k<this.number_of_topics;k++)
				{
					sum = sum + (document_topic_probabilty[i][k] * topic_term_probabilty[k][j]);
				
				}
				double parameter = sum * (1 - lambda) + this.background_probability[j]*lambda;
				
				//double parameter = sum;
				
				likelihood = likelihood + (document_term_count[i][j] * Math.log(parameter));
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
	
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException
	{
		
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		int CVFold = 10; //k fold-cross validation
		String classifier = "NB"; //Which classifier to use.
		System.out.println("--------------------------------------------------------------------------------------");
		System.out.println("Parameters of this run:" + "\nClassNumber: " + classNumber + "\tNgram: " + Ngram + "\tFeatureValue: " + featureValue + "\tClassifier: " + classifier + "\nCross validation: " + CVFold);

		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/test";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String finalLocation = "./FinalFeatureStat.txt";
		String featureLocation = "./SelectedFeatures.txt";

		/*****Parameters in feature selection.*****/
		String providedCV = "";
		String featureSelection = "TF"; //Feature selection method.
		double startProb = 0.4; // Used in feature selection, the starting point of the features.
		double endProb = 1; // Used in feature selection, the ending point of the features.
		int DFthreshold = 10; // Filter the features with DFs smaller than this threshold.
		System.out.println("Feature Seleciton: " + featureSelection + "\tStarting probability: " + startProb + "\tEnding probability:" + endProb);
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer;
		analyzer = new jsonAnalyzer(tokenModel, classNumber, Ngram);
		
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		
		
		/* Variable related to PLSA */
		double back_ground_probability [] = analyzer.get_back_ground_probabilty();
		ArrayList<_Doc> docs = analyzer.returnCorpus(finalLocation).getCollection(); // Get the collection of all the documents.
		int N = docs.size();
		
		double document_term_count [][] = new double [N][];
		
		for(int i = 0; i < docs.size(); i++){
			_Doc temp = docs.get(i);
			document_term_count [i] = analyzer.get_term_frequency_in_doc(temp); 
		}
		
		int number_of_topics = 10;
		double lambda = 0.9;
	    int number_of_iteration = 500;
		int vocabulary_size = back_ground_probability.length;
		
		//plsa model = new plsa();
		
		plsa model = new plsa(N, number_of_topics, number_of_iteration, vocabulary_size, lambda, back_ground_probability, document_term_count);
		
		
	}
	

}
