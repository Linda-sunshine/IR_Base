package topicModel;

import java.io.*;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.Random;

import Analyzer.jsonAnalyzer;
import structures._Doc;
import structures._SparseFeature;

/**
 * @author Md. Mustafizur Rahman (mr4xb@virginia.edu)
 * two-topic Topic Modeling 
 */

public class twoTopic extends TopicModel{
	
	private double count [];
	private double topic_probability [];
	private double p_z_1 [];
	private double lambda;
	private double likelihood [];
	twoTopic()
	{
		
		lambda = 0.5;
		vocabulary_size = 4;
		number_of_iteration = 20;
		
		background_probability = new double [vocabulary_size];
		background_probability[0] = 0.5;
		background_probability[1] = 0.3;
		background_probability[2] = 0.1;
		background_probability[3] = 0.1;
		
		
		count = new double [vocabulary_size];
		count [0] = 4;
		count [1] = 2;
		count [2] = 4;
		count [3] = 2;
		
		p_z_1 = new double [vocabulary_size];
		initialize_probability();
		
	
		
		for(int i = 0; i<number_of_iteration; i++)
		{
			System.out.println("----------------------\n topic probaility");
			print(topic_probability);
			calculate_E_step();
			System.out.println("----------------------\n p_z_i");
			print(p_z_1);
			calculate_M_step();
			calculate_log_likelihood();
			
		}	
		
		
	}
	
	
	twoTopic(int number_of_iteration, int vocabulary_size, double lambda, double back_ground [], double count [])
	{
		
		this.number_of_iteration = number_of_iteration;
		this.lambda = lambda;
		this.background_probability = back_ground;
		this.count = count;
		this.vocabulary_size = vocabulary_size;
		
		
		p_z_1 = new double [this.vocabulary_size];
		likelihood = new double [this.number_of_iteration];
		initialize_probability();
		
		for(int i = 0; i<this.number_of_iteration; i++)
		{
			
		
			calculate_E_step();
			calculate_M_step();
			likelihood [i] = calculate_log_likelihood();
			
		}
		
		print(likelihood);
		
	}
	
	public void initialize_probability()
	{
		
    	topic_probability = new double [this.vocabulary_size];
		
		for(int i=0;i<this.vocabulary_size;i++)
		{
			 
			topic_probability [i] = (double) 1 / this.vocabulary_size;
			
		}
		
	}
	
	public void normalize_probability()
	{
		int n = this.vocabulary_size;
		double sum = 0.0;
		for(int i=0;i<n;i++)
		{
			sum = sum + background_probability [i];
		}
		
		for(int i=0;i<n;i++)
		{
			 
			background_probability [i] = (double) background_probability [i] / (double) sum;
			
		}
		
	}
	
	public void calculate_E_step()
	{
		for(int i=0;i<this.vocabulary_size;i++)
		{
			double numerator = lambda * background_probability[i];
			double denumerator = numerator + (1 - lambda)*topic_probability [i];
			p_z_1 [i] = numerator / denumerator;
		}
		
	}
	
	
	public void calculate_M_step()
	{
		
		double denumerator = 0.0;
		for(int j=0;j<this.vocabulary_size;j++)
		{
			denumerator = denumerator +  count[j] * ( 1 - p_z_1 [j]);
		}
		
		for(int i=0;i<this.vocabulary_size;i++)
		{
			double numerator = count[i] * ( 1 - p_z_1 [i]);
			
			topic_probability [i] = numerator / denumerator;
		}
	}
	
	public double calculate_log_likelihood()
	{
		
		double likelihood = 0.0;
		for(int i=0;i<this.vocabulary_size;i++)
		{
			double parameter = lambda*background_probability[i] + (1 - lambda)*topic_probability[i];
			likelihood = likelihood + count[i] * Math.log(parameter);
			
		}
		
		System.out.println("----------------------");
		System.out.println("Likelihoo at:"+ likelihood);
		return likelihood;
	}
	
	
	public void print(double [] array)
	{
		for(int i=0;i<array.length;i++)
		{
			System.out.println("array["+i+"] =" + array[i]);
		}
	}
	
	public double [] get_topic_probability()
	{
		return this.topic_probability;
	}
	
	public static void main(String[] args) throws IOException, ParseException
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
		
		double lambda = 0.9;
	    int number_of_iteration = 10;
		
		
		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer;
		analyzer = new jsonAnalyzer(tokenModel, classNumber, Ngram);
		
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
	
		double back_ground_probability [] = analyzer.get_back_ground_probabilty();
		
		
		ArrayList<_Doc> docs = analyzer.returnCorpus(finalLocation).getCollection(); // Get the collection of all the documents.
		int N = docs.size();
		
			for(int i = 0; i < docs.size(); i++){
				_Doc temp = docs.get(i);
				twoTopic model = new twoTopic(number_of_iteration,back_ground_probability.length, lambda, back_ground_probability, analyzer.get_term_frequency_in_doc(temp));
			
				
				double topic_probability [] = model.get_topic_probability();
				
				double sum = 0.0;
				for(int k=0; k<topic_probability.length;k++)
				{
					sum = sum + topic_probability[k];
				}
				
				System.out.println("Topic Probability sum:" + sum);
				
				Hashtable<String, Double> dictionary = new Hashtable<String, Double>();
				
				for(int k=0; k<topic_probability.length;k++)
				{
					dictionary.put(analyzer.get_Term_Name(k), topic_probability[k]);
				}
				
				List<Map.Entry> list = new ArrayList<Map.Entry>(dictionary.entrySet());
		        Collections.sort(list, new Comparator<Map.Entry>() {
		           @Override public int compare(Map.Entry e1, Map.Entry e2) {
		        	    Double i1 = (Double) e1.getValue();
		        	    Double i2 = (Double) e2.getValue();
		                if(i1 > i2)
		                	return -1;
		                else if(i1 < i2)
		                	return 1;
		                else
		                	return 0;
		            }
		        });
		        
		        for(Map.Entry e : list) 
		        {
		        	System.out.println("Term:"+e.getKey()+ ", P(W|theta):"+ e.getValue());
					
		        }
				
				
				
			}
			
		
	}
}
