package topicmodels;

import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import Analyzer.jsonAnalyzer;

/**
 * @author Md. Mustafizur Rahman (mr4xb@virginia.edu)
 * two-topic Topic Modeling 
 */

public class twoTopic extends TopicModel{
	private double lambda;
	private double[] m_theta;//p(w|\theta)
	private double[] m_sstat;//c(w,d)p(z|w) - sufficient statistics for word under topic
	
	public twoTopic(int number_of_iteration, int vocabulary_size, double lambda, double beta, double back_ground [])
	{
		super(vocabulary_size, number_of_iteration, lambda, beta);
		
		background_probability = back_ground;
		m_theta = new double[vocabulary_size];
		m_sstat = new double[vocabulary_size];
	}
	
	@Override
	protected void initialize_probability() {	
    	Utils.randomize(m_theta);	
	}
	
	@Override
	public void calculate_E_step(_Doc d) {
		for(_SparseFeature fv:d.getSparse()) {
			int wid = fv.getIndex();
			m_sstat[wid] = lambda * background_probability[wid];
			m_sstat[wid] = fv.getValue() * (1 - m_sstat[wid]/(m_sstat[wid] + (1-lambda)*m_theta[wid]));
		}
	}
	
	@Override
	public void calculate_M_step()
	{		
		double sum = Utils.sumOfArray(m_sstat) + vocabulary_size * beta;
		for(int i=0;i<vocabulary_size;i++)
			m_theta[i] = (beta+m_sstat[i]) / sum;
	}
	
	protected double calculate_log_likelihood(_Doc d)
	{		
		double logLikelihood = 0.0, prob;
		for(_SparseFeature fv:d.getSparse())
		{
			int wid = fv.getIndex();
			prob = lambda*background_probability[wid] + (1 - lambda)*m_theta[wid];
			logLikelihood += fv.getValue() * Math.log(prob);
		}
		
		return logLikelihood;
	}
	
	//this is mini-EM in a single document 
	@Override
	public double[] get_topic_probability(_Doc d)
	{
		initialize_probability();
		
		double delta, last = calculate_log_likelihood(), current;
		int  i = 0;
		do
		{
			calculate_E_step(d);
			calculate_M_step();
			
			current = calculate_log_likelihood(d);
			delta = Math.abs((current - last)/last);
			last = current;
			System.out.format("Likelihood %.4f at step %s converge to %.3f...\n", current, i, delta);
			i++;
		} while (delta>1e-6 && i<this.number_of_iteration);
		return m_theta;
	}
	
	public static void main(String[] args) throws IOException, ParseException
	{	
		int featureSize = 0; //Initialize the fetureSize to be zero at first.
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 1; //The default value is unigram. 
		String featureValue = "TF"; //The way of calculating the feature value, which can also be "TFIDF", "BM25"
		int norm = 0;//The way of normalization.(only 1 and 2)
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/test";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		String finalLocation = "data/Features/selected_fv.dat";

		/*****Parameters in feature selection.*****/
		String providedCV = "";
		String featureSelection = "DF"; //Feature selection method.
		double startProb = 0.4; // Used in feature selection, the starting point of the features.
		double endProb = 1; // Used in feature selection, the ending point of the features.
		int DFthreshold = 10; // Filter the features with DFs smaller than this threshold.
		
		/*****Parameters in time series analysis.*****/
		int window = 0;
		System.out.println("Window length: " + window);
		System.out.println("--------------------------------------------------------------------------------------");
		
		double lambda = 0.9, beta = 1e-3;
	    int number_of_iteration = 10;
		
		
		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer;
		analyzer = new jsonAnalyzer(tokenModel, classNumber, Ngram);
		
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
	
		double back_ground_probability [] = analyzer.get_back_ground_probabilty();
		_Corpus c = analyzer.returnCorpus(finalLocation); // Get the collection of all the documents.
		twoTopic model = new twoTopic(number_of_iteration, analyzer.getFeatureSize(), lambda, beta, back_ground_probability);
		
		for(_Doc d:c.getCollection()) {
			double topic_probability [] = model.get_topic_probability(d);
			
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
