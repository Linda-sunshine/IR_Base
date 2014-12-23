package topicmodels;

import java.io.IOException;
import java.text.ParseException;
import java.util.Arrays;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import Analyzer.jsonAnalyzer;

/**
 * @author Md. Mustafizur Rahman (mr4xb@virginia.edu)
 * two-topic Topic Modeling 
 */

public class twoTopic extends TopicModel {
	private double[] m_theta;//p(w|\theta) - the only topic for each document
	private double[] m_sstat;//c(w,d)p(z|w) - sufficient statistics for each word under topic
	/*p (w|theta_b) */
	protected double[] background_probability;
	
	public twoTopic(int number_of_iteration, int vocabulary_size, double lambda, double beta, 
			double back_ground [], _Corpus c) {
		super(vocabulary_size, number_of_iteration, lambda, beta, c);
		
		background_probability = back_ground;
		m_theta = new double[vocabulary_size];
		m_sstat = new double[vocabulary_size];
	}
	
	@Override
	protected void initialize_probability() {	
    	Utils.randomize(m_theta, beta);
    	Arrays.fill(m_sstat, 0);
	}
	
	@Override
	public void calculate_E_step(_Doc d) {
		for(_SparseFeature fv:d.getSparse()) {
			int wid = fv.getIndex();
			m_sstat[wid] = (1-lambda)*m_theta[wid];
			m_sstat[wid] = fv.getValue() * m_sstat[wid]/(m_sstat[wid]+lambda*background_probability[wid]);//compute the expectation
		}
	}
	
	@Override
	public void calculate_M_step()
	{		
		double sum = Utils.sumOfArray(m_sstat) + vocabulary_size * beta;//with smoothing
		for(int i=0;i<vocabulary_size;i++)
			m_theta[i] = (beta+m_sstat[i]) / sum;
	}
	
	protected double calculate_log_likelihood(_Doc d)
	{		
		double logLikelihood = 0.0;
		for(_SparseFeature fv:d.getSparse())
		{
			int wid = fv.getIndex();
			logLikelihood += fv.getValue() * Math.log(lambda*background_probability[wid] + (1-lambda)*m_theta[wid]);
		}
		
		return logLikelihood;
	}

	@Override
	public void printTopWords(int k) {
		//we only have one topic to show
		MyPriorityQueue<_RankItem> fVector = new MyPriorityQueue<_RankItem>(k);
		for(int i=0; i<m_theta.length; i++) 
			fVector.add(new _RankItem(m_corpus.getFeature(i), m_theta[i]));
		
		for(_RankItem it:fVector)
			System.out.format("%s(%.3f)\t", it.m_name, it.m_value);
		System.out.println();
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
			i++;
		} while (delta>1e-4 && i<this.number_of_iteration);
		
		double perplexity = Math.exp(-current/d.getTotalDocLength());
		System.out.format("Likelihood in document %s converges to %.4f after %d steps...\n", d.getName(), perplexity, i);
		return m_theta;
	}
	
	public static void main(String[] args) throws IOException, ParseException
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
		String finalLocation = "data/Features/selected_fv_stat.txt";

		/*****Parameters in feature selection.*****/
//		String stopwords = "./data/Model/stopwords.dat";
//		String featureSelection = "DF"; //Feature selection method.
//		double startProb = 0.3; // Used in feature selection, the starting point of the features.
//		double endProb = 0.999; // Used in feature selection, the ending point of the features.
//		int DFthreshold = 10; // Filter the features with DFs smaller than this threshold.
//		
//		System.out.println("Performing feature selection, wait...");
//		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, "", Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(featureLocation, featureSelection, startProb, endProb, DFthreshold); //Select the features.

		
		System.out.println("Creating feature vectors, wait...");
		jsonAnalyzer analyzer = new jsonAnalyzer(tokenModel, classNumber, featureLocation, Ngram, lengthThreshold);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.setFeatureValues(featureValue, norm);
		_Corpus c = analyzer.returnCorpus(finalLocation); // Get the collection of all the documents.
		
		/*****parameters for the two-topic topic model*****/
		double lambda = 0.9, beta = 1e-3;
		int topK = 10, number_of_iteration = 20;
		twoTopic model = new twoTopic(number_of_iteration, analyzer.getFeatureSize(), lambda, beta, 
				analyzer.get_back_ground_probabilty(), c);
		
		for(_Doc d:c.getCollection()) {
			model.get_topic_probability(d);
			model.printTopWords(topK);
		}
	}
}
