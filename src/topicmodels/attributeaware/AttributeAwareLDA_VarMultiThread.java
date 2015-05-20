/**
 * 
 */
package topicmodels.attributeaware;

import java.io.IOException;
import java.text.ParseException;
import java.util.Arrays;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import topicmodels.multithreads.LDA_Variational_multithread;
import utils.Utils;
import Analyzer.newEggAnalyzer;

/**
 * @author hongning
 * Attribute aware LDA - using posterior regularization to incorporate attribute for topic modeling
 */
public class AttributeAwareLDA_VarMultiThread extends LDA_Variational_multithread {
	
	public class AttributeAwareLDA_worker extends LDA_worker {		
		public AttributeAwareLDA_worker() {
			super();
		}
		
		@Override
		public double calculate_E_step(_Doc d) {	
			double last = calculate_log_likelihood(d), current = last, converge, logSum, v;
			int iter = 0, wid;
			double[] values;
			_SparseFeature fv[] = d.getSparse(), spFea;
			
			do {
				//variational inference for p(z|w,\phi)
				for(int n=0; n<fv.length; n++) {
					//reset the estimates
					Arrays.fill(d.m_phi[n], -100);//exp(-100) should be small enough
					
					//allocate the words by attribute and topic combination
					spFea = fv[n];
					wid = spFea.getIndex();
					values = spFea.getValues();
					for(int a=0; a<values.length; a++) {
						v = values[a];
						if (v<1)//no observations (at least 1.0)
							continue;
						else if (a<m_attributeSize) {
							for(int i=0; i<number_of_topics; i++) {
								//special organization of topics
								if (i%m_attributeSize==a)//disable the proportion from the other attributes
									d.m_phi[n][i] = v*topic_term_probabilty[i][wid] + Utils.digamma(d.m_sstat[i]);
							}
						} else {//mixing part of all possible attributes
							for(int i=0; i<number_of_topics; i++)
								d.m_phi[n][i] = Utils.logSum(d.m_phi[n][i], v*topic_term_probabilty[i][wid] + Utils.digamma(d.m_sstat[i]));
						}
					}
					
					logSum = Utils.logSumOfExponentials(d.m_phi[n]);
					for(int i=0; i<number_of_topics; i++)
						d.m_phi[n][i] = Math.exp(d.m_phi[n][i] - logSum);
				}
				
				//variational inference for p(\theta|\gamma)
				System.arraycopy(m_alpha, 0, d.m_sstat, 0, m_alpha.length);
				for(int n=0; n<fv.length; n++) {
					v = fv[n].getValue();
					for(int i=0; i<number_of_topics; i++)
						d.m_sstat[i] += d.m_phi[n][i] * v;// 
				}
				
				if (m_varConverge>0) {
					current = calculate_log_likelihood(d);			
					converge = Math.abs((current - last)/last);
					last = current;
					
					if (converge<m_varConverge)
						break;
				}
			} while(++iter<m_varMaxIter);
			
			//collect the sufficient statistics after convergence
			this.collectStats(d);
			
			return current;
		}
	}
	
	int m_attributeSize;//[0, m_attributeSize] sections in _SparseFeature have attribute-based constraints
	
	public AttributeAwareLDA_VarMultiThread(int number_of_iteration,
			double converge, double beta, _Corpus c, double lambda,
			double[] back_ground, int number_of_topics, double alpha,
			int varMaxIter, double varConverge, int attributeSize) {
		super(number_of_iteration, converge, beta, c, lambda, back_ground,
				number_of_topics*attributeSize, alpha, varMaxIter, varConverge);
		
		m_attributeSize = attributeSize;
	}

	public static void main(String[] args) throws IOException, ParseException {	
		int classNumber = 5; //Define the number of classes in this Naive Bayes.
		int Ngram = 2; //The default value is unigram. 
		int lengthThreshold = 5; //Document length threshold
		
		int number_of_topics = 15;
		double alpha = 1.0 + 1e-2, beta = 1.0 + 1e-3, eta = 5.0;//these two parameters must be larger than 1!!!
		double converge = -1, lambda = 0.7; // negative converge means do need to check likelihood convergency
		int topK = 20, number_of_iteration = 100;
		int attributeSize = 2;//pro and con in NewEgg
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/NewEgg";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model.
		
		String fvFile = String.format("./data/Features/fv_%dgram_topicmodel_newegg.txt", Ngram);

		/*****Parameters in feature selection.*****/
//		String stopwords = "./data/Model/stopwords.dat";
//		String featureSelection = "DF"; //Feature selection method.
//		double startProb = 0.2; // Used in feature selection, the starting point of the features.
//		double endProb = 0.999; // Used in feature selection, the ending point of the features.
//		int DFthreshold = 10; // Filter the features with DFs smaller than this threshold.
//		
//		System.out.println("Performing feature selection, wait...");
//		newEggAnalyzer analyzer = new newEggAnalyzer(tokenModel, classNumber, null, Ngram, lengthThreshold, "camera");
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
		
		System.out.println("Creating feature vectors, wait...");
		newEggAnalyzer analyzer = new newEggAnalyzer(tokenModel, classNumber, fvFile, Ngram, lengthThreshold, "camera");
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		_Corpus c = analyzer.returnCorpus(null); // Get the collection of all the documents.
		
		AttributeAwareLDA_VarMultiThread model 
			= new AttributeAwareLDA_VarMultiThread(number_of_iteration, converge, beta, c, 
				lambda, analyzer.getBackgroundProb(), 
				number_of_topics, alpha, 10, -1, attributeSize);
		
		model.LoadPrior(aspectlist, eta);
		model.setDisplay(true);
		model.EMonCorpus();
		model.printTopWords(topK);
	}
}
