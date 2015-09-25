package topicmodels;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import markovmodel.FastRestrictedHMM;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import utils.Utils;

public class HTMM extends pLSA {
	// HTMM parameter both in log space
	double epsilon;   // estimated epsilon
	
	// cache structure
	double[][] p_dwzpsi;  // The state probabilities that is Pr(z,psi | d,w)
 	double[][] emission;  // emission probability of p(s|z)
	
 	// HMM-style inferencer
 	FastRestrictedHMM m_hmm; 
 	
	// sufficient statistics for p(\epsilon)
	int total; 
	double lot;
	
	double loglik;
	protected int constant;

	public HTMM(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			int number_of_topics, double alpha) {//arguments for pLSA	
		super(number_of_iteration, converge, beta, c,
				0, //HTMM does not have a background setting
				number_of_topics, alpha);
		
		this.epsilon = Math.random();		
		this.constant = 2;
		m_logSpace = true;
		createSpace();
	}
	
	public HTMM(int number_of_iteration, double converge, double beta, _Corpus c, //arguments for general topic model
			int number_of_topics, double alpha, //arguments for pLSA	
			int constant) {
		super(number_of_iteration, converge, beta, c,
				0, //HTMM does not have a background setting
				number_of_topics, alpha);
		
		this.epsilon = Math.random();
		this.constant = constant;
		m_logSpace = true;
		createSpace();
	}
	
	@Override
	protected void createSpace() {
		super.createSpace();
		
		int maxSeqSize = m_corpus.getLargestSentenceSize();		
		m_hmm = new FastRestrictedHMM(epsilon, maxSeqSize, this.number_of_topics, this.constant); 
		
		//cache in order to avoid frequently allocating new space
		p_dwzpsi = new double[maxSeqSize][this.constant * this.number_of_topics]; // max|S_d| * (2*K)
		emission = new double[maxSeqSize][this.number_of_topics]; // max|S_d| * K
	}
	
	@Override
	public String toString() {
		return String.format("HTMM[k:%d, alpha:%.3f, beta:%.3f]", number_of_topics, d_alpha, d_beta);
	}
	
	// Construct the emission probabilities for sentences under different topics in a particular document.
	void ComputeEmissionProbsForDoc(_Doc d) {
		for(int i=0; i<d.getSenetenceSize(); i++) {
			_Stn stn = d.getSentence(i);
			Arrays.fill(emission[i], 0);
			for(int k=0; k<this.number_of_topics; k++) {
				for(_SparseFeature w:stn.getFv()) {
					emission[i][k] += w.getValue() * topic_term_probabilty[k][w.getIndex()];//all in log-space
				}
			}
		}
	}
	
	@Override
	public double calculate_E_step(_Doc d) {
		//Step 1: pre-compute emission probability
		if (m_collectCorpusStats)//indicates this is training
			ComputeEmissionProbsForDoc(d);
		
		//Step 2: use forword/backword algorithm to compute the posterior
		double logLikelihood = m_hmm.ForwardBackward(d, emission);
		loglik += logLikelihood;
		
		//Step 3: collection expectations from the posterior distribution
		m_hmm.collectExpectations(p_dwzpsi);//expectations will be in the original space	
		accTheta(d);
		estThetaInDoc(d);//get the posterior of theta
		
		if (m_collectCorpusStats) {
			accEpsilonStat(d);
			accPhiStat(d);
		}
		
		return logLikelihood + docThetaLikelihood(d);
	}
	
	public int[] get_MAP_topic_assignment(_Doc d) {
		int path [] = new int [d.getSenetenceSize()];
		m_hmm.BackTrackBestPath(d, emission, path);
		return path;
	}	

	//probabilities of topic switch
	void accEpsilonStat(_Doc d) {
		for(int t=1; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<(this.constant-1)*this.number_of_topics; i++) 
				this.lot += this.p_dwzpsi[t][i];
			this.total ++;
		}
	}
	
	//probabilities of topic assignment
	void accPhiStat(_Doc d) {
		double prob;
		for(int t=0; t<d.getSenetenceSize(); t++) {
			_Stn s = d.getSentence(t);
			for(_SparseFeature f:s.getFv()) {
				int wid = f.getIndex();
				double v = f.getValue();//frequency
				for(int i=0; i<this.number_of_topics; i++) {
					prob = this.p_dwzpsi[t][i];
					for(int j=1; j<this.constant; j++)
						prob += this.p_dwzpsi[t][i + j*this.number_of_topics];
					this.word_topic_sstat[i][wid] += v * prob;
				}
			}
		}
	}
	
	void accTheta(_Doc d) {
		for(int t=0; t<d.getSenetenceSize(); t++) {
			for(int i=0; i<this.number_of_topics; i++) 
				for(int j=0; j<this.constant-1; j++)
					d.m_sstat[i] += this.p_dwzpsi[t][i + j*this.number_of_topics];//only consider \psi=1
		}
	}
	
	@Override
	public void calculate_M_step(int iter) {
		if (iter>0) {
			this.epsilon = this.lot/this.total; // to make the code structure concise and consistent, keep epsilon in real space!!
			m_hmm.setEpsilon(this.epsilon);
		}
		
		for(int i=0; i<this.number_of_topics; i++) {
			double sum = Math.log(Utils.sumOfArray(word_topic_sstat[i]));
			for(int v=0; v<this.vocabulary_size; v++)
				topic_term_probabilty[i][v] = Math.log(word_topic_sstat[i][v]) - sum;
		}
	}
	
	protected void init() {
		super.init();
		
		this.loglik = 0;
		this.total = 0;
		this.lot = 0.0;// sufficient statistics for epsilon
	}
	
	@Override
	public double calculate_log_likelihood(_Doc d) {//it is very expensive to re-compute this
		System.err.println("This function should not be called");
		System.exit(-1);
		return -1;
	}
	
	@Override
	public double inference(_Doc d) {
		ComputeEmissionProbsForDoc(d);//to avoid repeatedly computing emission probability 
		
		double current = super.inference(d);
		
		int path[] = get_MAP_topic_assignment(d);
		_Stn[] sentences = d.getSentences();
		for(int i=0; i<path.length;i++)
			sentences[i].setTopic(path[i]);
		
		return current;
	}
	
	public void printTopFeaturesSet(int topK, ArrayList<String> features, String path){
		
		HashMap<Integer, String> list = new HashMap<Integer, String>();
		HashMap<Integer, Double> values = new HashMap<Integer, Double>();
		
		int size = features.size();
		int range = this.number_of_topics/2;
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(size, false);
		PrintWriter topicWiseFeatureWriter = null;
		
		System.out.println("Feature Wise File Path: "+ path);
		try{
			topicWiseFeatureWriter = new PrintWriter(new File(path));
		}catch(Exception e){
			System.err.println(path+" Not found!!");
		}
		
		for(int i=0; i<this.number_of_topics/2; i++){
			MyPriorityQueue<_RankItem> queuePos = new MyPriorityQueue<_RankItem>(150, true);
			MyPriorityQueue<_RankItem> queueNeg = new MyPriorityQueue<_RankItem>(150, true);
			
			for(int n=0; n<features.size(); n++) {
				queuePos.add(new _RankItem(m_corpus.getFeature(n), topic_term_probabilty[i][n]));
				queueNeg.add(new _RankItem(m_corpus.getFeature(n), topic_term_probabilty[i+range][n]));
			}
			topicWiseFeatureWriter.print("Aspect Related Words topic:"+ i +"\n");
			int count = 0;
			for(_RankItem itemPos:queuePos){
				for(_RankItem itemNeg:queueNeg){
					if(itemPos.m_name.equalsIgnoreCase(itemNeg.m_name)){
						topicWiseFeatureWriter.format("%d:%s(%f),(%f)\n", count++, itemPos.m_name,itemPos.m_value, itemNeg.m_value);
					}
				}
			}
			queuePos.clear();
			queueNeg.clear();
		}
		
		
		double logRatio = 0.0;
		
		
		for(int i=0; i<this.number_of_topics/2; i++){
			queue = new MyPriorityQueue<_RankItem>(size, false);
			for(int n=0; n<features.size(); n++) {
				logRatio = Math.log(Math.exp(topic_term_probabilty[i][n])) - Math.log(Math.exp(topic_term_probabilty[i+range][n])); // log Ratio since we have only two classes
				queue.add(new _RankItem(n, logRatio));
			}
		
			System.out.print("Most discriminative features for topic:"+ i +"\n");
			topicWiseFeatureWriter.print("Most discriminative features for topic:"+ i +"\n");
			int j=0;
			int middlePointDetector = 0;
			for(_RankItem item:queue) {
				list.put(j, features.get(item.m_index));
				values.put(j, item.m_value);
				//System.out.format("%d:%s(%f)\n", j, features.get(item.m_index), item.m_value);
				//debugWriter.format("%d:%s(%f)\n", j, features.get(item.m_index), item.m_value);
				j++;
				//since in ascending order so negative values in top and posiitve in bottom 
				if(item.m_value<=0.0){
					middlePointDetector++;
				}
			}
			
			System.out.println("List Size:"+list.size());
			System.out.println("Middle Point:"+ middlePointDetector);
			
			System.out.print("Top 50 featuresi for topic:"+ i +"\n");
			topicWiseFeatureWriter.print("Top 50 features for topic:"+ i +"\n");
			for(j=0;j<topK;j++){
				//System.out.format("%d:%s\n", j, list.get(j));
				topicWiseFeatureWriter.format("%d:%s(%f)\n", j, list.get(j),values.get(j));
			}
			System.out.println();
			topicWiseFeatureWriter.println();
			
			System.out.print("Bottom 50 features for topic:"+ i +"\n");
			topicWiseFeatureWriter.print("Bottom 50 features for topic:"+ i +"\n");
			int k =0;
			for(j=list.size()-1; j>=list.size() - topK;j--){
				//System.out.format("%d:%s\n", k, list.get(j));
				topicWiseFeatureWriter.format("%d:%s(%f)\n", k, list.get(j),values.get(j));
				k++;
			}
			System.out.println();
			topicWiseFeatureWriter.println();
			
			System.out.print("Middle 50 features for topic:"+ i +"\n");
			topicWiseFeatureWriter.print("Middle 50 features for topic:"+ i +"\n");
			//int middleOfList = (int)Math.ceil(list.size()/2.0);
			int middleOfList = middlePointDetector;
			k = 0;
			for(j=middleOfList-topK/2; j<=middleOfList+topK/2;j++){
				//System.out.format("%d:%s\n", k, list.get(j));
				topicWiseFeatureWriter.format("%d:%s(%f)\n", k, list.get(j),values.get(j));
				k++;
			}
			System.out.println();
			topicWiseFeatureWriter.println();
			
			//clearing the list for next topic
			list.clear();
		}
		topicWiseFeatureWriter.flush();
		topicWiseFeatureWriter.close();
	}
	
	public void docSummary(String[] productList){
		for(String prodID : productList) {
			for(int i=0; i<this.number_of_topics; i++){
				MyPriorityQueue<_RankItem> stnQueue = new MyPriorityQueue<_RankItem>(3);//top three sentences per topic per product
				
				for(_Doc d:m_trainSet) {
					if(d.getItemID().equalsIgnoreCase(prodID)) {
						for(int j=0; j<d.getSenetenceSize(); j++){
							_Stn sentence = d.getSentence(j);
							double prob = d.m_topics[i];
							for(_SparseFeature f:sentence.getFv())
								prob += f.getValue() * topic_term_probabilty[i][f.getIndex()];
							prob /= sentence.getLength();
							
							stnQueue.add(new _RankItem(sentence.getRawSentence(), prob));
						}
					}
				}				
				
				System.out.format("Product: %s, Topic: %d\n", prodID, i);
				summaryWriter.format("Product: %s, Topic: %d\n", prodID, i);
				for(_RankItem it:stnQueue){
					System.out.format("%s\t%.3f\n", it.m_name, it.m_value);	
					summaryWriter.format("%s\t%.3f\n", it.m_name, it.m_value);	
				}			
			}
		}
		summaryWriter.flush();
		summaryWriter.close();
	}
}
