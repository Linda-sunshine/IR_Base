package Classifier.metricLearning;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import clustering.KMeansAlg;
import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._Pair;
import structures._QUPair;
import structures._Query;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.LogisticRegression;
import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.SolverType;
import Ranker.LambdaRank;
import Ranker.LambdaRank.OptimizationType;
import Ranker.LambdaRankParallel;

public class L2RMetricLearning extends GaussianFieldsByRandomWalk {
	
	int m_topK;//top K initial ranking results 
	double m_noiseRatio; // to what extend random neighbors can be added 
	double m_queryRatio; // added by Lin, control the ratio of the class sentiment.
	double m_documentRatio; // added by Lin, controlt the ratio of the document selection for each query.
	
	double[] m_LabeledCache; // cached pairwise similarity between labeled examples
	protected Model m_rankSVM;
	protected LogisticRegression m_rankLR;
	protected LambdaRank m_lambdaRank;
	double m_tradeoff;
	boolean m_multithread = false; // by default we will use single thread

	int m_ranker; // 0: pairwise rankSVM; 1: LambdaRank
	ArrayList<_Query> m_queries = new ArrayList<_Query>();
	final int RankFVSize = 11;// features to be defined in genRankingFV()
	ArrayList<ArrayList<_Doc>> m_clusters;
	HashMap<_Pair, Integer> m_LCSMap;//Added by Lin for storing LCS pairs.
	
	//Added by Lin for lambdaRank parameter tuning.
	double m_shrinkage=0.98;
	double m_stepSize=1;
	int m_maxIter = 300;
	int m_windowSize = 20;
	
	public L2RMetricLearning(_Corpus c, String classifier, double C, int topK) {
		super(c, classifier, C);
		m_topK = topK;
		m_noiseRatio = 0.0; // no random neighbor is needed 
		m_tradeoff = 1.0;
		m_ranker = 0; // default ranker is rankSVM
		m_queryRatio = 1.0;
	}

	public L2RMetricLearning(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta,
			double delta, double eta, boolean weightedAvg,
			int topK, double noiseRatio, int ranker, boolean multithread) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta,
				weightedAvg);
		m_topK = topK;
		m_noiseRatio = noiseRatio;
		m_tradeoff = 1.0; // should be specified by the user
		m_ranker = ranker;
		m_multithread = multithread;
		m_queryRatio = 1.0;
	}
	
	//In lambdaRank, the tradeoff = lambda.added by Lin.
	public void setLambda(double lambda){
		m_tradeoff = lambda;
	}

	public void setShrinkage(double sk){
		m_shrinkage = sk;
	}

	public void setStepSize(double ss){
		m_stepSize = ss;
	}
	
	public void setWindowSize(int ws){
		m_windowSize = ws;
	}
	
	public void setMaxIter(int maxIter){
		m_maxIter = maxIter;
	}
	@Override
	public String toString() {
		String ranker;
		if (m_ranker==0)
			ranker = "RankSVM";
		else
			ranker = "LambdaRank@MAP";
		return String.format("%s-[%s]", super.toString(), ranker);
	}
	
	//NOTE: this similarity is no longer symmetric!!
	@Override
	public double getSimilarity(_Doc di, _Doc dj) {
		
		double similarity = 0;
		
		if (m_ranker==0) 
			similarity = Linear.predictValue(m_rankSVM, genRankingFV(di, dj), 0);
//		else if(m_ranker == 2)
//			similarity = m_rankLR.score(genRankingFV(di, dj), 0);
		else
			similarity = m_lambdaRank.score(genRankingFV(di, dj));
		
		if (Double.isNaN(similarity)){
			System.out.println("similarity calculation hits NaN!");
			System.exit(-1);
		} else if (Double.isInfinite(similarity)){
			System.out.println("similarity calculation hits infinite!");
			System.exit(-1);
		} 
		
		return Math.exp(similarity);//to make sure it is positive			
	}
	
	@Override
	protected void init() {
		super.init();
		
		if (m_queries==null)
			m_queries = new ArrayList<_Query>();
		else
			m_queries.clear();
	}

	@Override
	public void train(Collection<_Doc> trainSet) {
		super.train(trainSet);
		
		L2RModelTraining();
	}
	
	protected void L2RModelTraining() {
		//select the training pairs
		createTrainingCorpus();
		
		double[] w;
		if (m_ranker==0) {
			ArrayList<Feature[]> fvs = new ArrayList<Feature[]>();
			ArrayList<Integer> labels = new ArrayList<Integer>();
			
			for(_Query q:m_queries)
				q.extractPairs4RankSVM(fvs, labels);
			
			m_rankSVM = SVM.libSVMTrain(fvs, labels, RankFVSize, SolverType.L2R_L1LOSS_SVC_DUAL, m_tradeoff, -1);
			w = m_rankSVM.getFeatureWeights();	
		} else if(m_ranker==2) {//RankLR
			ArrayList<Feature[]> fvs = new ArrayList<Feature[]>();
			ArrayList<Integer> labels = new ArrayList<Integer>();
			
			for(_Query q:m_queries)
				q.extractPairs4RankSVM(fvs, labels);
				
			//Transform the instances to _Doc to pass in LR.
			int index = 0;
			ArrayList<_Doc> trainSet = new ArrayList<_Doc>();
			for(int i =0; i<fvs.size(); i++){
				Feature[] features = fvs.get(i); //get one instance.
				_SparseFeature[] sparseFeatures = new _SparseFeature[features.length];
				for(int j=0; j < features.length; j++)
					sparseFeatures[j] = new _SparseFeature(features[j].getIndex(), features[j].getValue());
	
				_Doc tmpDoc = new _Doc(index++, 1, sparseFeatures);
				trainSet.add(tmpDoc);
			}
			m_rankLR = new LogisticRegression(m_classNo, m_featureSize, 0.5);
			m_rankLR.train(trainSet);			
			w = m_rankLR.getParameter();	

		} else{//all the rest use LambdaRank with different evaluator
			if (m_multithread) {
				/**** multi-thread version ****/
				m_lambdaRank = new LambdaRankParallel(RankFVSize, m_tradeoff, m_queries, OptimizationType.OT_MAP, 10);
				m_lambdaRank.train(m_maxIter, m_windowSize, m_stepSize, m_shrinkage);//lambdaRank specific parameters
			} else {
				/**** single-thread version ****/
				m_lambdaRank = new LambdaRank(RankFVSize, m_tradeoff, m_queries, OptimizationType.OT_NDCG);//tradeoff is the lambda in LambdaRank.
				m_lambdaRank.train(m_maxIter, m_windowSize, m_stepSize, m_shrinkage);//lambdaRank specific parameters
			}			
			w = m_lambdaRank.getWeights();
		} 
		
		for(int i=0; i<RankFVSize; i++)
			System.out.format("%.5f ", w[i]);
		System.out.println();
	}
	
	//this is an important feature and will be used repeated
	private void calcLabeledSimilarities() {
		System.out.println("Creating cache for labeled documents...");
		
		int L = m_trainSet.size(), size = L*(L-1)/2;//no need to compute diagonal
		if (m_LabeledCache==null || m_LabeledCache.length<size)
			m_LabeledCache = new double[size];
		
		//using Collection<_Doc> trainSet to pass corpus parameter is really awkward
		_Doc di, dj;
		for(int i=1; i<m_trainSet.size(); i++) {
			di = m_trainSet.get(i);
			for(int j=0; j<i; j++) {
				dj = m_trainSet.get(j);
				m_LabeledCache[getIndex(i,j)] = super.getSimilarity(di, dj);
			}
		}
	}
	
	int getIndex(int i, int j) {
		if (i<j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		return i*(i-1)/2+j;//lower triangle for the square matrix, index starts from 1 in liblinear
	}
 	
	
	public void setQueryRatio(double r){
		m_queryRatio = r;
	}
	
	public void setDocumentRatio(double r){
		m_documentRatio = r;
	}
	//In this training process, we want to get the weight of all pairs of samples.
	protected int createTrainingCorpus(){
		//pre-compute the similarity between labeled documents
		calcLabeledSimilarities();
		
		MyPriorityQueue<_RankItem> simRanker = new MyPriorityQueue<_RankItem>(m_topK);
		ArrayList<_Doc> neighbors = new ArrayList<_Doc>();
		
		_Query q;		
		_Doc di, dj;
		int posQ = 0, negQ = 0, pairSize = 0;
		double relevant = 0, irrelevant = 0;
		
		for(int i=0; i<m_trainSet.size(); i++) {
			//candidate query document
			di = m_trainSet.get(i);
			relevant = 0;
			irrelevant = 0;
			
			//using content similarity to construct initial ranking
			for(int j=0; j<m_trainSet.size(); j++) {
				if (i==j)
					continue;	
				dj = m_trainSet.get(j);
				simRanker.add(new _RankItem(j, m_LabeledCache[getIndex(i,j)]));
			}
			
			//find the top K similar documents by default similarity measure
			for(_RankItem it:simRanker) {
				dj = m_trainSet.get(it.m_index);
				neighbors.add(dj);
				if (di.getYLabel() == dj.getYLabel())
					relevant ++;
				else
					irrelevant ++;
			}
			
			//inject some random neighbors 
			int j = 0;
			while(neighbors.size()<(1.0+m_noiseRatio)*m_topK) {
				if (i!=j) {
					dj = m_trainSet.get(j);
					if (Math.random()<0.02 && !neighbors.contains(dj)) {
						neighbors.add(dj);
						if (di.getYLabel() == dj.getYLabel())
							relevant ++;
						else
							irrelevant ++;
					}
				}
				
				j = (j+1) % m_trainSet.size();//until we use up all the random budget 
			}
			
			if (relevant==0 || irrelevant==0 
				|| (di.getYLabel() == 1 && negQ < m_queryRatio*posQ) || (di.getYLabel()==1 && relevant/irrelevant > m_documentRatio)){
				//clear the cache for next query
				simRanker.clear();
				neighbors.clear();
				continue;
			} else if (di.getYLabel()==1)
				posQ ++;
			else
				negQ ++;
				
			//accept the query
			q = new _Query();
			m_queries.add(q);
			
			//construct features for the most similar documents with respect to the query di
			for(_Doc d:neighbors)
				q.addQUPair(new _QUPair(d.getYLabel()==di.getYLabel()?1:0, genRankingFV(di, d)));
			pairSize += q.createRankingPairs();
			
			//clear the cache for next query
			simRanker.clear();
			neighbors.clear();
		}
		
		System.out.format("Generate %d(%d:%d) ranking pairs for L2R model training...\n", pairSize, posQ, negQ);
		return pairSize;
	}
	// Added by Lin, pass the clustering results back to L2R.
	public void setClusters(ArrayList<ArrayList<_Doc>> clusters){
		m_clusters = clusters;
	}
	
	//In this create training corpus, we use clustering to do document selection.
//	protected int createTrainingCorpus(){
//		//pre-compute the similarity between labeled documents
//		calcLabeledSimilarities();
//		
//		MyPriorityQueue<_RankItem> simRanker = new MyPriorityQueue<_RankItem>(m_topK);
//		ArrayList<_Doc> neighbors = new ArrayList<_Doc>();
//		
//		_Query q;		
//		_Doc di, dj;
//		int posQ = 0, negQ = 0, pairSize = 0, index = 0;
//		double relevant = 0, irrelevant = 0;
//		
//		for(int i=0; i<m_trainSet.size(); i++) {
//			//Candidate query.
//			di = m_trainSet.get(i);
//			//Filter out unlabeled data + calculate sim(q, d_L);
//			for(int j=0; j<m_clusters.size(); j++){
//				simRanker.clear();
//				//Select the most similar reviews in each cluster as the documents.
//				for(_Doc d: m_clusters.get(j)){
//					if(m_trainSet.contains(d)){
//						index = m_trainSet.indexOf(d);
//						if(i != index)
//							simRanker.add(new _RankItem(index, m_LabeledCache[getIndex(i, index)]));
//					}
//				}			
//				//Pick the top k from each cluster;
//				for(_RankItem it:simRanker) {
//					dj = m_trainSet.get(it.m_index);
//					neighbors.add(dj);
//					if (di.getYLabel() == dj.getYLabel())
//						relevant ++;
//					else
//						irrelevant ++;
//				}
//			}
//			
//			if (relevant==0 || irrelevant==0 
//			|| (di.getYLabel() == 1 && negQ < m_queryRatio*posQ)){
//			//clear the cache for next query
//				simRanker.clear();
//				neighbors.clear();
//				continue;
//			} else if (di.getYLabel()==1)
//				posQ ++;
//			else
//				negQ ++;
//				
//			//accept the query
//			q = new _Query();
//			m_queries.add(q);
//			
//			//construct features for the most similar documents with respect to the query di
//			for(_Doc d:neighbors)
//				q.addQUPair(new _QUPair(d.getYLabel()==di.getYLabel()?1:0, genRankingFV(di, d)));
//			pairSize += q.createRankingPairs();
//			
//			//clear the cache for next query
//			neighbors.clear();
//		}
//		
//		System.out.format("Generate %d(%d:%d) ranking pairs for L2R model training...\n", pairSize, posQ, negQ);
//		return pairSize;
//	}
	
	//generate ranking features for a query document pair
	double[] genRankingFV(_Doc q, _Doc d) {
		double[] fv = new double[RankFVSize];
		
		//Part I: pairwise features for query document pair
		//feature 1: cosine similarity
		fv[0] = getBoWSim(q, d);//0.03900
		
		//feature 2: topical similarity
		fv[1] = getTopicalSim(q, d);//-0.08513
		
		//feature 3: belong to the same product
		fv[2] = q.sameProduct(d)?1:0;//0.02104

		//feature 4: sparse feature length difference
		fv[3] = Math.abs((double)(q.getDocLength() - d.getDocLength())/(double)q.getDocLength());//-0.01580
		
		//feature 5: jaccard coefficient
		fv[4] = Utils.jaccard(q.getSparse(), d.getSparse());//0.02190
 		
		//feature 6: the sentiwordnet score for a review.
		fv[5] = Math.abs(q.getSentiScore() - d.getSentiScore());//-0.00103
		
		// feature 7: the pos tagging score for a pair of reviews.
		fv[6] = getPOSScore(q, d);//0.04831
		
		// feature 8: the aspect score for a pair of reviews.
		fv[7] = getAspectScore(q, d);//0.10005
		
		//Part II: pointwise features for document
		//feature 9: stop words proportion
		fv[8] = d.getStopwordProportion();//0.00060
		
		//feature 10: average IDF
		fv[9] = d.getAvgIDF();//0.02447
		
//		fv[10] = Utils.LCS2Doc(q, d);
		//Part I: pairwise features for query document pair
////		// feature 11: the longest subsequence of a query and a document.
//		if(m_LCSMap.containsKey(new _Pair(q.getID(), d.getID())))
//				fv[10] = m_LCSMap.get(new _Pair(q.getID(), d.getID()));
//		else{
//			fv[10] = 0; 
//			System.out.println("The pair does not exist!");
//		}
//		
//		// feature 12: the title of review
//		// fv[11] = d.getTitleScore();

		return fv;
	}
	
	//Set the lcs map in the class, @added by Lin
	public void setLCSMap(HashMap<_Pair, Integer> map){
		m_LCSMap = map;
		System.out.println(map.size() + " is maped to the learning to rank!");
	}
}
