package Classifier.metricLearning;

import java.util.ArrayList;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._QUPair;
import structures._Query;
import structures._RankItem;
import utils.Utils;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
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
	double[] m_LabeledCache; // cached pairwise similarity between labeled examples
	protected Model m_rankSVM;
	protected LambdaRank m_lambdaRank;
	double m_tradeoff;
<<<<<<< HEAD
	double m_negRatio;

//	final int RankFVSize = 12;// features to be defined in genRankingFV()
	int m_ranker; // 0: pairwise rankSVM; 1: LambdaRank
	ArrayList<_Query> m_queries = new ArrayList<_Query>();
	final int RankFVSize = 12;// features to be defined in genRankingFV()
=======
	boolean m_multithread = false; // by default we will use single thread
	
	int m_ranker; // 0: pairwise rankSVM; 1: LambdaRank
	ArrayList<_Query> m_queries = new ArrayList<_Query>();
	final int RankFVSize = 12;// features to be defined in genRankingFV()
	
>>>>>>> master
	
	public L2RMetricLearning(_Corpus c, String classifier, double C, int topK) {
		super(c, classifier, C);
		m_topK = topK;
		m_noiseRatio = 0.0; // no random neighbor is needed 
		m_tradeoff = 1.0;
		m_ranker = 0; // default ranker is rankSVM
		m_negRatio = 1.0;
	}

	public L2RMetricLearning(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta,
			double delta, double eta, boolean weightedAvg,
			int topK, double noiseRatio, boolean multithread) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta,
				weightedAvg);
		m_topK = topK;
		m_noiseRatio = noiseRatio;
		m_tradeoff = 1.0; // should be specified by the user
		m_ranker = 1;
<<<<<<< HEAD
		m_negRatio = 1;
	}
	
	public L2RMetricLearning(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta,
			double delta, double eta, boolean storeGraph,
			int topK, double noiseRatio, double negRatio) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta,
				storeGraph);
		m_topK = topK;
		m_noiseRatio = noiseRatio;
		m_tradeoff = 1.0; // should be specified by the user
		m_ranker = 1;
		m_negRatio = negRatio;
=======
		m_multithread = multithread;
>>>>>>> master
	}
	
	//NOTE: this similarity is no longer symmetric!!
	@Override
	public double getSimilarity(_Doc di, _Doc dj) {
		
		double similarity = 0;
		
		if (m_ranker==0) 
			similarity = Linear.predictValue(m_rankSVM, genRankingFV(di, dj), 0);
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
		} else {//all the rest use LambdaRank with different evaluator
			if (m_multithread) {
				/**** multi-thread version ****/
				m_lambdaRank = new LambdaRankParallel(RankFVSize, m_tradeoff, m_queries, OptimizationType.OT_MAP, 10);
				m_lambdaRank.train(100, 20, 1.0, 0.98);//lambdaRank specific parameters
			} else {
				/**** single-thread version ****/
				m_lambdaRank = new LambdaRank(RankFVSize, m_tradeoff, m_queries, OptimizationType.OT_MAP);
				m_lambdaRank.train(300, 20, 1.0, 0.98);//lambdaRank specific parameters
			}			
			w = m_lambdaRank.getWeights();
		}
		
		for(int i=0; i<RankFVSize; i++)
			System.out.format("%.5f ", w[i]);
		System.out.println();
	}
	
	//this is an important feature and will be used repeated
	private void calcLabeledSimilarities() {
<<<<<<< HEAD
=======
		System.out.println("Creating cache for labeled documents...");
		
>>>>>>> master
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
 	
	//In this training process, we want to get the weight of all pairs of samples.
	protected int createTrainingCorpus(){
		//pre-compute the similarity between labeled documents
		calcLabeledSimilarities();
		
		MyPriorityQueue<_RankItem> simRanker = new MyPriorityQueue<_RankItem>(m_topK);
		ArrayList<_Doc> neighbors = new ArrayList<_Doc>();
		
		_Query q;		
		_Doc di, dj;
		int posQ = 0, negQ = 0, pairSize = 0;
		int relevant = 0, irrelevant = 0;
		
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
			
//			for(int j=0; j < m_topK; i++){
//				//get the top one.
//				dj = m_trainSet.get(simRanker.get(j).m_index);
//				neighbors.add(dj);
//				if(di.getYLabel()==dj.getYLabel())
//					relevant++;
//				else irrelevant++;
//				//get the bottom one.
//				dj = m_trainSet.get(simRanker.get(m_trainSet.size()-1-j).m_index);
//				neighbors.add(dj);
//				if(di.getYLabel()==dj.getYLabel())
//					relevant++;
//				else irrelevant++;		
//			}
			
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
<<<<<<< HEAD
				|| (di.getYLabel() == 1 && negQ < m_negRatio*posQ)){
=======
				|| (di.getYLabel() == 1 && negQ < 1.1*posQ)){
>>>>>>> master
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
	
	//generate ranking features for a query document pair
	double[] genRankingFV(_Doc q, _Doc d) {
		double[] fv = new double[RankFVSize];
		
		//Part I: pairwise features for query document pair
		//feature 1: cosine similarity
		fv[0] = getBoWSim(q, d);//0.04298
		
		//feature 2: topical similarity
<<<<<<< HEAD
		fv[1] = getTopicalSim(q, d);
			
=======
		fv[1] = getTopicalSim(q, d);//-0.09567
		
>>>>>>> master
		//feature 3: belong to the same product
		fv[2] = q.sameProduct(d)?1:0;//0.02620
		
		//feature 4: classifier's prediction difference
<<<<<<< HEAD
//		fv[3] = Math.abs(m_classifier.score(q, 1) - m_classifier.score(d, 1));//how to deal with multi-class instances?
=======
		//fv[3] = Math.abs(m_classifier.score(q, 1) - m_classifier.score(d, 1));//how to deal with multi-class instances?
		fv[3] = 0;
>>>>>>> master
		
		//feature 5: sparse feature length difference
		fv[4] = Math.abs((double)(q.getDocLength() - d.getDocLength())/(double)q.getDocLength());//-0.01410
		
		//feature 6: jaccard coefficient
<<<<<<< HEAD
		fv[5] = Utils.jaccard(q.getSparse(), d.getSparse());
		
		//feature 7: lexicon based sentiment scores
		fv[6] = Utils.cosine(q.m_sentiment, d.m_sentiment);
=======
		fv[5] = Utils.jaccard(q.getSparse(), d.getSparse());//0.02441		
>>>>>>> master
 		
		//Part II: pointwise features for document
		//feature 8: stop words proportion
		fv[7] = d.getStopwordProportion();//-0.00005
		
		//feature 9: average IDF
		fv[8] = d.getAvgIDF();//0.03732
		
		//feature 10: the sentiwordnet score for a review.
		fv[9] = d.getSentiScore();

		// feature 11: the postagging score for a pair of reviews.
		fv[10] = getPOSScore(q, d);

		// feature 12: the aspect score for a pair of reviews.
		fv[11] = getAspectScore(q, d);

		// feature 13: the title of review
		// fv[12] = d.getTitleScore();
		
		//feature 10: the sentiwordnet score for a review.
		fv[9] = d.getSentiScore();
		
		//feature 11: the postagging score for a pair of reviews.
		fv[10] = getPOSScore(q, d);
		
		//feature 12: the aspect score for a pair of reviews.
		fv[11] = getAspectScore(q, d);
		
		//feature 13: the title of review
//		fv[12] = d.getTitleScore();
		return fv;
	}
}
