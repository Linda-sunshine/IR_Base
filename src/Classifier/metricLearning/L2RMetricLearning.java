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

public class L2RMetricLearning extends GaussianFieldsByRandomWalk {
	
	int m_topK;//top K initial ranking results 
	double m_noiseRatio; // to what extend random neighbors can be added 
	double[] m_LabeledCache; // cached pairwise similarity between labeled examples
	protected Model m_rankSVM;
	protected LambdaRank m_lambdaRank;
	double m_tradeoff;
	
	int m_ranker; // 0: pairwise rankSVM; 1: LambdaRank
	ArrayList<_Query> m_queries = new ArrayList<_Query>();
	final int RankFVSize = 9;// features to be defined in genRankingFV()
	
	
	public L2RMetricLearning(_Corpus c, String classifier, double C, int topK) {
		super(c, classifier, C);
		m_topK = topK;
		m_noiseRatio = 0.0; // no random neighbor is needed 
		m_tradeoff = 1.0;
		m_ranker = 0; // default ranker is rankSVM
	}

	public L2RMetricLearning(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta,
			double delta, double eta, boolean storeGraph,
			int topK, double noiseRatio) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta,
				storeGraph);
		m_topK = topK;
		m_noiseRatio = noiseRatio;
		m_tradeoff = 1.0; // should be specified by the user
		m_ranker = 1;
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
			m_lambdaRank = new LambdaRank(RankFVSize, m_tradeoff, m_queries);
			m_lambdaRank.train(300, 20, 1.0);//lambdaRank specific parameters
			
			w = m_lambdaRank.getWeights();
		}
		
		for(int i=0; i<RankFVSize; i++)
			System.out.print(w[i] + " ");
		System.out.println();
	}
	
	private void calcLabeledSimilarities() {
		int L = m_trainSet.size(), size = L*(L-1)/2;//no need to compute diagnoal
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
				|| (di.getYLabel() == 1 && negQ < 0.9*posQ)){
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
		
		System.out.format("Generate %d(%d:%d) queries for L2R model training...\n", pairSize, posQ, negQ);
		return pairSize;
	}
	
	//generate ranking features for a query document pair
	double[] genRankingFV(_Doc q, _Doc d) {
		double[] fv = new double[RankFVSize];
		
		//Part I: pairwise features for query document pair
		//feature 1: cosine similarity
		fv[0] = getBoWSim(q, d);
		
		//feature 2: topical similarity
		fv[1] = getTopicalSim(q, d);
		
		//feature 3: belong to the same product
		fv[2] = q.sameProduct(d)?1:0;
		
		//feature 4: classifier's prediction difference
		fv[3] = Math.abs(m_classifier.score(q, 1) - m_classifier.score(d, 1));//how to deal with multi-class instances?
		
		//feature 5: sparse feature length difference
		fv[4] = Math.abs((double)(q.getDocLength() - d.getDocLength())/(double)q.getDocLength());
		
		//feature 6: jaccard coefficient
		fv[5] = Utils.jaccard(q.getSparse(), d.getSparse());
		
		//feature 7: lexicon based sentiment scores
		fv[6] = 0;
 		
		//Part II: pointwise features for document
		//feature 8: stop words proportion
		fv[7] = d.getStopwordProportion();
		
		//feature 9: average IDF
		fv[8] = d.getAvgIDF();
		//average neighborhood similarity
		
		return fv;
	}
}
