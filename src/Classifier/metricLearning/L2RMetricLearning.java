package Classifier.metricLearning;

import java.util.ArrayList;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import utils.Utils;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.SolverType;

public class L2RMetricLearning extends GaussianFieldsByRandomWalk {
	
	int m_topK;//top K initial ranking results 
	double m_noiseRatio; // to what extend random neighbors can be added 
	double[] m_LabeledCache; // cached pairwise similarity between labeled examples
	protected Model m_rankSVM;
	
	final int RankFVSize = 12;// features to be defined in genRankingFV()
	
	public L2RMetricLearning(_Corpus c, String classifier, double C, int topK) {
		super(c, classifier, C);
		m_topK = topK;
		m_noiseRatio = 0.0; // no random neighbor is needed 
	}

	public L2RMetricLearning(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta,
			double delta, double eta, boolean storeGraph,
			int topK, double noiseRatio) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta,
				storeGraph);
		m_topK = topK;
		m_noiseRatio = noiseRatio;
	}
	
	//NOTE: this similarity is no longer symmetric!!
	@Override
	public double getSimilarity(_Doc di, _Doc dj) {
		
		double similarity = Linear.predictValue(m_rankSVM, genRankingFV(di, dj), 0);
		
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
	public void train(Collection<_Doc> trainSet) {
		super.train(trainSet);
		
		m_rankSVM = trainRankSVM();
		
		double[] w = m_rankSVM.getFeatureWeights();
		for(int i=0; i<m_rankSVM.getNrFeature(); i++)
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
	protected Model trainRankSVM(){
		//pre-compute the similarity between labeled documents
		calcLabeledSimilarities();
		
		MyPriorityQueue<_RankItem> simRanker = new MyPriorityQueue<_RankItem>(m_topK);
		ArrayList<_Doc> neighbors = new ArrayList<_Doc>();
		Feature[] rankFvs = null;
		ArrayList<Feature[]> featureArray = new ArrayList<Feature[]>();
		ArrayList<Integer> targetArray = new ArrayList<Integer>();
		
		_Doc di, dj, dk;
		int label_j, label_k, posQ = 0, negQ = 0;
		for(int i=0; i<m_trainSet.size(); i++) {
			//query document
			di = m_trainSet.get(i);
			
			if (di.getYLabel() == 1 && negQ < 0.8*posQ)
				continue;
			else if (di.getYLabel() == 0 && posQ < 0.8*negQ)
				continue;
			
			//using content similarity to construct initial ranking
			for(int j=0; j<m_trainSet.size(); j++) {
				if (i==j)
					continue;	
				dj = m_trainSet.get(j);
				simRanker.add(new _RankItem(j, m_LabeledCache[getIndex(i,j)]));
			}
			
			//find the top K similar documents by default similarity measure
			for(_RankItem it:simRanker)
				neighbors.add(m_trainSet.get(it.m_index));
			
			//inject some random neighbors 
			for(int j=0; j<m_trainSet.size() && neighbors.size()<(1.0+m_noiseRatio)*m_topK; j++) {
				if (i==j)
					continue;	
				
				dj = m_trainSet.get(j);
				if (Math.random()<0.005 && !neighbors.contains(dj))
					neighbors.add(dj);
			}
			
			//construct features for the most similar documents with respect to the query di
			for(_Doc d:neighbors) 
				d.setRankingFvs(genRankingFV(di, d));
			
			//extract all preference pairs based on the ranking features
			for(int j=0; j<neighbors.size(); j++) {			
				dj = neighbors.get(j);
				label_j = di.getYLabel() == dj.getYLabel()?1:0;
				for(int k=j+1; k<neighbors.size(); k++) {
					dk = neighbors.get(k);
					label_k = di.getYLabel() == dk.getYLabel()?1:0;
					
					//test rank preference
					if (label_j == label_k)
						continue;
					
					//test feature difference
					rankFvs = genPairwiseRankingFV(dj, dk);
					if (rankFvs==null)
						continue;
						
					//store the preference pair
					featureArray.add(rankFvs);
					if (label_j > label_k)
						targetArray.add(1);
					else
						targetArray.add(-1);
				}
			}
			
			if (di.getYLabel()==1)
				posQ ++;
			else
				negQ ++;
			
			//clear the cache for next query
			simRanker.clear();
			neighbors.clear();
		}
		
		System.out.format("Generate %d(%d:%d) pairs for rankSVM training...\n", featureArray.size(), posQ, negQ);
		return SVM.libSVMTrain(featureArray, targetArray, RankFVSize, SolverType.L2R_L1LOSS_SVC_DUAL, 1.0, -1);
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
		fv[6] = Utils.cosine(q.m_sentiment, d.m_sentiment);
 		
		//Part II: pointwise features for document
		//feature 8: stop words proportion
		fv[7] = d.getStopwordProportion();
		
		//feature 9: average IDF
		fv[8] = d.getAvgIDF();
		//average neighborhood similarity
		
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
	
	//di should be ranked higher than dj
	Feature[] genPairwiseRankingFV(_Doc di, _Doc dj) {
		ArrayList<Feature> fvs = new ArrayList<Feature>();
		double value;
		for(int i=0; i<RankFVSize; i++) {
			value = di.m_rankingFvs[i] - dj.m_rankingFvs[i]; 
			if (value != 0)
				fvs.add(new FeatureNode(i+1, value));
		}
		
		if (fvs.size()==0)
			return null;
		return fvs.toArray(new Feature[fvs.size()]);
	}
}
