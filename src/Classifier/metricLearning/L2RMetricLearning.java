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
	double[] m_LabeledCache; // cached pairwise similarity between labeled examples
	protected Model m_rankSVM;
	
	final int RankFVSize = 6;// features to be defined in genRankingFV()
	
	public L2RMetricLearning(_Corpus c, String classifier, double C, int topK) {
		super(c, classifier, C);
		m_topK = topK;
	}

	public L2RMetricLearning(_Corpus c, String classifier, double C,
			double ratio, int k, int kPrime, double alhpa, double beta,
			double delta, double eta, boolean storeGraph,
			int topK) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, eta,
				storeGraph);
		m_topK = topK;
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
		
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_topK);
		Feature[] rankFvs = null;
		ArrayList<Feature[]> featureArray = new ArrayList<Feature[]>();
		ArrayList<Integer> targetArray = new ArrayList<Integer>();
		
		_Doc di, dj;
		int label;
		_RankItem ritm, ritn;
		for(int i=0; i<m_trainSet.size(); i++) {
			//query document
			di = m_trainSet.get(i);
			
			//using content similarity to construct initial ranking
			for(int j=0; j<m_trainSet.size(); j++) {
				if (i==j)
					continue;	
				dj = m_trainSet.get(j);
				label = di.getYLabel() == dj.getYLabel() ? 1 : 0;
				neighbors.add(new _RankItem(j, m_LabeledCache[getIndex(i,j)], label));
			}
			
			//construct features for the most similar documents with respect to the query di
			for(_RankItem it:neighbors) {
				dj = m_trainSet.get(it.m_index);
				dj.setRankingFvs(genRankingFV(di, dj));
			}
			
			//extract all preference pairs based on the ranking features
			for(int m=0; m<neighbors.size(); m++) {			
				ritm = neighbors.get(m);
				for(int n=m+1; n<neighbors.size(); n++) {
					ritn = neighbors.get(n);
					
					//test rank preference
					if (ritm.m_label == ritn.m_label)
						continue;
					
					//test feature difference
					rankFvs = genPairwiseRankingFV(m_trainSet.get(ritm.m_index), m_trainSet.get(ritn.m_index));
					if (rankFvs==null)
						continue;
						
					//store the preference pair
					featureArray.add(rankFvs);
					if (ritm.m_label > ritn.m_label)
						targetArray.add(1);
					else
						targetArray.add(0);
				}
			}
			
			//clear the cache for next query
			neighbors.clear();
		}
		
		System.out.format("Generate %d pairs for rank SVM training...\n", featureArray.size());
		return SVM.libSVMTrain(featureArray, targetArray, RankFVSize, SolverType.L2R_L1LOSS_SVC_DUAL, 1.0, 1);
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
		fv[3] = m_classifier.score(q, 1) - m_classifier.score(d, 1);//how to deal with multi-class instances?
		
		//feature 5: sparse feature length difference
		fv[4] = (double)(q.getDocLength() - d.getDocLength())/(double)q.getDocLength();
		
		//feature 6: jaccard coefficient
		fv[5] = Utils.jaccard(q.getSparse(), d.getSparse());
		
		//feature 7: lexicon based sentiment scores
		
		
		//Part II: pointwise features for document
		//stop words proportion
		//average IDF
		//average neighborhood similarity
		
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
