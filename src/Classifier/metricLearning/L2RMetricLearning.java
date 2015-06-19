package Classifier.metricLearning;

import java.util.ArrayList;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.SolverType;

public class L2RMetricLearning extends GaussianFieldsByRandomWalk {
	
	int m_topK;//top K initial ranking results 
	double[] m_LabeledCache; // cached pairwise similarity between labeled examples
	protected Model m_rankSVM;
	
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

	@Override
	public void train(Collection<_Doc> trainSet) {
		super.train(trainSet);
		
		m_rankSVM = trainRankSVM();
	}
	
	private void calcLabeledSimilarities() {
		int L = m_trainSet.size();
		if (m_LabeledCache==null || m_LabeledCache.length<L)
			m_LabeledCache = new double[L*(L+1)/2];
		
		//using Collection<_Doc> trainSet to pass corpus parameter is really awkward
		_Doc di, dj;
		for(int i=0; i<m_trainSet.size(); i++) {
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
		return i*(i+1)/2+j;//lower triangle for the square matrix, index starts from 1 in liblinear
	}
 	
	//In this training process, we want to get the weight of all pairs of samples.
	protected Model trainRankSVM(){
		calcLabeledSimilarities();
		MyPriorityQueue<_RankItem> neighbors = new MyPriorityQueue<_RankItem>(m_topK);
		ArrayList<Feature[]> featureArray = new ArrayList<Feature[]>();
		ArrayList<Integer> targetArray = new ArrayList<Integer>();
		
		_Doc di, dj;
		int label;
		_RankItem ritm, ritn;
		for(int i=0; i<m_trainSet.size(); i++) {
			di = m_trainSet.get(i);
			for(int j=0; j<m_trainSet.size(); j++) {
				if (i==j)
					continue;	
				dj = m_trainSet.get(j);
				label = di.getYLabel() == dj.getYLabel() ? 1 : 0;
				neighbors.add(new _RankItem(j, m_LabeledCache[getIndex(i,j)], label));
			}
			
			for(int m=0; m<neighbors.size(); m++) {			
				ritm = neighbors.get(m);
				for(int n=m+1; n<neighbors.size(); n++) {
					ritn = neighbors.get(n);
					
					if (ritm.m_label > ritn.m_label) {
						featureArray.add(genRankingFV(m_trainSet.get(ritm.m_index), m_trainSet.get(ritn.m_index)));
						targetArray.add(1);
					} else if (ritm.m_label < ritn.m_label) {
						featureArray.add(genRankingFV(m_trainSet.get(ritn.m_index), m_trainSet.get(ritm.m_index)));
						targetArray.add(1);
					}
				}
			}
		}
		
		int fSize = 5;		
		return SVM.libSVMTrain(featureArray, targetArray, fSize, SolverType.L2R_L1LOSS_SVC_DUAL, 1.0, -1);
	}
	
	Feature[] genRankingFV(_Doc di, _Doc dj) {
		return null;
	}
}
