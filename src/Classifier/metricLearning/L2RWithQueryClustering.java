package Classifier.metricLearning;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import structures._Corpus;
import structures._Doc;
/***
 * Learning to rank for different cluster of queries.
 * @author lin
 *
 */
public class L2RWithQueryClustering extends L2RMetricLearning {
	
	int m_kmeans; //The number of clusters.
	Model[] m_rankSVMs; //Different rankSVM models for different clusters.
	HashMap<Integer, ArrayList<_Doc>> m_clusterNoDocs;
	
	public L2RWithQueryClustering(_Corpus c, String classifier, double C, 
								  double ratio, int k, int kPrime, double alhpa, double beta, 
								  double delta, double eta, boolean weightedAvg, int topK, 
								  double noiseRatio, int ranker, boolean multithread) {
		super(c, classifier, C, ratio, k, kPrime, alhpa, beta, delta, 
		      eta, weightedAvg, topK, noiseRatio, ranker, multithread);
		m_clusterNoDocs = new HashMap<Integer, ArrayList<_Doc>>();
	}
	
	//Set the number of clusters.
	public void setClusterNo(int k){
		m_kmeans = k;
	}
	
	public void train(Collection<_Doc> trainSet){
		
		super.init();
		m_classifier.train(m_trainSet);
		
		m_L = m_trainSet.size();
		m_U = m_testSet.size();
		m_labeled = m_trainSet;
		
		int clusterNo;
		
		//Init hashmap.
		for(int i=0; i<m_kmeans; i++)
			m_clusterNoDocs.put(i, new ArrayList<_Doc>());
		
		//Split the train set based on different clusters.
		for(_Doc d: trainSet){
			clusterNo = d.getCluseterNo();
			m_clusterNoDocs.get(clusterNo).add(d);
		}
		
		//The model array stores all the rankSVM for all clusters.
		m_rankSVMs = new Model[m_clusterNoDocs.size()];
		
		//Train different cluster of documents respectively.
		for(int cNo: m_clusterNoDocs.keySet()){
			m_trainSet = m_clusterNoDocs.get(cNo);
			//Get some stat of training reviews.
			int[] count = new int[2];
			for(_Doc d: m_trainSet)
				count[d.getYLabel()]++;
			System.out.format("There are %d (pos:%d, neg:%d) training documents in the corpus.\n", m_trainSet.size(), count[1], count[0]);
			L2RModelTraining();
			m_rankSVMs[cNo] = returnModel();
		}
	}
	
	//NOTE: this similarity is no longer symmetric!!
	@Override
	public double getSimilarity(_Doc di, _Doc dj) {
		
		double similarity = 0;
		int clusterNo = di.getCluseterNo();
		Model rankSVM = m_rankSVMs[clusterNo];
		if (m_ranker==0)
			similarity = Linear.predictValue(rankSVM, genRankingFV(di, dj), 0);
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
}
