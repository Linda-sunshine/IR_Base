package clustering;

import java.util.ArrayList;
import cc.mallet.cluster.Clustering;
import cc.mallet.cluster.KMeans;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Metric;
import cc.mallet.types.SparseVector;

public class KMeansAlg4Vct {
	Alphabet m_dict;
	InstanceList m_instances;
	InstanceList[] m_clusters;
	ArrayList<SparseVector> m_centroids;
	
	int m_k, m_featureSize;
	int[] m_indices; 
	Metric m_distance = new CosineDistance();
	double[][] m_weights;
	fvInstance[] m_fvInstances;
	
	class fvInstance{
		int m_index;
		double[] m_x;
		public fvInstance(int in, double[] x){
			m_index = in;
			m_x = x;
		}
	}
	
	public KMeansAlg4Vct(double[][] weights, int k){
		m_weights = weights;
		m_featureSize = m_weights.length;
		m_k = k;
	}
	
	public void init() {
		m_dict = new Alphabet();
		m_instances = new InstanceList(m_dict, null);
		m_indices = new int[m_featureSize];
		for(int i=0; i<m_featureSize; i++)
			m_indices[i] = i;
	}
	
	public void initInstances(){
		m_fvInstances = new fvInstance[m_weights[0].length];
		for(int i=0; i<m_weights[0].length; i++){
			m_fvInstances[i] = new fvInstance(i, getOneColumn(i));
		}
	}
	
	FeatureVector createInstance(fvInstance ins) {
		for(int i: m_indices)
			m_dict.lookupIndex(i, true);
		return new FeatureVector(m_dict, m_indices, ins.m_x);
	}
	
	public void setWeights(double[][] ws){
		m_weights = ws;
	}
	
	// The given weight is kFold*v matrix with each column being one x for kmeans training.
	public double train() {
		initInstances();
		for(int i=0; i<m_weights[0].length; i++)
			m_instances.add(new Instance(createInstance(m_fvInstances[i]), null, null, m_fvInstances[i]));
		
		KMeans alg = new KMeans(m_instances.getPipe(), m_k, m_distance);
		Clustering result = alg.cluster(m_instances);	
		m_centroids = alg.getClusterMeans();
		m_clusters = result.getClusters();
		return 0; // we can compute the corresponding loss function
	}
	
	public double[] getOneColumn(int col){
		double[] column = new double[m_weights.length];
		for(int i=0; i < m_weights.length; i++){
			column[i] = m_weights[i][col];
		}
		return column;
	}
	
	// Return the corresponding cluster numbers.
	public int[] getClusters(){
		int[] clusterNos = new int[m_instances.size()];
		fvInstance tmp;
		for(int i=0; i<m_k; i++){
			for(Instance ins: m_clusters[i]){
				tmp = (fvInstance)ins.getSource();
				clusterNos[tmp.m_index] = i;
			}
		}
		return clusterNos;
	}
}
