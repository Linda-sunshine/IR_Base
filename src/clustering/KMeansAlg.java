/**
 * 
 */
package clustering;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;

import Classifier.BaseClassifier;
import cc.mallet.cluster.Clustering;
import cc.mallet.cluster.KMeans;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Metric;
import cc.mallet.types.SparseVector;
import structures._Corpus;
import structures._Doc;
import structures._Review;

/**
 * @author hongning
 *
 */
public class KMeansAlg extends BaseClassifier {

	Alphabet m_dict;
	InstanceList m_instances;
	InstanceList[] m_clusters;
	ArrayList<SparseVector> m_centroids;
	int m_k;
	Metric m_distance = new CosineDistance();
	
	public KMeansAlg(_Corpus c, int k) {
		super(c);
		m_k = k;
	}

	public KMeansAlg(int classNo, int featureSize, int k) {
		super(classNo, featureSize);
		m_k = k;
	}
	
	FeatureVector createInstance(_Doc d) {
		int[] indices = d.getIndices();
		for(int i:indices) {
			m_dict.lookupIndex(i, true);
		}
		return new FeatureVector(m_dict, indices, d.getValues());
	}
	
	@Override
	public double train(Collection<_Doc> trainSet) {
		init();
		
		for(_Doc d:trainSet)			
			m_instances.add(new Instance(createInstance(d), null, null, d));
		
		KMeans alg = new KMeans(m_instances.getPipe(), m_k, m_distance);
		Clustering result = alg.cluster(m_instances);	
		m_centroids = alg.getClusterMeans();
		m_clusters = result.getClusters();
		return 0; // we can compute the corresponding loss function
	}
	
	//assign to the closest cluster 
	@Override
	public int predict(_Doc doc) {
		int cid = 0;
		SparseVector spVct = createInstance(doc);
		double minDistance = m_distance.distance(m_centroids.get(0), spVct), dist;
		for(int i=1; i<m_centroids.size(); i++) {
			dist = m_distance.distance(m_centroids.get(i), spVct);
			if (dist<minDistance) {
				minDistance = dist;
				cid = i;
			}
		}
		return cid;
	}
	
	//distance to the corresponding cluster
	@Override
	public double score(_Doc d, int label) {
		if (label>=m_centroids.size())
			return -1;
		
		return m_distance.distance(m_centroids.get(label), createInstance(d));
	}

	@Override
	protected void init() {
		m_dict = new Alphabet();
		m_instances = new InstanceList(m_dict, null);
	}

	@Override
	protected void debug(_Doc d) {
		// TODO Auto-generated method stub
	}

	@Override
	public void saveModel(String modelLocation) {
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(modelLocation), "UTF-8"));
			for(SparseVector spVct:m_centroids) {
				int[] indices = spVct.getIndices();
				double[] values = spVct.getValues();
				for(int i=0; i<indices.length; i++) {
					writer.write(String.format("%d:%.5f ", indices[i], values[i]));
				}
				writer.write("\n");
			}
			
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	// Added by Lin for experimental purpose.
	public void writeResults(String filename, HashSet<String> ctgs){
		PrintWriter writer;
		try{
			double sum;
			writer = new PrintWriter(new File(filename));
			for(String s: ctgs)
				writer.write(s+"\t");
			writer.write("\n");
			for(InstanceList ls: m_clusters){
				sum = 0;
				initCtg(ctgs);
				organizeCtg(ls);
				for(String s: m_ctgCount.keySet())
					sum += m_ctgCount.get(s);
				for(String s: m_ctgCount.keySet())
					writer.write(m_ctgCount.get(s)/sum+"\t");
				writer.write("\n");
			}
			writer.close();
		} catch (IOException e){
			e.printStackTrace();
		}
	}
	
	public double trainKmeans(Collection<_Review> trainSet) {
		init();
		
		for(_Review d: trainSet)			
			m_instances.add(new Instance(createInstance(d), null, null, d));
		
		KMeans alg = new KMeans(m_instances.getPipe(), m_k, m_distance);
		Clustering result = alg.cluster(m_instances);	
		m_centroids = alg.getClusterMeans();
		m_clusters = result.getClusters();
		return 0; // we can compute the corresponding loss function
	}
	
	HashMap<String, Integer> m_ctgCount = new HashMap<String, Integer>();
	public void initCtg(HashSet<String> ctgs){
		for(String s: ctgs)
			m_ctgCount.put(s, 0);
	}
	public void organizeCtg(InstanceList ls){
		String key;
		for(Instance l: ls){
			key = ((_Review)l.getSource()).getCategory();
			m_ctgCount.put(key, m_ctgCount.get(key)+1);
		}
	}
}