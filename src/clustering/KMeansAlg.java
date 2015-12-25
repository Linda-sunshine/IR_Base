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
import structures._Corpus;
import structures._Doc;
import Classifier.BaseClassifier;
import cc.mallet.cluster.Clustering;
import cc.mallet.cluster.KMeans;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Metric;
import cc.mallet.types.SparseVector;

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
	ArrayList<ArrayList<_Doc>> m_clustersDocs;
//	ArrayList<ArrayList<Integer>> m_docsClusterNo; //Each arraylist contains the document indexes of one cluster.
	
	double[][] m_clusterStat;//added by Lin for basic stat.
	
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
	public void train(Collection<_Doc> trainSet) {
		init();
		
		for(_Doc d:trainSet)			
			m_instances.add(new Instance(createInstance(d), null, null, d));
		
		KMeans alg = new KMeans(m_instances.getPipe(), m_k, m_distance);
		Clustering result = alg.cluster(m_instances);	
		m_centroids = alg.getClusterMeans();
		m_clusters = result.getClusters();
		sortClustersBySize();
//		for(InstanceList cluster: m_clusters)
//			System.out.println(cluster.size());
	}
	
	public void sortClustersBySize(){
		int tmp = 0;
		for(int i=1; i<m_clusters.length; i++){
			tmp = i;
			for(int k=i-1; k>=0 && tmp >=0; k--){
				if(m_clusters[k].size() < m_clusters[tmp].size()){
					swap(k, tmp);
					tmp--;
				} else
					break;
		    }
		}
	}
	
	public void swap(int k, int l){
		InstanceList tmp = m_clusters[k];
		m_clusters[k] = m_clusters[l];
		m_clusters[l] = tmp;		
	}
	//added by Lin, transfer instances in cluster back to clusters of docs.
	public void tranferClusters2Docs(){
		m_clusterStat = new double[m_k][9];
		m_clustersDocs = new ArrayList<ArrayList<_Doc>>();
		for(int i=0; i < m_k; i++){
			double[] classNo = new double[10];
			ArrayList<_Doc> cluster = new ArrayList<_Doc>();
			for(Instance ins: m_clusters[i]){
				_Doc tmp = (_Doc) ins.getSource();
				classNo[tmp.getYLabel()]++;
				cluster.add(tmp);
			}
			classNo[5] = classNo[0] + classNo[1] + classNo[2] + classNo[3];//total negative documents, 0-3.
			classNo[6] = classNo[4];//totoal positive documents, 4.
			classNo[7] = classNo[5] + classNo[6]; //sum
			classNo[8] = classNo[6]/classNo[5];
			m_clusterStat[i] = classNo;
			m_clustersDocs.add(cluster);
		}
	}
	
	//Assign different cluster number to documents.
	public void setDocsClusterNo(){
		_Doc doc;
		ArrayList<_Doc> documents = m_corpus.getCollection();
//		m_docsClusterNo = new ArrayList<ArrayList<Integer>>();
		for(int i=0; i<m_k; i++){
			System.out.print(m_clusters[i].size() + "\t");
			for(Instance ins: m_clusters[i]){
				doc = (_Doc) ins.getSource();
				documents.get(doc.getID()).setClusterNo(i);
			}
		}
		System.out.println("\nFinish setting document cluster indexes.");
	}
	
//	public ArrayList<ArrayList<Integer>> getDocsClusterNo(){
//		return m_docsClusterNo;
//	}
	
	public ArrayList<ArrayList<_Doc>> getClustersDocs(){
		return m_clustersDocs;
	}
	
	//Write the cluster stat out to file, added by Lin.
	public void writeStat(String filename){
		try{
			PrintWriter printer = new PrintWriter(new File(filename));
			printer.write("class 0\tclass 1\tclass 2\tclass 3\tclass 4\tneg\tpos\tsum\tratio\n");
			for(double[] stat: m_clusterStat)
				printer.format("%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.3f\n", (int)stat[0], (int)stat[1], (int)stat[2], (int)stat[3], (int)stat[4], (int)stat[5], (int)stat[6], (int)stat[7], stat[8]);
//			for(SparseVector sp : m_centroids)
//				System.out.println(sp);
			printer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void writeContent(String filename){
		try{
			PrintWriter printer = new PrintWriter(new File(filename));
			for(int i=0; i< m_clustersDocs.size(); i++){
				printer.format("=======================%dst cluster========================\n", i);
				for(_Doc d: m_clustersDocs.get(i)){
					printer.write("---------------------------------------------------------\n");
					printer.write(d.getSource()+"\n");
				}
			}
			printer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
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
}
