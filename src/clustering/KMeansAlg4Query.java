package clustering;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;

import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import structures._Corpus;
import structures._Doc;

/**
 * Overwrite the kmeans algorithm by considering each document as a query and new features.
 * @author lin
 */
public class KMeansAlg4Query extends KMeansAlg{
	
	int m_dim;
	double[][] m_clusterStat; // Added by Lin for basic stat.

	ArrayList<ArrayList<_Doc>> m_clustersDocs; // Added by Lin for storing clustered documents.
	ArrayList<ArrayList<Integer>> m_docsClusterNo; // Added by Lin, each arraylist contains the document indexes of one cluster.
	
	public KMeansAlg4Query(_Corpus c, int k, int dim) {
		super(c, k);
		m_dim = dim;
	}
	
	/***Overwrite the create instance method
	 * indices: 0 - the number of features.
	 * values: corresponding values. */
	FeatureVector createInstance(_Doc d) {
		d.setQueryDim(m_dim);
		d.setQueryValues();
		int[] indices = d.getQueryIndices();
		for(int i:indices) {
			m_dict.lookupIndex(i, true);
		}
		return new FeatureVector(m_dict, indices, d.getQueryValues());
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
		
	public ArrayList<ArrayList<Integer>> getDocsClusterNo(){
		return m_docsClusterNo;
	}
		
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
					printer.format("-----------------------label: %d-----------------------------------\n", d.getYLabel());
					printer.write(d.getSource()+"\n");
				}
			}
			printer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
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

	@Override
	public double train(Collection<_Doc> trainSet) {
		super.train(trainSet);
		sortClustersBySize();
//		for(InstanceList cluster: m_clusters)
//			System.out.println(cluster.size());
		return 0;
	}
}
