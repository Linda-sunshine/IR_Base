/**
 * 
 */
package structures;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Random;

/**
 * @author lingong
 * General structure of corpus of a set of documents
 */
public class _Corpus {
	ArrayList<_Doc> m_collection; //All the documents in the corpus.
	ArrayList<String> m_features; //ArrayList for features
	
	// m_mask is used to do shuffle and its size is the total number of all the documents in the corpus.
	int[] m_mask; 
			
	//Constructor.
	public _Corpus() {
		this.m_collection = new ArrayList<_Doc>();
 	}
	
	public void reset() {
		m_collection.clear();
	}
	
	public void setFeatures(ArrayList<String> features) {
		m_features = features;
	}
	
	public String getFeature(int i) {
		return m_features.get(i);
	}
	
	public int getFeatureSize() {
		return m_features.size();
	}
	
	//Initialize the m_mask, the default value is false.
	public void setMasks() {
		this.m_mask = new int[this.m_collection.size()];
	}
	
	//Get all the documents of the corpus.
	public ArrayList<_Doc> getCollection(){
		return this.m_collection;
	}
	
	//Get the corpus's size, which is the total number of documents.
	public int getSize(){
		return m_collection.size();
	}
	
	//Get the total count of all the tokens in the corpus 
	public int getCorpusTotalLenght()
	{
		int size = 0;
		for(int i=0; i < m_collection.size(); i++)
		{
			size = size + m_collection.get(i).getTotalDocLength();
		}
		return size;
	}
	
	/*
	 rand.nextInt(k) will always generates a number between 0 ~ (k-1).
	 Access the documents with the masks can help us split the whole whole 
	 corpus into k folders. The function is used in cross validation.
	*/
	//Why this method is defined as private??? It will be used in Classifier.
	public void shuffle(int k) {
		Random rand = new Random();
		for(int i=0; i< m_mask.length; i++) {
			this.m_mask[i] = rand.nextInt(k);
		}
	}
	
	//Add a new doc to the corpus.
	public void addDoc(_Doc doc){
		m_collection.add(doc);
	}
	public void removeDoc(int index){
		m_collection.remove(index);
	}
	
	//Get the mask array of the corpus.
	public int[] getMasks(){
		return this.m_mask;
	}
	
	public void save2File(String filename) {
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
			for(_Doc doc:m_collection) {
				writer.write(String.format("%d", doc.getYLabel()));
				for(_SparseFeature fv:doc.getSparse()){
					writer.write(String.format(" %d:%f", fv.getIndex()+1, fv.getValue()));//index starts from 1
				}
				writer.write('\n');
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
}
