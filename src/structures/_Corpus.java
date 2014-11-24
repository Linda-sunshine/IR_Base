/**
 * 
 */
package structures;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

/**
 * @author lingong
 * General structure of corpus of a set of documents
 */
public class _Corpus {
	private ArrayList<_Doc> m_collection; //All the documentations in the corpus.
	private int m_size = 0; //The total documents the corpus has.
	protected int m_corClassNo = 0;
	// m_mask is used to do shuffle and its size is the total number of all the documents in the corpus.
	private int[] m_mask; 
	protected HashMap<Integer, Integer> m_classMemberNo;
			
	//Constructor.
	public _Corpus() {
		this.m_collection = new ArrayList<_Doc>();
		this.m_classMemberNo = new HashMap<Integer, Integer>();
 	}
	
	//Initialize the m_mask, the default value is false.
	public void setMasks() {
		this.m_mask = new int[this.m_collection.size()];
		for (int i = 0; i < this.m_collection.size(); i++) {
			this.m_mask[i] = 0;
		}
	}
	
	//Get all the documents of the corpus.
	public ArrayList<_Doc> getCollection(){
		return this.m_collection;
	}
	
	//Get the corpus's size, which is the total number of documents.
	public int getSize(){
		return this.m_size;
	}
	
	//The value of size is increased by one.
	public void sizeAddOne(){
		this.m_size++;
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
	public ArrayList<_Doc> addDoc(_Doc doc){
		this.m_collection.add(doc);
		return this.m_collection;
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
					writer.write(String.format(" %d:%f", fv.getIndex()+1, fv.getNormValue()));
				}
				writer.write('\n');
			}
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
}
