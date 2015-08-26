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
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Random;

/**
 * @author lingong
 * General structure of corpus of a set of documents
 */
public class _Corpus {
	static final int ReviewSizeCut = 3;
	
	ArrayList<_Doc> m_collection; //All the documents in the corpus.
	ArrayList<String> m_features; //ArrayList for features
	HashMap<String, _stat> m_featureStat; //statistics about the features
	boolean m_withContent = false; // by default all documents' content has been released
	
	public void setFeatureStat(HashMap<String, _stat> featureStat) {
		this.m_featureStat = featureStat;
	}

	// m_mask is used to do shuffle and its size is the total number of all the documents in the corpus.
	int[] m_mask; 
			
	//Constructor.
	public _Corpus() {
		this.m_collection = new ArrayList<_Doc>();
 	}
	
	public void reset() {
		m_collection.clear();
	}
	
	public void setContent(boolean content) {
		m_withContent = content;
	}
	
	public boolean hasContent() {
		return m_withContent;
	}
	
	public void setFeatures(ArrayList<String> features) {
		m_features = features;
	}
	
	public String getFeature(int i) {
		return m_features.get(i);
	}
	
	// used to call from AmazonReviewMain to pass this feature set to 
	// Naive Bayes classifier
	public ArrayList<String> getAllFeatures(){
		return m_features;
	}
	
	public int getFeatureSize() {
		return m_features.size();
	}
	
	public int getClassSize() {
		HashSet<Integer> labelSet = new HashSet<Integer>();
		for(_Doc d:m_collection)
			labelSet.add(d.getYLabel());
		return labelSet.size();
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
	
	public int getLargestSentenceSize()
	{
		int max = 0;
		for(int i=0; i<m_collection.size(); i++)
		{
			int length = m_collection.get(i).getSenetenceSize();
			if(length > max)
				max = length;
		}
		
		return max;
	}
	/*
	 rand.nextInt(k) will always generates a number between 0 ~ (k-1).
	 Access the documents with the masks can help us split the whole whole 
	 corpus into k folders. The function is used in cross validation.
	*/
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
		
	public int getItemSize() {
		String lastPID = null;
		int pid = 0, rSize = 0;
		for(_Doc d:m_collection) {
			if (lastPID==null)
				lastPID = d.getItemID();
			else if (!lastPID.equals(d.getItemID())) {
				//save co-occurrence of words in the reviews of this particular product
				if (rSize>ReviewSizeCut)
					pid ++;

				lastPID = d.getItemID();
				rSize = 0;
			}
			
			rSize++;
		}
		return pid+1;
	}
	
	public void mapLabels(int threshold) {
		int y;
		for(_Doc d:m_collection) {
			y = d.getYLabel();
			if (y<threshold)
				d.setYLabel(0);
			else
				d.setYLabel(1);
		}
	}
	
	public void save2File(String filename) {
		if (filename==null || filename.isEmpty()) {
			System.out.println("Please specify the file name to save the vectors!");
			return;
		}
		
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename)));
			for(_Doc doc:m_collection) {
				writer.write(String.format("%d", doc.getYLabel()));
				for(_SparseFeature fv:doc.getSparse()){
					writer.write(String.format(" %d:%f", fv.getIndex()+1, fv.getValue()));//index starts from 1
				}
				writer.write(String.format(" #%s-%s\n", doc.m_itemID, doc.m_name));//product ID and review ID
			}
			writer.close();
			
			System.out.format("%d feature vectors saved to %s\n", m_collection.size(), filename);
		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
	
	public void saveAs3WayTensor(String filename) {
		System.out.format("Save 3-way tensor to %s...\n", filename);
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "ASCII"));
		
			writer.write(String.format("%d\t%d\t%d\n", getItemSize(), getFeatureSize(), getFeatureSize()));
			ArrayList<_Doc> pReviews = new ArrayList<_Doc>();
			String lastPID = null;
			int pid = 0;
			for(_Doc d:m_collection) {
				if (lastPID==null)
					lastPID = d.getItemID();
				else if (!lastPID.equals(d.getItemID())) {
					//save co-occurrence of words in the reviews of this particular product
					if (pReviews.size()>ReviewSizeCut) {
						saveCoOccurrance2File(pReviews, pid, writer);
						pid ++;
					}
					lastPID = d.getItemID();
					pReviews.clear();
				}
				
				pReviews.add(d);
			}
			
			//for the last product
			if (pReviews.size()>ReviewSizeCut) 
				saveCoOccurrance2File(pReviews, pid, writer);
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	void saveCoOccurrance2File(ArrayList<_Doc> pReviews, int pid, BufferedWriter writer) throws IOException {
		HashMap<String, Integer> stats = new HashMap<String, Integer>();
		
		for(_Doc d:pReviews) {
			_SparseFeature[] fvs = d.getSparse();
			for(int i=0; i<fvs.length; i++) {
				for(int j=i+1; j<fvs.length; j++) {
					String key = getCoOccurranceKey(fvs[i].getIndex(), fvs[j].getIndex());
					if (stats.containsKey(key))
						stats.put(key, 1+stats.get(key));
					else
						stats.put(key, 1);
				}
			}
		}
		
		Iterator<Entry<String, Integer>> it = stats.entrySet().iterator();
		while(it.hasNext()) {
			Entry<String, Integer> entry = (Entry<String, Integer>)it.next();
			writer.write(String.format("%d\t%s\t%d\n", pid, entry.getKey(), entry.getValue()));
		}
	}
	
	String getCoOccurranceKey(int i, int j) {
		if (i<j)
			return String.format("%d\t%d", i, j);
		else
			return String.format("%d\t%d", j, i);
	}
}
