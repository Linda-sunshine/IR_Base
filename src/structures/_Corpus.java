/**
 * 
 */
package structures;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Random;

import utils.Utils;

/**
 * @author lingong
 * General structure of corpus of a set of documents
 */
public class _Corpus {
	static final int ReviewSizeCut = 3;
	ArrayList<_Doc> m_collection; //All the documents in the corpus.
	ArrayList<String> m_features; //ArrayList for features
	HashMap<String, _stat> m_featureStat; //statistics about the features
	
	
	public SentiWordNetDemoCode sentiWordNet;
	public ArrayList<String> m_posPriorList;
	public ArrayList<String> m_negPriorList;
	public ArrayList<String> m_negationList;
	
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
	
	public void setFeatures(ArrayList<String> features) {
		m_features = features;
	}
	
	public String getFeature(int i) {
		return m_features.get(i);
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
	
	public void setStnFeatures() {
		for(_Doc d:m_collection) {
			d.setSentenceFeatureVector();
		}
	}
	
	public void loadPriorPosNegWords(String pathToSentiWordNet, String pathToPosWords, String pathToNegWords,String pathToNegationWords)
	{
		m_posPriorList = new ArrayList<String>();
		m_negPriorList = new ArrayList<String>();
		m_negationList = new ArrayList<String>();
		
		BufferedReader file = null;
		try {
			file = new BufferedReader(new FileReader(pathToPosWords));
			String line;
			while ((line = file.readLine()) != null) {
				m_posPriorList.add(line);
			}
			file.close();
			
			file = new BufferedReader(new FileReader(pathToNegWords));
			while ((line = file.readLine()) != null) {
				m_negPriorList.add(line);
			}
			file.close();
			
			file = new BufferedReader(new FileReader(pathToNegationWords));
			while ((line = file.readLine()) != null) {
				m_negationList.add(line);
			}
			file.close();
		} catch (Exception e) {
			e.printStackTrace();
		}

		// loading the sentiWordnet
		try{
			sentiWordNet = new SentiWordNetDemoCode(pathToSentiWordNet);
		}
		catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public void setStnFeaturesForSentiment(String pathToSentiWordNet, String pathToPosWords, String pathToNegWords,String pathToNegationWords) {
		// load all prior pos & neg words and also the sentiwordNet
		loadPriorPosNegWords(pathToSentiWordNet, pathToPosWords, pathToNegWords, pathToNegationWords);
		
		for(_Doc d:m_collection)
			setSentenceFeatureVectorForSentiment(d);
	}
	
	
	// used by LR-HTSM for constructing transition features for sentiment
	public void setSentenceFeatureVectorForSentiment(_Doc d) {
		_Stn[] sentences = d.getSentences();
		
		// start from 2nd sentence
		double pSim = Utils.cosine(sentences[0].getFv(), sentences[1].getFv()), nSim;
		double pKL = Utils.klDivergence(calculatePOStagVector(sentences[0]), calculatePOStagVector(sentences[1])), nKL;
		double pSenScore = sentiWordScore(sentences[0]), cSenScore;
		int pPosNeg= posNegCount(sentences[0]), cPosNeg;
		int pNegationCount= negationCount(sentences[0]), cNegationCount;
		int stnSize = d.getSenetenceSize();
		
		for(int i=1; i<stnSize; i++){
			//cosine similarity			
			sentences[i-1].m_sentiTransitFv[0] = pSim;			

			//sentiWordScore
			cSenScore = sentiWordScore(sentences[i]);
			if((cSenScore<0 && pSenScore>0) || (cSenScore>0 && pSenScore<0))
				sentences[i-1].m_sentiTransitFv[1] = 1; // transition
			else if((cSenScore<=0 && pSenScore<=0) || (cSenScore>=0 && pSenScore>=0))
				sentences[i-1].m_sentiTransitFv[1] = -1; // no transition
			pSenScore = cSenScore;

			//positive/negative count 
			cPosNeg = posNegCount(sentences[i]);
			if(pPosNeg==cPosNeg)
				sentences[i-1].m_sentiTransitFv[2] = -1; // no transition
			else if (pPosNeg!=cPosNeg)
				sentences[i-1].m_sentiTransitFv[2] = 1; // transition
			pPosNeg = cPosNeg;

			//similar to previous or next
			if (i<stnSize-1) {
				nSim = Utils.cosine(sentences[i].getFv(), sentences[i+1].getFv());
				if (nSim>pSim)
					sentences[i-1].m_sentiTransitFv[3] = 1;
				else if (nSim<pSim)
					sentences[i-1].m_sentiTransitFv[3] = -1;
				pSim = nSim;
			}

			//similar to previous or next
			if (i<stnSize-1) {
				nKL = Utils.klDivergence(calculatePOStagVector(sentences[i]), calculatePOStagVector(sentences[i+1]));
				if (nKL>pKL)
					sentences[i-1].m_sentiTransitFv[4] = 1;
				else if (nKL<pKL)
					sentences[i-1].m_sentiTransitFv[4] = -1;
				pKL = nKL;
			}

			//positive negative count 
			cNegationCount = negationCount(sentences[i]);
			if(pNegationCount==0 && cNegationCount>0)
				sentences[i-1].m_sentiTransitFv[5] = 1; // transition
			else if (pNegationCount>0 && cNegationCount==0)
				sentences[i-1].m_sentiTransitFv[5] = 1; // transition
			else
				sentences[i-1].m_sentiTransitFv[5] = -1; // no transition
			pNegationCount = cNegationCount;
		}
	}

	// receive sentence index as parameter
	public double sentiWordScore(_Stn s) {
		_SparseFeature[] wordsInSentence = s.getFv();
		int index;
		String token;
		double senScore = 0.0;
		double tmp;

		for(_SparseFeature word:wordsInSentence){
			index = word.getIndex();
			token = m_features.get(index);
			tmp = sentiWordNet.extract(token, "n");
			if(tmp!=-2) // word found in SentiWordNet
				senScore+=tmp;
		}
		return senScore/wordsInSentence.length;
	}

	// receive sentence index as parameter
	public int posNegCount(_Stn s) {
		_SparseFeature[] wordsInSentence = s.getFv();
		int index;
		String token;
		int posCount = 0;
		int negCount = 0;

		for(_SparseFeature word:wordsInSentence){
			index = word.getIndex();
			token = m_features.get(index);
			if(m_posPriorList.contains(token))
				posCount++;
			else if(m_negPriorList.contains(token))
				negCount++;
		}

		if(posCount>negCount)
			return 1; // 1 means sentence is more positive
		else if (negCount>posCount)
			return 2; // 2 means sentence is more negative
		else
			return 0; // sentence is neutral or no match
	}

	// receive sentence index as parameter
	public int negationCount(_Stn s) {
		_SparseFeature[] wordsInSentence = s.getFv();
		int index;
		String token;
		int negationCount = 0;

		for(_SparseFeature word:wordsInSentence){
			index = word.getIndex();
			token = m_features.get(index);
			if(m_negationList.contains(token))
				negationCount++;
		}
		return negationCount;
	}

	// calculate the number of Noun, Adjectives, Verb & AdVerb in a vector for a sentence
	// here i the index of the sentence
	public double[] calculatePOStagVector(_Stn s) {
		String[] posTag = s.getSentencePosTag();
		double tagVector[] = new double[4]; 
		// index = 0 for noun
		// index = 1 for adjective
		// index = 2 for verb
		// index = 3 for adverb
		tagVector[0]= tagVector[1] = tagVector[2]  = tagVector[3] = 0.0;
		for(String tag:posTag){
			if(tag.equalsIgnoreCase("NN") || tag.equalsIgnoreCase("NNS") || tag.equalsIgnoreCase("NNP") || tag.equalsIgnoreCase("NNPS"))
				tagVector[0]++;
			else if(tag.equalsIgnoreCase("JJ") || tag.equalsIgnoreCase("JJR") || tag.equalsIgnoreCase("JJS"))
				tagVector[1]++;
			else if(tag.equalsIgnoreCase("VB") || tag.equalsIgnoreCase("VBD") || tag.equalsIgnoreCase("VBG"))
				tagVector[2]++;
			else if(tag.equalsIgnoreCase("RB") || tag.equalsIgnoreCase("RBR") || tag.equalsIgnoreCase("RBS"))
				tagVector[3]++;
		}
		return tagVector;
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
