/**
 * 
 */
package Analyzer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import structures._stat;
import utils.Utils;

/**
 * @author hongning
 * Keyword based bootstrapping for aspect annotation
 */
public class AspectAnalyzer extends jsonAnalyzer {
	class _Aspect{
		String m_name;
		HashSet<Integer> m_keywords; // index corresponding to the controlled vocabulary
		MyPriorityQueue<_RankItem> m_candidates; // candidate words to expand the aspect keyword list
		
		_Aspect(String name, HashSet<Integer> keywords, int chisize){
			m_name = name;
			m_keywords = keywords;
			m_candidates = new MyPriorityQueue<_RankItem>(chisize);
		}
		
		public void addCandidateKeyword(int wid, double value) {
			m_candidates.add(new _RankItem(wid, value));
		}
		
		public void clearCandidateKeywords() {
			m_candidates.clear();
		}
		
		public boolean expandKeywords() {
			boolean expand = false;
			for(_RankItem it:m_candidates)
				expand |= m_keywords.add(it.m_index);
			return expand;
		}
	}
	int m_aspDimension; //The number of aspects.
	int m_chiSize; // top words to be added to aspect keyword list 
	ArrayList<_Aspect> m_aspects; // a list of aspects specified by keywords
	int[] m_aspectDist; // distribution of aspects (count in DF)
	int m_count;
	boolean m_topicFlag;
	
	public AspectAnalyzer(String tokenModel, String stnModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, stnModel);
	}
	
	public AspectAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String aspectFile, int chiSize, boolean topicFlag)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		//public jsonAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String stnModel)
		//public jsonAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		
		m_chiSize = chiSize;
		LoadAspectKeywords(aspectFile);
		m_count = 0;
		m_topicFlag = topicFlag;
	}
	
	public AspectAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String aspectFile, int chiSize) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_chiSize = chiSize;
		LoadAspectKeywords(aspectFile);
		m_count = 0;
		m_topicFlag = false;

	}

	public void LoadAspectKeywords(String filename){
		try {
			m_aspects = new ArrayList<_Aspect>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			String[] container;
			HashSet<Integer> keywords;
			while( (tmpTxt=reader.readLine()) != null ){
				container = tmpTxt.split("\\s+");
				keywords = new HashSet<Integer>(container.length-1);

				for(int i=1; i<container.length; i++) {
					if (m_featureNameIndex.containsKey(container[i]))
						keywords.add(m_featureNameIndex.get(container[i])); // map it to a controlled vocabulary term
				}
				
				m_aspects.add(new _Aspect(container[0], keywords, m_chiSize));
				System.out.println("Keywords for " + container[0] + ": " + keywords.size());
			}
			reader.close();
			m_aspDimension = m_aspects.size();
			m_aspectDist = new int[m_aspects.size()];
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	void Annotate(_Doc d){
		for(int i=0; i<d.getSenetenceSize(); i++){
			int maxCount = 0, count, sel = -1;
			_Stn s = d.getSentence(i);
			for(int index=0; index<m_aspects.size(); index++){				
				if ( (count=s.AnnotateByKeyword(m_aspects.get(index).m_keywords))>maxCount ){
					maxCount = count;
					sel = index;
				}
				else if (count==maxCount)
					sel = -1;//how should we break the tie?
			}
			s.setTopic(sel);
		}
	}
	
	void collectStats(_Doc d){
		int aspectID, wordID;
		for(_Stn s:d.getSentences()){
			if ( (aspectID=s.getTopic())>-1 ){//if it is annotated
				for(_SparseFeature f:s.getFv()){
					wordID = f.getIndex();
					m_featureStat.get(m_featureNames.get(wordID)).addOneDF(aspectID);
				}
				m_aspectDist[aspectID] ++;
			}			
		}
	}
	
	void clearFvStats() {
		Iterator<Map.Entry<String, _stat>> it = m_featureStat.entrySet().iterator();
		int aspectSize = m_aspects.size();
	    while (it.hasNext()) {
	        Map.Entry<String, _stat> pair = (Map.Entry<String, _stat>)it.next();
	        pair.getValue().reset(aspectSize);
	    }
	    
	    //clear DF statistics for aspects
	    Arrays.fill(m_aspectDist, 0);
	    
	    //clear aspect candidate keywords
	    for(_Aspect aspect:m_aspects)
	    	aspect.clearCandidateKeywords();
	}
	
	void ChiSquareTest(){	
		clearFvStats();
		for(_Doc d:m_corpus.getCollection()){
			Annotate(d);
			collectStats(d);
		}
	}
	
	boolean expandKeywordsByChi(double ratio){
		int selID = -1, aspectSize = m_aspects.size(), N = Utils.sumOfArray(m_aspectDist), DF;
		double maxChi, chiV;
		int[] DFarray;
		_Aspect aspect;
		
		Iterator<Map.Entry<String, _stat>> it = m_featureStat.entrySet().iterator();
		while(it.hasNext()){//set aspect assignment for each word
			Map.Entry<String, _stat> entry = it.next();
			
			_stat temp = entry.getValue();
			DFarray = temp.getDF();
			DF = Utils.sumOfArray(DFarray);
			
			maxChi = 0.0;
			selID = -1;
			for(int i=0; i<aspectSize; i++){				
				chiV = FeatureSelector.ChiSquare(N, DF, DFarray[i], m_aspectDist[i]);				
				if (chiV > ratio * maxChi){
					maxChi = chiV;
					selID = i;
				}
			}
			
			if (selID>-1) { // expand candidate keyword list in the selected aspect
				aspect = m_aspects.get(selID);
				aspect.addCandidateKeyword(m_featureNameIndex.get(entry.getKey()), maxChi);
			}
		}
		
		boolean extended = false;
		for(int i=0; i<aspectSize; i++){//expand each aspect accordingly
			aspect = m_aspects.get(i);
			extended |= aspect.expandKeywords();
		}
		return extended;
	}
	
	public void BootStrapping(String aspectFile, String filename, int chi_size, double chi_ratio, int chi_iter){
		m_chiSize = chi_size;
		System.out.println("Vocabulary size: " + m_featureNames.size());
		LoadAspectKeywords(aspectFile);//load aspects first
		
		int iter = 0;
		do {
			ChiSquareTest();
			System.out.println("Bootstrapping for " + iter + " iterations...");
		}while(expandKeywordsByChi(chi_ratio) && ++iter<chi_iter);
		
		try {
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			for(int i=0; i<m_aspects.size(); i++){
				_Aspect asp = m_aspects.get(i);
				writer.write(asp.m_name);
				Iterator<Integer> wIter = asp.m_keywords.iterator();
				while(wIter.hasNext())
					writer.write(" " + m_featureNames.get(wIter.next()));
				writer.write("\n");
			}
			writer.close();
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	protected boolean AnalyzeDoc(_Doc doc) {
		String[] tokens = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		int index = 0;
		double value = 0;
		// Construct the sparse vector.
		for (String token : tokens) {
			// CV is not loaded, take all the tokens as features.
			if (!m_isCVLoaded) {
				if (m_featureNameIndex.containsKey(token)) {
					index = m_featureNameIndex.get(token);
					if (spVct.containsKey(index)) {
						value = spVct.get(index) + 1;
						spVct.put(index, value);
					} else {
						spVct.put(index, 1.0);
						m_featureStat.get(token).addOneDF(doc.getYLabel());
					}
				} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
					expandVocabulary(token);// update the m_featureNames.
					index = m_featureNameIndex.get(token);
					spVct.put(index, 1.0);
					m_featureStat.get(token).addOneDF(doc.getYLabel());
				}
				m_featureStat.get(token).addOneTTF(doc.getYLabel());
			} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
				index = m_featureNameIndex.get(token);
				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);
					m_featureStat.get(token).addOneDF(doc.getYLabel());
				}
				m_featureStat.get(token).addOneTTF(doc.getYLabel());
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		
		if (spVct.size()>=m_lengthThreshold) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
//			if(m_topicFlag){//This is for aspect annotation with given aspects and keywords.
//				double[] tmp = detectAspects(spVct);
//				if(NotEmpty(tmp)){
//					m_count++;
//					doc.setAspVct(tmp);
////					m_corpus.addDoc(doc);
//				}
//			}
			m_corpus.addDoc(doc);
			m_classMemberNo[doc.getYLabel()]++;
//			if (m_releaseContent)
//				doc.clearSource();
			return true;
		} else
			return false;
	}
	
	public int returnCount(){
		return m_count;
	}
	
	public double[] detectAspects(HashMap<Integer, Double> spVct){
		double[] aspVct = new double[m_aspDimension];
		for(int i = 0; i < m_aspects.size(); i++){
			HashSet<Integer> keywords = m_aspects.get(i).m_keywords;
			for(int key: keywords){
				if(spVct.containsKey(key))
					aspVct[i] = 1.0;
				break;
			}
		}
		return aspVct;
	}
	
	public boolean NotEmpty(double[] aspVct){
		int sum = 0;
		for(double a: aspVct){
			sum += a * 1.0;
		}
		if(sum != 0)
			return true;
		else 
			return false;
	}
	
//	//Set the topic vector for every document.
//	public void setTopicVector(double[][] ttp){
//		for(_Doc d: m_corpus.getCollection()){
//			double[] topicVector = new double[ttp.length];
//			for(int i=0; i < ttp.length; i++){
//				for(_SparseFeature sf: d.getSparse()){
//					int index = sf.getIndex();
//					topicVector[i] += ttp[i][index] * sf.getValue();
//				}
//			}
//			d.setAspVct(topicVector);
//		}
//	}
	
//	public static void main(String[] args){
//		int[] a = {1, 0, 0, 1};
//		int[] b = {0, 0, 0};
//		int[] c = {1, 0, 0, 1, 0};
//		System.out.println(isEmpty(a));
//		System.out.println(isEmpty(b));
//		System.out.println(isEmpty(c));
//	}
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException {
		int classNumber = 5; //Define the number of classes
		int Ngram = 2; //The default value is bigram. 
		int lengthThreshold = 10; //Document length threshold
		
//		/*****Parameters in feature selection.*****/
		String featureSelection = "DF"; //Feature selection method.
		int chiSize = 50; // top ChiSquare words for aspect keyword selection
		String stopwords = "./data/Model/stopwords.dat";
		double startProb = 0.2; // Used in feature selection, the starting point of the features.
		double endProb = 0.999; // Used in feature selection, the ending point of the features.
		int DFthreshold = 20; // Filter the features with DFs smaller than this threshold.
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/small";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model
		String stnModel = "./data/Model/en-sent.bin"; //Sentence model
		String aspectModel = "./data/Model/aspect_tablet.txt"; // list of keywords in each aspect
		String aspectOutput = "./data/Model/aspect_output.txt"; // list of keywords in each aspect
		
		String pattern = String.format("%dgram_%s", Ngram, featureSelection);
		String fvFile = String.format("data/Features/fv_%s_small.txt", pattern);
		String fvStatFile = String.format("data/Features/fv_stat_%s_small.txt", pattern);
		
		/****Loading json files*****/
//		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, null, Ngram, lengthThreshold);
		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, fvFile, Ngram, lengthThreshold);
		analyzer.LoadStopwords(stopwords);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		
//		/****Feature selection*****/
//		System.out.println("Performing feature selection, wait...");
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
//		analyzer.SaveCVStat(fvStatFile);
		
		/****Aspect annotation*****/
		analyzer.BootStrapping(aspectModel, aspectOutput, chiSize, 0.9, 10);
	}
}
