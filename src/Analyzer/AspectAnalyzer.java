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
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;

import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures.Post;
import structures.Product;
import structures.TokenizeResult;
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
	int m_count=0;
	boolean m_aspFlag=false;
		
	public AspectAnalyzer(String tokenModel, String stnModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, stnModel);
	}
	
	public AspectAnalyzer(String tokenModel, String stnModel, int classNo, String providedCV, int Ngram, int threshold, String tagModel, String aspectFile, boolean aspFlag) throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, stnModel, tagModel);
		LoadAspectKeywords(aspectFile);
		m_aspFlag = aspFlag;
	}
	
	public AspectAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String aspectFile, int chiSize, boolean aspFlag)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		m_chiSize = chiSize;
		LoadAspectKeywords(aspectFile);
		m_aspFlag = aspFlag;
	}
	
//	public AspectAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold, String aspectFile, int chiSize) throws InvalidFormatException, FileNotFoundException, IOException{
//		super(tokenModel, classNo, providedCV, Ngram, threshold);
//		m_chiSize = chiSize;
//		LoadAspectKeywords(aspectFile);
//		m_count = 0;
//		m_topicFlag = false;
//
//	}

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
	
	public int returnCount(){
		return m_count;
	}
	
	public double[] detectAspects(HashMap<Integer, Double> spVct){
		double[] aspVct = new double[m_aspDimension];
		for(int i = 0; i < m_aspects.size(); i++){
			HashSet<Integer> keywords = m_aspects.get(i).m_keywords;
			for(int key: keywords){
				if(m_aspFlag){
					if(spVct.containsKey(key))
						aspVct[i] += spVct.get(key);
				} else{
					if(spVct.containsKey(key))
						aspVct[i] = 1;
					break;	
				}
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
	
	//Analyze document with POS Tagging, set postagging sparse vector and senti score.
	protected boolean AnalyzeDocWithPOSTagging(_Doc doc) {
		
		TokenizeResult result;
		String[] sentences = m_stnDetector.sentDetect(doc.getSource());
		int y = doc.getYLabel();
		
		double sentiScore = 0, count= 0;
		HashMap<Integer, Double> posTaggingVct = new HashMap<Integer, Double>();//Collect the index and counts of projected features.	
		
		result = TokenizerNormalizeStemmer(doc.getSource());
		String[] tokens = result.getTokens();
		doc.setStopwordProportion(result.getStopwordProportion());
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
		doc.setAspVct(detectAspects(spVct));
		
		for(String sentence: sentences){
			int posIndex = 0;
			double posValue = 0;
			String[] posTokens = Tokenizer(sentence);
			String[] tags = m_tagger.tag(posTokens);
			HashMap<Integer, Double> postaggingSentenceVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
			
			for(int i = 0; i < posTokens.length; i++){
				String tmpToken = SnowballStemming(Normalize(posTokens[i]));
				if (isLegit(tmpToken)){
					//If the word is adj/adv, construct the sparse vector.
					if(tags[i].equals("RB")||tags[i].equals("RBR")||tags[i].equals("RBS")||tags[i].equals("JJ")||tags[i].equals("JJR")||tags[i].equals("JJS")){
						if(m_posTaggingFeatureNameIndex.containsKey(tmpToken)){
							posIndex = m_posTaggingFeatureNameIndex.get(tmpToken);
							if(postaggingSentenceVct.containsKey(posIndex)){
								posValue = postaggingSentenceVct.get(posIndex) + 1;
								postaggingSentenceVct.put(posIndex, posValue);
							} else
								postaggingSentenceVct.put(posIndex, 1.0);
						} else{
							posIndex = m_posTaggingFeatureNameIndex.size();
							m_posTaggingFeatureNameIndex.put(tmpToken, posIndex);
							postaggingSentenceVct.put(posIndex, 1.0);
						}
					}
					//If the word is in sentiwordnet, accumulate the score.
					if(tags[i].equals("RB")||tags[i].equals("RBR")||tags[i].equals("RBS")){
						tmpToken = posTokens[i] + "#r";
					} else if (tags[i].equals("JJ")||tags[i].equals("JJR")||tags[i].equals("JJS")){
						tmpToken = posTokens[i] + "#a";
					} else if (tags[i].equals("NN")||tags[i].equals("NNS")||tags[i].equals("NNP")||tags[i].equals("NNPS")){
						tmpToken = posTokens[i] + "#n";
					} else if (tags[i].equals("VB")||tags[i].equals("VBD")||tags[i].equals("VBG")||tags[i].equals("VBN")||tags[i].equals("VBP")||tags[i].equals("VBZ")){
						tmpToken = posTokens[i] + "#v";
					} 
					if(m_sentiwordScoreMap.containsKey(tmpToken)){
						sentiScore += m_sentiwordScoreMap.get(tmpToken);
						count++;
					}
				}
			}
			if (postaggingSentenceVct.size() > 0) //avoid empty sentence
				Utils.mergeVectors(postaggingSentenceVct, posTaggingVct);
		}

		//the document should be long enough
		if (spVct.size()>=m_lengthThreshold) { 
			doc.createSpVct(spVct);
			doc.createPOSVct(posTaggingVct);
			
			if(count == 0)
				doc.setSentiScore(0);
			else
				doc.setSentiScore(sentiScore/count);

			m_corpus.addDoc(doc);
			m_classMemberNo[y]++;
					
//			if (m_releaseContent)
//			doc.clearSource();
			return true;
		} else {
			/****Roll back here!!******/
			rollBack(spVct, y);
			return false;
		}
	}
	//previous LoadDoc, in case we need it.
	public void LoadDoc(String filename) {
		Product prod = null;
		JSONArray jarray = null;

		try {
			JSONObject json = LoadJson(filename);
			prod = new Product(json.getJSONObject("ProductInfo"));
			jarray = json.getJSONArray("Reviews");
		} catch (Exception e) {
			System.out.print('X');
			return;
		}

		for (int i = 0; i < jarray.length(); i++) {
			try {
				Post post = new Post(jarray.getJSONObject(i));
				if (checkPostFormat(post)) {
					long timeStamp = m_dateFormatter.parse(post.getDate())
							.getTime();
					String content;
					if (Utils.endWithPunct(post.getTitle()))
						content = post.getTitle() + " " + post.getContent();
					else
						content = post.getTitle() + ". " + post.getContent();
					// int label = 0;
					// if(post.getLabel()>=4) label = 1;
					// _Doc review = new _Doc(m_corpus.getSize(), post.getID(), post.getTitle(), prod.getID(), label, timeStamp);
					_Doc review = new _Doc(m_corpus.getSize(), post.getID(), post.getTitle(), content, prod.getID(), post.getLabel() - 1, timeStamp);
					if (this.m_stnDetector != null)
						AnalyzeDocWithPOSTagging(review);
					else
						AnalyzeDoc(review);
				}
			} catch (ParseException e) {
				System.out.print('T');
			} catch (JSONException e) {
				System.out.print('P');
			}
		}
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
	
	public static void main(String[] args) throws InvalidFormatException, FileNotFoundException, IOException{
		int classNumber = 2; //Define the number of classes
		int Ngram = 2; //The default value is bigram. 
		int lengthThreshold = 10; //Document length threshold
		
//		/*****Parameters in feature selection.*****/
		String featureSelection = "DF"; //Feature selection method.
		int chiSize = 50; // top ChiSquare words for aspect keyword selection
		String stopwords = "./data/Model/stopwords.dat";
		double startProb = 0.2; // Used in feature selection, the starting point of the features.
		double endProb = 1; // Used in feature selection, the ending point of the features.
		int DFthreshold = 25; // Filter the features with DFs smaller than this threshold.
		
		/*****The parameters used in loading files.*****/
		String folder = "./data/amazon/small/dedup/RawData";
		String suffix = ".json";
		String tokenModel = "./data/Model/en-token.bin"; //Token model
		String stnModel = "./data/Model/en-sent.bin"; //Sentence model
		String aspectModel = "./data/Model/topic_sentiment_input.txt"; // list of keywords in each aspect
		String aspectOutput = "./data/Model/topic_sentiment_output.txt"; // list of keywords in each aspect
		
		String pattern = String.format("%dgram_%s", Ngram, featureSelection);
		String fvFile = "data/Features/fv_2gram_topic_sentiment.txt";
//		String fvStatFile = String.format("data/Features/fv_stat_%s_small.txt", pattern);
		
		/****Loading json files to do feature selection first.*****/
//		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, null, Ngram, lengthThreshold);
//		analyzer.LoadStopwords(stopwords);
//		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
//		System.out.println("Performing feature selection, wait...");
//		analyzer.featureSelection(fvFile, featureSelection, startProb, endProb, DFthreshold); //Select the features.
//		analyzer.SaveCVStat(fvStatFile);
		
		/****After feature selection, do aspect annotation*****/
		AspectAnalyzer analyzer = new AspectAnalyzer(tokenModel, stnModel, classNumber, fvFile, Ngram, lengthThreshold);
		analyzer.LoadStopwords(stopwords);
		analyzer.LoadDirectory(folder, suffix); //Load all the documents as the data set.
		analyzer.BootStrapping(aspectModel, aspectOutput, chiSize, 0.9, 10);		
	}
}
