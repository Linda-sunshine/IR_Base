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
import structures.TokenizeResult;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import structures._Stn;
import structures._stat;
import utils.Utils;


/**
 * @author hongning
 * Keyword based bootstrapping for aspect segmentation and annotation
 * Wang, Hongning, Yue Lu, and Chengxiang Zhai. "Latent aspect rating analysis on review text data: a rating regression approach." 
 * Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2010.
 */

public class AspectAnalyzer extends DocAnalyzer {
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
	boolean m_aspByCount = false; // select aspect by maximum count or simple matching
		
	public AspectAnalyzer(String tokenModel, String stnModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, classNo, providedCV, Ngram, threshold);
	}

	public AspectAnalyzer(String tokenModel, String stnModel, String tagModel, int classNo, String providedCV, int Ngram, int threshold, String aspectFile, boolean aspFlag) 
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, tagModel, classNo, providedCV, Ngram, threshold);
		LoadAspectKeywords(aspectFile);
		m_aspByCount = aspFlag;
	}

	public void LoadAspectKeywords(String filename){
		try {
			m_aspects = new ArrayList<_Aspect>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			String[] container;
			HashSet<Integer> keywords;
			while( (tmpTxt=reader.readLine()) != null ) {
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
				} else if (count==maxCount)
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
				chiV = Utils.ChiSquare(N, DF, DFarray[i], m_aspectDist[i]);				
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
	
	double[] detectAspects(HashMap<Integer, Double> spVct){
		double[] aspVct = new double[m_aspDimension];
		for(int i = 0; i < m_aspects.size(); i++){
			HashSet<Integer> keywords = m_aspects.get(i).m_keywords;
			for(int key: keywords){
				if(m_aspByCount){//we will accumulate the matching count for this aspect in the sentence
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
	
	@Override
	protected boolean AnalyzeDocWithStnSplit(_Doc doc) {
		double sentiScore = 0;
		TokenizeResult result;
		String[] sentences = m_stnDetector.sentDetect(doc.getSource());
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		
		//Added by Lin for constructing postagging vector.
		HashMap<Integer, Double> posTaggingVct = new HashMap<Integer, Double>();//Collect the index and counts of projected features.	
		int y = doc.getYLabel();
		double stopwordCnt = 0, rawCnt = 0;
		
		for(String sentence : sentences) {
			result = TokenizerNormalizeStemmer(sentence);// Three-step analysis.
			String[] rawTokens = result.getRawTokens();
			String[] posTags = m_tagger.tag(rawTokens); // only tokenize then POS tagging

			HashMap<Integer, Double> sentence_vector = constructSpVct(result.getTokens(), y, spVct);	
			//Added by Lin for constructing postagging vector.
			HashMap<Integer, Double> postaggingSentenceVct = constructPOSSpVct(rawTokens, posTags); // Collect the index and counts of features.

			if (sentence_vector.size()>0) {//avoid empty sentence
				Utils.mergeVectors(sentence_vector, spVct);
				Utils.mergeVectors(postaggingSentenceVct, posTaggingVct);
				sentiScore += sentiWordScore(rawTokens, posTags);//since we already have the postagging, we don't need to repeat it.
				
				stopwordCnt += result.getStopwordCnt();
				rawCnt += result.getRawCnt();
			}
		} // End For loop for sentence	
	
		//the document should be long enough
		if (spVct.size()>=m_lengthThreshold) { 
			doc.createSpVct(spVct);
			doc.setStopwordProportion(stopwordCnt/rawCnt);
			doc.createPOSVct(posTaggingVct);//added by Lin.
			
			doc.setAspVct(detectAspects(spVct));//Added by Lin for detecting aspects of a document.
			doc.setSentiScore(sentiScore/spVct.size());//average sentence sentiWordNet score
			
			m_corpus.addDoc(doc);
			m_classMemberNo[y] ++;
			
			if (m_releaseContent)
				doc.clearSource();
			return true;
		} else {
			/****Roll back here!!******/
			rollBack(spVct, y);
			return false;
		}
	}
}
