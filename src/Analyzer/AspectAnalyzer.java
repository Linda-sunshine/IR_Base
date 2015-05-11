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
	
	int m_chiSize; // top words to be added to aspect keyword list 
	ArrayList<_Aspect> m_aspects; // a list of aspects specified by keywords
	int[] m_aspectDist; // distribution of aspects (count in DF)
	
	public AspectAnalyzer(String tokenModel, String stnModel, int classNo,
			String providedCV, int Ngram, int threshold, String aspectFile, int chiSize)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, classNo, providedCV, Ngram, threshold);
		
		LoadAspectKeywords(aspectFile);
		m_chiSize = chiSize;
	}

	public void LoadAspectKeywords(String filename){
		try {
			m_aspects = new ArrayList<_Aspect>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String tmpTxt;
			String[] container;
			HashSet<Integer> keywords;
			while( (tmpTxt=reader.readLine()) != null ){
				container = tmpTxt.split("\t");
				keywords = new HashSet<Integer>(container.length-1);
				for(int i=1; i<container.length; i++)
					keywords.add(m_featureNameIndex.get(container[i])); // map it to a controlled vocabulary term
				
				m_aspects.add(new _Aspect(container[0], keywords, m_chiSize));
				System.out.println("Keywords for " + container[0] + ": " + keywords.size());
			}
			reader.close();
			
			m_aspectDist = new int[m_aspects.size()];
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public void SaveAspectKeywords(String filename){
		try {	
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), "UTF-8"));
			for(_Aspect aspect:m_aspects){
				writer.write(aspect.m_name);
				for(Integer wid:aspect.m_keywords) {
					writer.write("\t" + m_featureNames.get(wid));
				}
				writer.write("\n");
			}
			writer.close();
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
			if ( (aspectID=s.getTopic())>-1){
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
		int selID = -1, aspectSize = m_aspects.size();
		double maxChi, chiV;
		Iterator<Map.Entry<String, _stat>> it = m_featureStat.entrySet().iterator();
		_Aspect aspect;
		
		int N = Utils.sumOfArray(m_aspectDist), DF;
		while(it.hasNext()){//set aspect assignment for each word
			Map.Entry<String, _stat> entry = it.next();
			
			_stat temp = entry.getValue();
			DF = Utils.sumOfArray(temp.getDF());
			
			maxChi = 0.0;
			selID = -1;
			for(int i=0; i<aspectSize; i++){				
				chiV = FeatureSelector.ChiSquare(N, DF, temp.getDF()[i], m_aspectDist[i]);				
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
	
	public void BootStrapping(String filename, double chi_ratio, int chi_iter){
		System.out.println("Vocabulary size: " + m_featureNames.size());
		
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
}
