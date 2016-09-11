/**
 * 
 */
package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.HashMap;

import json.JSONArray;
import json.JSONException;
import json.JSONObject;
import opennlp.tools.util.InvalidFormatException;
import structures.SentiWordNet;
import structures.TokenizeResult;
import structures._Doc;
import structures._NewEggPost;
import structures._Stn;
import utils.Utils;

/**
 * @author Md Mustafizur Rahman
 * Text analyzer dedicate for HTSM model 
 * 1. Load specific format of NewEgg reviews
 * 2. Generate sentence features
 */

public class HTSMAnalyzer extends DocAnalyzer {
	enum LoadType {
		LT_pros,
		LT_cons,
		LT_comments,
		LT_procon,
		LT_all
	}
	
	//category of NewEgg reviews
	String m_category; 
	int m_prosStnCount = 0;
	int m_consStnCount = 0;
	LoadType m_prosConsLoad = LoadType.LT_all; // 0 means only load pros, 1 means load only cons, 2 means load both pros and cons 
	
	protected ArrayList<String> m_posPriorList;//list of positive seed words
	protected ArrayList<String> m_negPriorList;//list of negative seed words
	protected ArrayList<String> m_negationList;//list of negation seed words	
	
	public HTSMAnalyzer(String tokenModel, int classNo, String providedCV,
			int Ngram, int threshold, String category, LoadType type) throws InvalidFormatException,
			FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		//11/7/2013 7:01:22 PM
		m_dateFormatter = new SimpleDateFormat("M/d/yyyy h:mm:ss a");// standard date format for this project
		m_category = category;
		m_prosConsLoad = type;
	}

	public HTSMAnalyzer(String tokenModel, String stnModel, String posModel, int classNo, String providedCV,
			int Ngram, int threshold, String category, LoadType type)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, stnModel, posModel, classNo, providedCV, Ngram, threshold);
		m_dateFormatter = new SimpleDateFormat("M/d/yyyy h:mm:ss a");// standard date format for this project
		m_category = category;
		m_prosConsLoad = type;
	}

	//since the seed words are stemmed, please double check when you use such words in generating the features
	public void loadPriorPosNegWords(String pathToSentiWordNet, String pathToPosWords, String pathToNegWords, String pathToNegationWords) {
		m_posPriorList = new ArrayList<String>();
		m_negPriorList = new ArrayList<String>();
		m_negationList = new ArrayList<String>();
		
		BufferedReader file = null;
		try {
			file = new BufferedReader(new FileReader(pathToPosWords));
			String line;
			while ((line = file.readLine()) != null) {
				line = SnowballStemming(line); // only stemming since the list contains only single word per line and there is no number
				m_posPriorList.add(line);
			}
			file.close();
			
			file = new BufferedReader(new FileReader(pathToNegWords));
			while ((line = file.readLine()) != null) {
				line = SnowballStemming(line);
				m_negPriorList.add(line);
			}
			file.close();
			
			file = new BufferedReader(new FileReader(pathToNegationWords));
			while ((line = file.readLine()) != null) {
				line = SnowballStemming(line);
				m_negationList.add(line);
			}
			file.close();
			
			// loading the sentiWordnet
			m_sentiWordNet = new SentiWordNet(pathToSentiWordNet);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	//Load all the files in the directory.
	public void LoadNewEggDirectory(String folder, String suffix) throws IOException {
		if (folder==null || folder.isEmpty())
			return;
		
		int current = m_corpus.getSize();
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				LoadNewEggDoc(f.getAbsolutePath());
			} else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}

		System.out.format("Loading %d reviews from %s\n", m_corpus.getSize()-current, folder);
		if(this.m_stnDetector != null)
			System.out.printf("Number of Positive Sentences %d\nNumber of Negative Sentences %d\n", m_prosStnCount, m_consStnCount);
	}

		//Load a document and analyze it.
	public void LoadNewEggDoc(String filename) {
		JSONObject prod = null;
		String item;
		JSONArray itemIds, reviews;
		
		try {
			JSONObject json = LoadJSON(filename);
			prod = json.getJSONObject(m_category);
			itemIds = prod.names();
			System.out.printf("Under %s category, Number of Items: %d\n", m_category, itemIds.length());
		} catch (Exception e) {
			System.out.print('X');
			return;
		}	
		
		for(int i=0; i<itemIds.length(); i++) {
			try {
				item = itemIds.getString(i);
				reviews = prod.getJSONArray(item);
				for(int j=0; j<reviews.length(); j++) {
					if(this.m_stnDetector != null)
						AnalyzeNewEggPostWithSentence(new _NewEggPost(reviews.getJSONObject(j), item));
					else
						AnalyzeNewEggPost(new _NewEggPost(reviews.getJSONObject(j), item));
				}
			} catch (JSONException e) {
				System.out.print('P');
				e.printStackTrace();
			} catch (ParseException e) {
				e.printStackTrace();
			}
		}
	}
	
	void analyzeSection(String content, int y, HashMap<Integer, Double> docVct, ArrayList<HashMap<Integer, Double>> spVcts) {
		TokenizeResult result = TokenizerNormalizeStemmer(content);
		String[] tokens = result.getTokens();
		HashMap<Integer, Double> vPtr = constructSpVct(tokens, y, docVct);
		spVcts.add(vPtr);
		Utils.mergeVectors(vPtr, docVct);
	}
	
	protected boolean AnalyzeNewEggPost(_NewEggPost post) throws ParseException {
		String content;
		StringBuffer buffer = m_releaseContent?null:new StringBuffer(256);
		HashMap<Integer, Double> docVct = new HashMap<Integer, Double>(); // docVct is used to collect DF
		ArrayList<HashMap<Integer, Double>> spVcts = new ArrayList<HashMap<Integer, Double>>(); // Collect the index and counts of features.
		int y = post.getLabel()-1;
		
		if( (m_prosConsLoad==LoadType.LT_all || m_prosConsLoad==LoadType.LT_procon || m_prosConsLoad==LoadType.LT_pros)
			&& (content=post.getProContent()) != null) { // load pro section			
			
			analyzeSection(content, y, docVct, spVcts);
			if (!m_releaseContent)
				buffer.append(String.format("Pros: %s\n", content));
		}

		if( (m_prosConsLoad==LoadType.LT_all || m_prosConsLoad==LoadType.LT_procon || m_prosConsLoad==LoadType.LT_cons)
			&& (content=post.getConContent()) != null) { // load con section
			
			analyzeSection(content, y, docVct, spVcts);
			if (!m_releaseContent)
				buffer.append(String.format("Cons: %s\n", content));
		}
		
		if( (m_prosConsLoad==LoadType.LT_all || m_prosConsLoad==LoadType.LT_comments)  
			&& (content=post.getComments()) != null){ // load comment section
			
			analyzeSection(content, y, docVct, spVcts);
			if (!m_releaseContent)
				buffer.append(String.format("Comments: %s\n", content));
		}
		
		if (docVct.size()>=m_lengthThreshold) {
			long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
			//int ID, String name, String prodID, String title, String source, int ylabel, long timeStamp
			_Doc doc = new _Doc(m_corpus.getSize(), post.getID(), post.getProdId(), post.getTitle(), (m_releaseContent?null:buffer.toString()), y, timeStamp);			
			
			doc.setSourceType(2); // 2 means from newEgg
			doc.createSpVct(spVcts);
			
			m_corpus.addDoc(doc);
			m_classMemberNo[y]++;
			return true;
		} else
			return false;
	}
	
	int analyzeSectionWithStnSplit(String content, int y, int sLabel, HashMap<Integer, Double> docVct, ArrayList<HashMap<Integer, Double>> spVcts, ArrayList<_Stn> stnList) {
		TokenizeResult result = TokenizerNormalizeStemmer(content);
		HashMap<Integer, Double> vPtr;
		int stnCount = 0;
		
		for(String sentence : m_stnDetector.sentDetect(content)) {
			result = TokenizerNormalizeStemmer(sentence);
			vPtr = constructSpVct(result.getTokens(), y, docVct);

			if (vPtr.size()>0) {//avoid empty sentence
				String[] posTags = m_tagger.tag(result.getRawTokens()); // POS tagging has to be on the raw tokens

				stnList.add(new _Stn(Utils.createSpVct(vPtr), result.getRawTokens(), posTags, sentence, sLabel)); // 0 for pos
				stnCount++;
				Utils.mergeVectors(vPtr, docVct);
				spVcts.add(vPtr);
			}
		}
		
		return stnCount;
	}
	
	protected boolean AnalyzeNewEggPostWithSentence(_NewEggPost post) throws ParseException {
		String content;
		ArrayList<_Stn> stnList = new ArrayList<_Stn>(); // to avoid empty sentences
		ArrayList<HashMap<Integer, Double>> spVcts = new ArrayList<HashMap<Integer, Double>>(); // Collect the index and counts of features.

		StringBuffer buffer = m_releaseContent?null:new StringBuffer(256);
		HashMap<Integer, Double> docVct = new HashMap<Integer, Double>(); // docVct is used to collect DF statistics
		int y = post.getLabel()-1;
		int prosSentenceCounter = 0, consSentenceCounter = 0;		
		
		if( (m_prosConsLoad == LoadType.LT_pros || m_prosConsLoad==LoadType.LT_procon || m_prosConsLoad == LoadType.LT_all)
			&& (content=post.getProContent()) != null) { // sentences in pro section
			
			prosSentenceCounter = analyzeSectionWithStnSplit(content, y, 0, docVct, spVcts, stnList);
			if (!m_releaseContent)
				buffer.append(String.format("Pros: %s\n", content));
		}

		if( (m_prosConsLoad == LoadType.LT_cons || m_prosConsLoad==LoadType.LT_procon || m_prosConsLoad == LoadType.LT_all)
		   && (content=post.getConContent()) != null) {// tokenize cons
			
			consSentenceCounter = analyzeSectionWithStnSplit(content, y, 1, docVct, spVcts, stnList);
			if (!m_releaseContent)
				buffer.append(String.format("Cons: %s\n", content));
		} 
		
		
		if ((m_prosConsLoad == LoadType.LT_comments || m_prosConsLoad == LoadType.LT_all)
			&& (content=post.getComments()) != null) {// tokenize comments
			
			analyzeSectionWithStnSplit(content, y, -1, docVct, spVcts, stnList);
			if (!m_releaseContent)
				buffer.append(String.format("Comments: %s\n", content));
		}
		
		if (docVct.size()>=m_lengthThreshold && stnList.size()>=m_stnSizeThreshold) {
			long timeStamp = m_dateFormatter.parse(post.getDate()).getTime();
			//int ID, String name, String prodID, String title, String source, int ylabel, long timeStamp
			_Doc doc = new _Doc(m_corpus.getSize(), post.getID(), post.getProdId(), post.getTitle(), (m_releaseContent?null:buffer.toString()), y, timeStamp);			
			doc.setSourceType(2); // source = 2 means the Document is from newEgg
			doc.createSpVct(spVcts);
			doc.setSentences(stnList);
			setStnFvs(doc);
			
			m_corpus.addDoc(doc);
			m_classMemberNo[y] ++;
			m_prosStnCount += prosSentenceCounter;
			m_consStnCount += consSentenceCounter;
			return true;
		} else
			return false;
	}
	
	//this will generate sentence features for all other types of text documents accordingly
	@Override
	protected boolean AnalyzeDocWithStnSplit(_Doc doc) {
		if (super.AnalyzeDocWithStnSplit(doc)) {
			setStnFvs(doc);//construct feature vector
			return true;
		} else 
			return false;
	}
	
	// used by LR-HTSM for constructing topic/sentiment transition features for sentiment
	protected void setStnFvs(_Doc d) {
		_Stn[] sentences = d.getSentences();
		
		// start from 2nd sentence
		double pSim = Utils.cosine(sentences[0].getFv(), sentences[1].getFv()), nSim;
		double cLength, pLength = Utils.sumOfFeaturesL1(sentences[0].getFv());
		double pKL = Utils.klDivergence(calculatePOStagVector(sentences[0]), calculatePOStagVector(sentences[1])), nKL;
		double pSenScore = sentiWordScore(sentences[0]), cSenScore;
		int pPosNeg= posNegCount(sentences[0]), cPosNeg;
		int pNegationCount= negationCount(sentences[0]), cNegationCount;
		int stnSize = d.getSenetenceSize();
		
		for(int i=1; i<stnSize; i++){
			//cosine similarity	for both sentiment and topical transition		
			sentences[i-1].m_sentiTransitFv[0] = pSim;	
			sentences[i-1].m_transitFv[0] = pSim;		

			//length_ratio for topical transition
			cLength = Utils.sumOfFeaturesL1(sentences[i].getFv());			
			sentences[i-1].m_transitFv[1] = (pLength-cLength)/Math.max(cLength, pLength);
			pLength = cLength;
			
			//position for topical transition
			sentences[i-1].m_transitFv[2] = (double)i / stnSize;
			
			//sentiWordScore for sentiment transition
			cSenScore = sentiWordScore(sentences[i]);
			if(cSenScore<=-2 || pSenScore<=-2)
				sentences[i-1].m_sentiTransitFv[1] = 0;
			else if (cSenScore*pSenScore<0)
				sentences[i-1].m_sentiTransitFv[1] = 1; // transition
			else
				sentences[i-1].m_sentiTransitFv[1] = -1; // no transition
			pSenScore = cSenScore;

			//positive/negative count 
			cPosNeg = posNegCount(sentences[i]);
			if(pPosNeg==cPosNeg)
				sentences[i-1].m_sentiTransitFv[2] = -1; // no transition
			else
				sentences[i-1].m_sentiTransitFv[2] = 1; // transition
			pPosNeg = cPosNeg;

			//similar to previous or next for both topical and sentiment transitions
			if (i<stnSize-1) {
				nSim = Utils.cosine(sentences[i].getFv(), sentences[i+1].getFv());
				if (nSim>pSim) {
					sentences[i-1].m_sentiTransitFv[3] = 1;
					sentences[i-1].m_transitFv[3] = 1;
				} else if (nSim<pSim) {
					sentences[i-1].m_sentiTransitFv[3] = -1;
					sentences[i-1].m_transitFv[3] = -1;
				}
				pSim = nSim;
			}

			//kl divergency between POS tag vector to previous or next
			if (i<stnSize-1) {
				nKL = Utils.klDivergence(calculatePOStagVector(sentences[i]), calculatePOStagVector(sentences[i+1]));
				if (nKL>pKL)
					sentences[i-1].m_sentiTransitFv[4] = 1;
				else if (nKL<pKL)
					sentences[i-1].m_sentiTransitFv[4] = -1;
				pKL = nKL;
			}

			//negation count 
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
	protected double sentiWordScore(_Stn s) {
		return sentiWordScore(s.getRawTokens(), s.getSentencePosTag());
	}	
	
	// receive sentence index as parameter
	// PosNeg count is done against the raw sentence
	// so stopword will also get counter here like not, none 
	// which is important for PosNeg count
	protected int posNegCount(_Stn s) {
		String[] wordsInSentence = Tokenizer(s.getRawSentence()); //Original tokens.
		//Normalize them and stem them.		
		for(int i = 0; i < wordsInSentence.length; i++)
			wordsInSentence[i] = SnowballStemming(Normalize(wordsInSentence[i]));
		
		int posCount = 0;
		int negCount = 0;

		for(String word:wordsInSentence){
			if(m_posPriorList.contains(word))
				posCount++;
			else if(m_negPriorList.contains(word))
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
	// Negation count is done against the raw sentence
	// so stopword will also get counter here like not, none 
	// which is important for negation count
	protected int negationCount(_Stn s) {
		String[] wordsInSentence = Tokenizer(s.getRawSentence()); //Original tokens.
		//Normalize them and stem them.		
		for(int i = 0; i < wordsInSentence.length; i++)
			wordsInSentence[i] = SnowballStemming(Normalize(wordsInSentence[i]));
		
		int negationCount = 0;

		for(String word:wordsInSentence){
			if(m_negationList.contains(word))
				negationCount++;
		}
		return negationCount;
	}

	// calculate the number of Noun, Adjectives, Verb & AdVerb in a vector for a sentence
	// here i the index of the sentence
	protected double[] calculatePOStagVector(_Stn s) {
		String[] posTag = s.getSentencePosTag();
		double tagVector[] = new double[4]; 
		// index = 0 for noun
		// index = 1 for adjective
		// index = 2 for verb
		// index = 3 for adverb
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
		Utils.L1Normalization(tagVector);
		return tagVector;
	}
}
