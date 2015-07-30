package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;

import opennlp.tools.cmdline.postag.POSModelLoader;
import opennlp.tools.postag.POSModel;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import structures.SentiWordNetDemoCode;
import structures.TokenizeResult;
import structures._Doc;
import structures._SparseFeature;
import structures._Stn;
import utils.Utils;

public class DocAnalyzer extends Analyzer {
	protected Tokenizer m_tokenizer;
	protected SnowballStemmer m_stemmer;
	protected SentenceDetectorME m_stnDetector;
	protected POSTaggerME m_tagger;
	Set<String> m_stopwords;
	
	protected HashMap<String, Integer> m_posTaggingFeatureNameIndex;
	protected HashMap<String, Integer> m_sentiwordNetFeatureNameIndex;
	protected HashMap<String, Double> m_sentiwordScoreMap;
	
	protected SentiWordNetDemoCode sentiWordNet;
	protected ArrayList<String> m_posPriorList;
	protected ArrayList<String> m_negPriorList;
	protected ArrayList<String> m_negationList;
	
	
	//Constructor with ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		m_stnDetector = null; // indicating we don't need sentence splitting
		
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
		
		m_posTaggingFeatureNameIndex = new HashMap<String, Integer>();
		m_sentiwordNetFeatureNameIndex = new HashMap<String, Integer>();
	}
	//TokenModel + stnModel.
	public DocAnalyzer(String tokenModel, String stnModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		
		if (stnModel!=null)
			m_stnDetector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnModel)));
		else
			m_stnDetector = null;
		
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
		m_posTaggingFeatureNameIndex = new HashMap<String, Integer>();
		m_sentiwordNetFeatureNameIndex = new HashMap<String, Integer>();

	}
	
	//TokenModel + stnModel + posModel.
	public DocAnalyzer(String tokenModel, String stnModel, String posModel, int classNo, String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		
		if (stnModel!=null)
			m_stnDetector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnModel)));
		else
			m_stnDetector = null;
		if (posModel!=null)
			m_tagger = new POSTaggerME(new POSModelLoader().load(new File(posModel)));
		else
			m_tagger = null;
		
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
		
		m_posTaggingFeatureNameIndex = new HashMap<String, Integer>();
		m_sentiwordNetFeatureNameIndex = new HashMap<String, Integer>();
	}
	
	//Constructor with ngram and fValue and sentence check.
	public DocAnalyzer(String tokenModel, String stnModel, int classNo,
			String providedCV, int Ngram, int threshold, String tagModel)
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();

		if (stnModel != null)
			m_stnDetector = new SentenceDetectorME(new SentenceModel(new FileInputStream(stnModel)));
		else
			m_stnDetector = null;
		m_tagger = new POSTaggerME(new POSModel(new FileInputStream(tagModel)));

		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
		m_posTaggingFeatureNameIndex = new HashMap<String, Integer>();
		m_sentiwordNetFeatureNameIndex = new HashMap<String, Integer>();
	}
	
	public void setReleaseContent(boolean release) {
		m_releaseContent = release;
	}
	
	public void LoadStopwords(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;

			while ((line = reader.readLine()) != null) {
				line = SnowballStemming(Normalize(line));
				if (!line.isEmpty())
					m_stopwords.add(line);
			}
			reader.close();
			System.out.format("Loading %d stopwords from %s\n", m_stopwords.size(), filename);
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
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

	
	//Tokenizer.
	protected String[] Tokenizer(String source){
		String[] tokens = m_tokenizer.tokenize(source);
		return tokens;
	}
	
	//Normalize.
	protected String Normalize(String token){
		token = Normalizer.normalize(token, Normalizer.Form.NFKC);
		token = token.replaceAll("\\W+", "");
		token = token.toLowerCase();
		
		if (Utils.isNumber(token))
			return "NUM";
		else
			return token;
	}
	
	//Snowball Stemmer.
	protected String SnowballStemming(String token){
		m_stemmer.setCurrent(token);
		if(m_stemmer.stem())
			return m_stemmer.getCurrent();
		else
			return token;
	}
	
	protected boolean isLegit(String token) {
		return !token.isEmpty() 
			&& !m_stopwords.contains(token)
			&& token.length()>1
			&& token.length()<20;
	}
	
	protected boolean isBoundary(String token) {
		return token.isEmpty();//is this a good checking condition?
	}
	
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	protected TokenizeResult TokenizerNormalizeStemmer(String source){
		String[] tokens = Tokenizer(source); //Original tokens.
		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]));
		
		
		LinkedList<String> Ngrams = new LinkedList<String>();
		int tokenLength = tokens.length, N = m_Ngram;	
		
		TokenizeResult result = new TokenizeResult(tokenLength);
		for(int i=0; i<tokenLength; i++) {
			String token = tokens[i];
			boolean legit = isLegit(token);
			if (legit) 
				Ngrams.add(token);//unigram
			else
				result.incStopwords();
			
			//N to 2 grams
			if (!isBoundary(token)) {
				for(int j=i-1; j>=Math.max(0, i-N+1); j--) {	
					if (isBoundary(tokens[j]))
						break;//touch the boundary
					
					token = tokens[j] + "-" + token;
					legit |= isLegit(tokens[j]);
					if (legit)//at least one of them is legitimate
						Ngrams.add(token);
				}
			}
		}
		
		result.setTokens(Ngrams.toArray(new String[Ngrams.size()]));
		return result;
	}

	//Load a movie review document and analyze it.
	//this is only specified for this type of review documents
	public void LoadDoc(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;

			while ((line = reader.readLine()) != null) {
				buffer.append(line);
			}
			reader.close();
			
			//How to generalize it to several classes???? 
			if(filename.contains("pos")){
				//Collect the number of documents in one class.
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 0));				
			}else if(filename.contains("neg")){
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 1));
			}
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}
	
	protected HashMap<Integer, Double> constructSpVct(String[] tokens, int y, HashMap<Integer, Double> docWordMap) {
		int index = 0;
		double value = 0;
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		
		for (String token : tokens) {//tokens could come from a sentence or a document
			// CV is not loaded, take all the tokens as features.
			if (!m_isCVLoaded) {
				if (m_featureNameIndex.containsKey(token)) {
					index = m_featureNameIndex.get(token);
					if (spVct.containsKey(index)) {
						value = spVct.get(index) + 1;
						spVct.put(index, value);
					} else {
						spVct.put(index, 1.0);
						if (docWordMap==null || !docWordMap.containsKey(index))
							if(m_featureStat.containsKey(token))
								m_featureStat.get(token).addOneDF(y);
//							else
//								System.out.println("Not found"+token);
					}
				} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
					expandVocabulary(token);// update the m_featureNames.
					index = m_featureNameIndex.get(token);
					spVct.put(index, 1.0);
					if(m_featureStat.containsKey(token))
						m_featureStat.get(token).addOneDF(y);
				}
				if(m_featureStat.containsKey(token))
					m_featureStat.get(token).addOneTTF(y);
			} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
				index = m_featureNameIndex.get(token);
				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);
					if (docWordMap==null || !docWordMap.containsKey(index))
						m_featureStat.get(token).addOneDF(y);
				}
				m_featureStat.get(token).addOneTTF(y);
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		
		return spVct;
	}
	
	/*Analyze a document and add the analyzed document back to corpus.
	 *In the case CV is not loaded, we need two if loops to check.
	 * The first is if the term is in the vocabulary.***I forgot to check this one!
	 * The second is if the term is in the sparseVector.
	 * In the case CV is loaded, we still need two if loops to check.*/
	protected boolean AnalyzeDoc(_Doc doc) {
		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource());// Three-step analysis.
		String[] tokens = result.getTokens();
		int y = doc.getYLabel();
		// Construct the sparse vector.
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
		if (spVct.size()>=m_lengthThreshold) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			doc.setStopwordProportion(result.getStopwordProportion());
			
			m_corpus.addDoc(doc);
			m_classMemberNo[y]++;
			if (m_releaseContent)
				doc.clearSource();
			return true;
		} else {
			/****Roll back here!!******/
			rollBack(spVct, y);
			return false;
		}
	}
	
	// adding sentence splitting function, modified for HTMM
	protected boolean AnalyzeDocWithStnSplit(_Doc doc) {
		TokenizeResult result;
		String[] sentences = m_stnDetector.sentDetect(doc.getSource());
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		ArrayList<_SparseFeature[]> stnList = new ArrayList<_SparseFeature[]>(); // to avoid empty sentences
		ArrayList<String[]> stnPosList = new ArrayList<String[]>(); // to avoid empty sentences
		ArrayList<String> rawStnList = new ArrayList<String>(); // to avoid empty sentences
		
		
		
		int y = doc.getYLabel();
		
		for(String sentence : sentences) {
			result = TokenizerNormalizeStemmer(sentence);// Three-step analysis.
			String[] posTags = m_tagger.tag(Tokenizer(sentence)); // only tokenize then POS tagging
			String[] tokens = result.getTokens();		
			HashMap<Integer, Double> sentence_vector = constructSpVct(tokens, y, spVct);			
			if (sentence_vector.size()>0) {//avoid empty sentence
				stnList.add(Utils.createSpVct(sentence_vector));
				rawStnList.add(sentence);
				stnPosList.add(posTags);
				Utils.mergeVectors(sentence_vector, spVct);
			}
		} // End For loop for sentence	
	
		//the document should be long enough
		if (spVct.size()>=m_lengthThreshold && stnList.size()>=m_minimumNumberofSentences) { 
			doc.createSpVct(spVct);
			doc.setSentences(stnList);
			doc.setRawSentences(rawStnList);
			doc.setSentencesPOSTag(stnPosList);
			setSentenceFeatureVectorForSentiment(doc);
			m_corpus.addDoc(doc);
			m_classMemberNo[y]++;
			
			if (m_releaseContent)
				doc.clearSource();
			return true;
		} else {
			/****Roll back here!!******/
			rollBack(spVct, y);
			return false;
		}
	}
	
	public void setSentenceWriter(String fileName) throws FileNotFoundException{
		m_sentenceWriter = new PrintWriter(new File(fileName));
	}
	
	//Load the sentinet word and store them in the dictionary for later use.
	public void LoadSNWWithScore(String filename) throws IOException {
		m_sentiwordScoreMap = new HashMap<String, Double>();// for tagging3 to
															// store features.
		// From String to list of doubles.
		HashMap<String, HashMap<Integer, Double>> tempDictionary = new HashMap<String, HashMap<Integer, Double>>();

		BufferedReader csv = null;
		try {
			csv = new BufferedReader(new FileReader(filename));
			int lineNumber = 0;
			String line;
			while ((line = csv.readLine()) != null) {
				lineNumber++;
				// If it's a comment, skip this line.
				if (!line.trim().startsWith("#")) {
					// We use tab separation
					String[] data = line.split("\t");
					String wordTypeMarker = data[0];
					// Is it a valid line? Otherwise, through exception.
					if (data.length != 6)
						throw new IllegalArgumentException("Incorrect tabulation format in file, line: " + lineNumber);
					// Calculate synset score as score = PosS - NegS. If it's 0,
					// then it is neutral word, ignore it.
					Double synsetScore = Double.parseDouble(data[2])
							- Double.parseDouble(data[3]);
					// Get all Synset terms
					String[] synTermsSplit = data[4].split(" ");
					// Go through all terms of current synset.
					for (String synTermSplit : synTermsSplit) {
						// Get synterm and synterm rank
						String[] synTermAndRank = synTermSplit.split("#"); // able#1 = [able, 1]
						String synTerm = synTermAndRank[0] + "#"
								+ wordTypeMarker; // able#a
						int synTermRank = Integer.parseInt(synTermAndRank[1]); // different senses of a word
						// Add the current term to map if it doesn't have one
						if (!tempDictionary.containsKey(synTerm))
							tempDictionary.put(synTerm,
									new HashMap<Integer, Double>());// <able#a, <<1, score>, <2, score>...>>
						// If the dict already has the synTerm, just add synset
						// link-<2, score> to synterm.
						tempDictionary.get(synTerm).put(synTermRank,
								synsetScore);
					}
				}
			}

			// Go through all the terms.
			Set<String> synTerms = tempDictionary.keySet();
			for (String synTerm : synTerms) {
				double score = 0;
				int count = 0;
				HashMap<Integer, Double> synSetScoreMap = tempDictionary.get(synTerm);
				Collection<Double> scores = synSetScoreMap.values();
				for (double s : scores) {
					if (s != 0) {
						score += s;
						count++;
					}
					if (score != 0)
						score = (double) score / count;
				}
				String[] termMarker = synTerm.split("#");
				m_sentiwordScoreMap.put(
						SnowballStemming(Normalize(termMarker[0])) + "#"
								+ termMarker[1], score);
			}
		} finally {
			if (csv != null) {
				csv.close();
			}
		}
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
			token = m_featureNames.get(index);
			tmp = sentiWordNet.extract(token, "n");
			if(tmp!=-2) // word found in SentiWordNet
				senScore+=tmp;
		}
		return senScore/wordsInSentence.length;
	}

	// receive sentence index as parameter
	// PosNeg count is done against the raw sentence
	// so stopword will also get counter here like not, none 
	// which is important for PosNeg count
	public int posNegCount(_Stn s) {
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
	public int negationCount(_Stn s) {
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
}	

