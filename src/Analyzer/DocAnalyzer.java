package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Set;

import opennlp.tools.cmdline.postag.POSModelLoader;
import opennlp.tools.postag.POSTaggerME;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import structures.SentiWordNet;
import structures.TokenizeResult;
import structures._Doc;
import structures._Stn;
import structures._stat;
import utils.Utils;

public class DocAnalyzer extends Analyzer {
	protected Tokenizer m_tokenizer;
	protected SnowballStemmer m_stemmer;
	protected SentenceDetectorME m_stnDetector;
	protected POSTaggerME m_tagger;
	Set<String> m_stopwords;
	
	protected SentiWordNet m_sentiWordNet;
	protected ArrayList<String> m_posPriorList;//list of positive seed words
	protected ArrayList<String> m_negPriorList;//list of negative seed words
	protected ArrayList<String> m_negationList;//list of negation seed words
	
	
	//Constructor with ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, int Ngram, int threshold) 
			throws InvalidFormatException, FileNotFoundException, IOException {
		super(classNo, threshold);
		m_tokenizer = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
		m_stemmer = new englishStemmer();
		m_stnDetector = null; // indicating we don't need sentence splitting
		
		m_Ngram = Ngram;
		m_isCVLoaded = LoadCV(providedCV);
		m_stopwords = new HashSet<String>();
		m_releaseContent = true;
	}
	
	//TokenModel + stnModel.
	public DocAnalyzer(String tokenModel, String stnModel, int classNo, 
			String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
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
	}
	
	//TokenModel + stnModel + posModel.
	public DocAnalyzer(String tokenModel, String stnModel, String posModel, int classNo, 
			String providedCV, int Ngram, int threshold) throws InvalidFormatException, FileNotFoundException, IOException{
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
	
	//Tokenizing input text string
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
	
	//check if it is a sentence's boundary
	protected boolean isBoundary(String token) {
		return token.isEmpty();//is this a good checking condition?
	}
	
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	protected TokenizeResult TokenizerNormalizeStemmer(String source){
		String[] tokens = Tokenizer(source); //Original tokens.
		TokenizeResult result = new TokenizeResult(tokens);
		
		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]));
		
		LinkedList<String> Ngrams = new LinkedList<String>();
		int tokenLength = tokens.length, N = m_Ngram;			
		
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
	//do we still need this function, or shall we normalize it with json format?
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
	
	//convert the input token sequence into a sparse vector (docWordMap cannot be changed)
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
						if (docWordMap==null || !docWordMap.containsKey(index)) {
							if(m_featureStat.containsKey(token))
								m_featureStat.get(token).addOneDF(y);
						}
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
					if (docWordMap==null || !docWordMap.containsKey(index)){
						if(m_featureStat.containsKey(token))
							m_featureStat.get(token).addOneDF(y);
					}
				}
				m_featureStat.get(token).addOneTTF(y);
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		return spVct;
	}
	
	//Added by Lin for constructing pos tagging vectors.
	public HashMap<Integer, Double> constructPOSSpVct(String[] tokens, String[] tags){
		int posIndex = 0;
		double posValue = 0;
		HashMap<Integer, Double> posTaggingVct = new HashMap<Integer, Double>();//Collect the index and counts of projected features.	

		for(int i = 0; i < tokens.length; i++){
			if (isLegit(tokens[i])){
				//If the word is adj/adv, construct the sparse vector.
				if(tags[i].equals("RB") || tags[i].equals("RBR") || tags[i].equals("RBS") 
					|| tags[i].equals("JJ") || tags[i].equals("JJR") || tags[i].equals("JJS")) {
					if(m_posTaggingFeatureNameIndex.containsKey(tokens[i])){
						posIndex = m_posTaggingFeatureNameIndex.get(tokens[i]);
						if(posTaggingVct.containsKey(posIndex)){
							posValue = posTaggingVct.get(posIndex) + 1;
							posTaggingVct.put(posIndex, posValue);
						} else
							posTaggingVct.put(posIndex, 1.0);
					} else {
						posIndex = m_posTaggingFeatureNameIndex.size();
						m_posTaggingFeatureNameIndex.put(tokens[i], posIndex);
						posTaggingVct.put(posIndex, 1.0);
					}
				}
			}
		}
		return posTaggingVct;
	}
	
	/*Analyze a document and add the analyzed document back to corpus.*/
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
		int y = doc.getYLabel();
		String[] sentences = m_stnDetector.sentDetect(doc.getSource());
		HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); // Collect the index and counts of features.
		ArrayList<_Stn> stnList = new ArrayList<_Stn>(); // sparse sentence feature vectors 
		double stopwordCnt = 0, rawCnt = 0;
		
		for(String sentence : sentences) {
			result = TokenizerNormalizeStemmer(sentence);// Three-step analysis.
			HashMap<Integer, Double> sentence_vector = constructSpVct(result.getTokens(), y, spVct);// construct bag-of-word vector based on normalized tokens	

			if (sentence_vector.size()>0) {//avoid empty sentence				
				String[] posTags = m_tagger.tag(result.getRawTokens());
				
				stnList.add(new _Stn(Utils.createSpVct(sentence_vector), result.getRawTokens(), posTags, sentence));
				Utils.mergeVectors(sentence_vector, spVct);
				
				stopwordCnt += result.getStopwordCnt();
				rawCnt += result.getRawCnt();
			}
		} // End For loop for sentence	
	
		//the document should be long enough
		if (spVct.size()>=m_lengthThreshold && stnList.size()>=m_stnSizeThreshold) { 
			doc.createSpVct(spVct);		
			doc.setStopwordProportion(stopwordCnt/rawCnt);
			doc.setSentences(stnList);
			
			setStnFvs(doc);
			
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
	
	// used by LR-HTSM for constructing topic/sentiment transition features for sentiment
	public void setStnFvs(_Doc d) {
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
	public double sentiWordScore(_Stn s) {
		return sentiWordScore(s.getRawTokens(), s.getSentencePosTag());
	}

	// added by Lin, the same function with different parameters.
	public double sentiWordScore(String[] tokens, String[] posTags) {
		double senScore = 0.0;
		double tmp;
		String word, tag;

		for(int i=0; i<tokens.length;i++){
			word = SnowballStemming(Normalize(tokens[i]));
			tag = posTags[i];
			if(tag.equalsIgnoreCase("NN") || tag.equalsIgnoreCase("NNS") || tag.equalsIgnoreCase("NNP") || tag.equalsIgnoreCase("NNPS"))
				tag = "n";
			else if(tag.equalsIgnoreCase("JJ") || tag.equalsIgnoreCase("JJR") || tag.equalsIgnoreCase("JJS"))
				tag = "a";
			else if(tag.equalsIgnoreCase("VB") || tag.equalsIgnoreCase("VBD") || tag.equalsIgnoreCase("VBG"))
				tag = "v";
			else if(tag.equalsIgnoreCase("RB") || tag.equalsIgnoreCase("RBR") || tag.equalsIgnoreCase("RBS"))
				tag = "r";
			
			tmp = m_sentiWordNet.extract(word, tag);
			if(tmp!=-2) // word found in SentiWordNet
				senScore+=tmp;
		}
		return senScore/tokens.length;//This is average, we may have different ways of calculation.
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
	
	//Load all the files in the directory.
	public void LoadDirectory(String folder, String suffix) throws IOException {
			if (folder==null || folder.isEmpty())
				return;
			
			int current = m_corpus.getSize();
			File dir = new File(folder);
			for (File f : dir.listFiles()) {
				if (f.isFile() && f.getName().endsWith(suffix)) {
					LoadDoc(f.getAbsolutePath());
				} else if (f.isDirectory())
					LoadDirectory(f.getAbsolutePath(), suffix);
			}
			System.out.format("Loading %d reviews from %s\n", m_corpus.getSize()-current, folder);
		}
	//Amazon review in SNAP: each file contains all the reviews in one category.
	public void LoadSNAPFiles(String folder){
		int count = 0;
		if(folder == null || folder.isEmpty())
			return;
		File dir = new File(folder);
		for(File f: dir.listFiles()){
			if(f.isFile()){
				LoadOneSNAPFile(f.getAbsolutePath());
				count++;
			}
		}
		System.out.format("Load files from %d categories in total.", count);
	}
	//Load reviews from each file and one file contains reviews from one category.	
	public void LoadOneSNAPFile(String filename){
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			boolean rmFlag = false;
			ArrayList<String> oneReview = new ArrayList<String>();
			while ((line = reader.readLine()) != null) {
				if(line.startsWith("ProductInfo")|| line.startsWith("summary") || line.startsWith("text") || line.startsWith("score")){
					if(line.split(":").length == 1 && line.split(":")[0].equals("summary")){
						System.out.print('S');
					} else if(line.split(":").length == 1 && line.split(":")[0].equals("text")){
						rmFlag = true;
					} else
						oneReview.add(line.split(":")[1]);
				} else if (line.isEmpty()){
					if(rmFlag){
						oneReview.clear();
						rmFlag = false;
					}
					else{
						String content = oneReview.get(0);
						int label = (int) (Double.valueOf(oneReview.get(2)) - 1);
						_Doc review = new _Doc(m_corpus.getSize(), oneReview.get(1), content, label);
						if(this.m_stnDetector!=null)
							AnalyzeDocWithStnSplit(review);
						else
							AnalyzeDoc(review);
						oneReview.clear();
						continue;
					}
				}
			}
			reader.close();
			System.out.format("Loading files from %s, %d files in total.\n", filename, m_corpus.getSize());
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}	
}	
