package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.instrument.IllegalClassFormatException;
import java.text.Normalizer;
import java.util.HashMap;
<<<<<<< HEAD
=======
import java.util.IllegalFormatException;
import java.util.Iterator;
import java.util.Map;
>>>>>>> bf3c75820239635d46277c5cb36bf3e51f69486d

import opennlp.tools.util.InvalidFormatException;
import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import structures._stat;
import utils.Utils;

public class DocAnalyzer extends Analyzer {
	private int m_Ngram; 
	
	//Constructor.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, String fs) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		if(providedCV != null)
			this.LoadCV(providedCV);
		if(fs != null){
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}	
		this.m_Ngram = 1;
		//this.m_fValue = "TF";
	}	
	//Constructor with ngram and fValue.
	public DocAnalyzer(String tokenModel, int classNo, String providedCV, String fs, int Ngram) throws InvalidFormatException, FileNotFoundException, IOException{
		super(tokenModel, classNo);
		if(providedCV != null)
			this.LoadCV(providedCV);
		if(fs != null){
			this.m_isFetureSelected = true;
			this.featureSelection = fs;
		}
		this.m_Ngram = Ngram;
	}
	
	//Load the features from a file and store them in the m_featurNames.
	public boolean LoadCV(String filename) {
		int count = 0;
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;

			while ((line = reader.readLine()) != null) {
				//buffer.append(line);
				this.m_featureNames.add(line);
			}
			reader.close();
		} catch(IOException e){
				System.err.format("[Error]Failed to open file %s!!", filename);
				e.printStackTrace();
				return false;
		}
		//Indicate we can only use the loaded features to construct the feature vector!!
		m_isCVLoaded = true;
		
		//Set the index of the features.
		for(String f: this.m_featureNames){
			this.m_featureNameIndex.put(f, count);
			this.m_featureIndexName.put(count, f);
			this.m_featureStat.put(f, new _stat(this.m_classNo));
			count++;
		}
		return true; // if loading is successful
	}
	
	//Save all the features and feature stat into a file.
	public PrintWriter SaveCV(String featureLocation) throws FileNotFoundException {
		// File file = new File(path);
		PrintWriter writer = new PrintWriter(new File(featureLocation));
		for (int i = 0; i < this.m_featureNames.size(); i++)
			writer.println(this.m_featureNames.get(i));
		writer.close();
		return writer;
	}
	
	//Save all the features and feature stat into a file.
	public PrintWriter SaveCVStat(String finalLocation) throws FileNotFoundException{
		//File file = new File(path);
		PrintWriter writer = new PrintWriter(new File(finalLocation));
		for(int i = 0; i < this.m_featureNames.size(); i++){
			writer.print("\nfeature: " + this.m_featureNames.get(i));
			_stat temp = this.m_featureStat.get(this.m_featureNames.get(i));
			for(int j = 0; j < temp.getDF().length; j++){
				writer.print("\tDF[" + j + "]:" + temp.getDF()[j]);
			}
			for(int j = 0; j < temp.getTTF().length; j++){
				writer.print("\tTTF[" + j + "]:" + temp.getTTF()[j]);
			}
		}
		writer.close();
		return writer;
	}
	
	//Tokenizer.
	public String[] Tokenizer(String source){
		String[] tokens = m_tokenizer.tokenize(source);
		return tokens;
	}
	
	//Normalize.
	public String Normalize(String token){
		token = Normalizer.normalize(token, Normalizer.Form.NFKC);
		token = token.replaceAll("^a-zA-Z", "");
		token = token.toLowerCase();
		return token;
	}
	
	//Snowball Stemmer.
	public String SnowballStemming(String token){
		m_stemmer.setCurrent(token);
		if(m_stemmer.stem())
			return m_stemmer.getCurrent();
		else
			return m_stemmer.getCurrent();
	}
	
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	public String[] TokenizerNormalizeStemmer(String source){
		String[] tokens = Tokenizer(source); //Original tokens.
		int tokenLength = tokens.length, N = this.m_Ngram, NgramNo = 0;
		ArrayList<String> Ngrams = new ArrayList<String>();
		//Collect all the grams, Ngrams, N-1grams...
		while(N > 0){
			NgramNo = tokenLength - N + 1;
			for(int i = 0; i < NgramNo; i++){
				String Ngram = "";
				for(int j = 0; j < N; j++)
					Ngram = Ngram.concat(tokens[i+j]);//If n gram, concat them.
				Ngrams.add(Ngram);
			}  N--;
		}
		//Normalize them and stem them.
		String[] ProcessedNgrams = new String[Ngrams.size()];
		for(int i = 0; i < Ngrams.size(); i++){
			String Ngram = Ngrams.get(i);
			Ngram = Normalize(Ngram);
			Ngram = SnowballStemming(Ngram);
			ProcessedNgrams[i] = Ngram;
		}
		return ProcessedNgrams;
	}
	
	//Load all the files in the directory.
	public void LoadDirectory(String folder, String suffix) throws IOException {
		File dir = new File(folder);
		for (File f : dir.listFiles()) {
			if (f.isFile() && f.getName().endsWith(suffix)) {
				LoadDoc(f.getAbsolutePath());
			} else if (f.isDirectory())
				LoadDirectory(f.getAbsolutePath(), suffix);
		}
	}

	//Load a document and analyze it.
	public void LoadDoc(String filename) {
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			StringBuffer buffer = new StringBuffer(1024);
			String line;

			while ((line = reader.readLine()) != null) {
				buffer.append(line);
			}
			reader.close();
			//At present, we just consider two classes. Just set the label directly.
			//How to generalize it to several classes???? 
			if(filename.contains("pos")){
				//Collect the number of documents in one class.
				//this.m_corpus.addOneClassMember(0);
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 0));
				this.m_classMemberNo[0]++;
			}else if(filename.contains("neg")){
				//this.m_corpus.addOneClassMember(1);
				AnalyzeDoc(new _Doc(m_corpus.getSize(), buffer.toString(), 1));
				this.m_classMemberNo[1]++;
			}
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
		this.m_corpus.sizeAddOne();
	}
	
	//Add one more token to the current vocabulary.
	private void expandVocabulary(String token) {
		m_featureNames.add(token); //Add the new feature.
		m_featureNameIndex.put(token, (m_featureNames.size() - 1)); //set the index of the new feature.
		m_featureIndexName.put((m_featureNames.size() - 1), token);
	}

	/*Analyze a document and add the analyzed document back to corpus.	
	 *In the case CV is not loaded, we need two if loops to check. 
	 * The first is if the term is in the vocabulary.***I forgot to check this one!
	 * The second is if the term is in the sparseVector.
	 * In the case CV is loaded, we still need two if loops to check.*/
	public void AnalyzeDoc(_Doc doc) {
		try{
			String[] tokens = TokenizerNormalizeStemmer(doc.getSource());//Three-step analysis.
			HashMap<Integer, Double> spVct = new HashMap<Integer, Double>(); //Collect the index and counts of features.
			int index = 0;
			double value = 0;
			//Construct the sparse vector.
			for(String token:tokens) {
				//CV is not loaded, take all the tokens as features.
				if(!m_isCVLoaded){
					if (m_featureNameIndex.containsKey(token)) {
						index = m_featureNameIndex.get(token);
						if(spVct.containsKey(index)){
							value = spVct.get(index) + 1;
							spVct.put(index, value);
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						} else{
							spVct.put(index, 1.0);
							this.m_featureStat.get(token).addOneDF(doc.getYLabel());
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						}
					} 
					else{
						//indicate we allow the analyzer to dynamically expand the feature vocabulary
						expandVocabulary(token);//update the m_featureNames.
						updateFeatureStat(token);
						index = m_featureNameIndex.get(token);
						spVct.put(index, 1.0);
						this.m_featureStat.get(token).addOneDF(doc.getYLabel());
						this.m_featureStat.get(token).addOneTTF(doc.getYLabel());					
						}
				}
				//CV is loaded.
				else{
					if (m_featureNameIndex.containsKey(token)) {
						index = m_featureNameIndex.get(token);
						if(spVct.containsKey(index)){
							value = spVct.get(index) + 1;
							spVct.put(index, value);
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						} else{
							spVct.put(index, 1.0);
							this.m_featureStat.get(token).addOneDF(doc.getYLabel());
							this.m_featureStat.get(token).addOneTTF(doc.getYLabel());
						}
						//if the token is not in the vocabulary, nothing to do.
					}
				}
			}
			//Create the sparse vector for the document.
			doc.createSpVct(spVct);
			//doc.L1Normalization(doc.getSparse());//Normalize the sparse vector.
			doc.L2Normalization(doc.getSparse());//Normalize the sparse vector.
		}catch(Exception e){
			e.printStackTrace();
		}
		m_corpus.addDoc(doc);
	}
	
	//With a new feature added into the vocabulary, add the stat into stat arraylist.
	public void updateFeatureStat(String token){
		this.m_featureStat.put(token, new _stat(this.m_classNo));
	}
	
	//Return the total number of words in vocabulary.
	public int getFeatureSize(){
		return this.m_featureNames.size();
	}
	
	//Set the counts of every feature with respect to the collected class number.
	public void setFeatureConfiguration() {
		//Initialize the counts of every feature.
		for (String featureName: this.m_featureStat.keySet()){
			this.m_featureStat.get(featureName).initCount(this.m_classNo);
		}
		for (String featureName : this.m_featureStat.keySet()) {
			this.m_featureStat.get(featureName).setCounts(this.m_classMemberNo);
		}
	}

	//Calculate the similarity between two docs.
	public double calculateSimilarity(_Doc d1, _Doc d2){
		double similarity = 0;
		_SparseFeature[] spVct1 = d1.getSparse();
		_SparseFeature[] spVct2 = d2.getSparse();
		int start = spVct1[0].getIndex() > spVct2[0].getIndex() ? spVct1[0].getIndex() : spVct2[0].getIndex();
		int end = spVct1[spVct1.length - 1].getIndex() < spVct2[spVct2.length - 1].getIndex() ? spVct1[spVct1.length - 1].getIndex() : spVct2[spVct2.length - 1].getIndex();
		for(int i = start; i <= end; i ++){
			//Have not finished.
		}
		return similarity;
	}
	
	//Select the features and store them in a file.
	public void featureSelection(String location, double startProb, double endProb) throws FileNotFoundException{
		FeatureSelection selector = new FeatureSelection(startProb, endProb);
		this.m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
		this.setFeatureConfiguration(); //Construct the table for features.
		
		if(this.m_isFetureSelected){
			System.out.println("*******************************************************************");
			if (this.featureSelection.equals("DF")){
				System.out.println("DF is used to do feature selection!!");
				//System.out.println("The start point is " + this.m_selector.m_startProb + " and the end point is " + this.m_selector.m_endProb);
				this.m_featureNames = selector.DF(this.m_featureStat);
			}
			else if(this.featureSelection.equals("IG")){
				System.out.println("IG is used to do feature selection!!");
				this.m_featureNames = selector.IG(this.m_featureStat, this.m_classMemberNo);
			}
			else if(this.featureSelection.equals("MI")){
				System.out.println("MI is used to do feature selection!!");
				this.m_featureNames = selector.MI(this.m_featureStat, this.m_classMemberNo);
			}
			else if(this.featureSelection.equals("CHI")){
				System.out.println("CHI is used to do feature selection!!");
				this.m_featureNames = selector.CHI(this.m_featureStat, this.m_classMemberNo);
			}
//			else if(this.featureSelection.equals("TS")){
//				System.out.println("TS is used to do feature selection!!");
//				this.m_featureNames = this.m_selector.TS();
//			}
		}
		System.out.println("Feature Selction: The start point is " + selector.m_startProb + " and the end point is " + selector.m_endProb);
		this.SaveCV(location); // Save all the features and probabilities we get after analyzing.
		System.out.println(this.m_featureNames.size() + " features are selected!");
	}
	
	//Return corpus without parameter and feature selection.
	public _Corpus returnCorpus(String finalLocation) throws FileNotFoundException{
		this.m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
		this.SaveCVStat(finalLocation);
		return this.m_corpus; 
	}
	
	//Give the option, which would be used as the method to calculate feature value and returned corpus, calculate the feature values.
	public _Corpus setFeatureValues(_Corpus c, DocAnalyzer analyzer, String fValue){
		HashMap<String, _stat> featureStat = analyzer.m_featureStat;
		HashMap<Integer, String> featureIndexName = analyzer.m_featureIndexName;
		
		ArrayList<_Doc> docs = c.getCollection(); //Get the collection of all the documents.
		int N = docs.size();
		if (fValue.equals("TFIDF")){
			for(int i = 0; i < docs.size(); i++){
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for(_SparseFeature sf: sfs){
					String featureName = featureIndexName.get(sf.getIndex());
					_stat stat = featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double TFIDF = TF * Math.log((N + 1)/DF);
					sf.setValue(TFIDF);
					//System.out.println("test");
				}
			}
		}else if(fValue.equals("BM25")){
			double k1 = 1.5; //[1.2, 2]
			double b = 10; //(0, 1000]
			//Iterate all the documents to get the average document length.
			double navg = 0;
			for(int k = 0; k < N; k++)
				navg += docs.get(k).getDocLength();
			navg = navg/N; 
			
			for(int i = 0; i < docs.size(); i++){
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for(_SparseFeature sf: sfs){
					String featureName = featureIndexName.get(sf.getIndex());
					_stat stat = featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double n = temp.getDocLength();
					double BM25 = Math.log((N - DF + 0.5) / (DF + 0.5)) * TF * (k1 + 1) / (k1 * (1 - b + b * n / navg) + TF);
					sf.setValue(BM25);
				}
			}
		} else if(fValue.equals("PLN")){
			double s = 0.5; //[0, 1]
			//Iterate all the documents to get the average document length.
			double navg = 0;
			for(int k = 0; k < N; k++)
				navg += docs.get(k).getDocLength();
			navg = navg/N; 
			
			for(int i = 0; i < docs.size(); i++){
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				for(_SparseFeature sf: sfs){
					String featureName = featureIndexName.get(sf.getIndex());
					_stat stat = featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double n = temp.getDocLength();
					double PLN = (1 + Math.log(1 + Math.log(TF)) / (1 - s + s * n / navg)) * Math.log((N + 1) / DF);
					sf.setValue(PLN);
				}
			}
		} else return c;
		return c;
	}
}	

