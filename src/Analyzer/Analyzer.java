package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;

import structures._Corpus;
import structures._Doc;
import structures._Pair;
import structures._SparseFeature;
import structures._stat;
import utils.Utils;

public abstract class Analyzer {
	
	protected _Corpus m_corpus;
	protected int m_classNo; //This variable is just used to init stat for every feature. How to generalize it?
	int[] m_classMemberNo; //Store the number of members in a class.
	protected int m_Ngram; 
	
	protected ArrayList<String> m_featureNames; //ArrayList for features
	protected HashMap<String, Integer> m_featureNameIndex;//key: content of the feature; value: the index of the feature
	protected HashMap<String, _stat> m_featureStat; //Key: feature Name; value: the stat of the feature
	/* Indicate if we can allow new features. After loading the CV file, the flag is set to true, 
	 * which means no new features will be allowed.*/
	protected boolean m_isCVLoaded = false;
	
	// indicate if we will collect corpus-level feature statistics
	protected boolean m_isCVStatLoaded = false;
	protected int m_TotalDF = -1;
	
	//minimal length of indexed document
	protected int m_lengthThreshold = 5;
	//minimal size of sentences
	protected int m_stnSizeThreshold = 2;
	//if we have store content of documents
	protected boolean m_releaseContent;
	
	/** for time-series features **/
	//The length of the window which means how many labels will be taken into consideration.
	private LinkedList<_Doc> m_preDocs;	

	protected HashMap<String, Integer> m_posTaggingFeatureNameIndex;//Added by Lin
	private HashMap<_Pair, Integer> m_LCSMap; //added by Lin for storing LCS pairs.s
 
	public Analyzer(int classNo, int minDocLength) {
		m_corpus = new _Corpus();
		
		m_classNo = classNo;
		m_classMemberNo = new int[classNo];
		
		m_featureNames = new ArrayList<String>();
		m_featureNameIndex = new HashMap<String, Integer>();//key: content of the feature; value: the index of the feature
		m_featureStat = new HashMap<String, _stat>();
		
		m_posTaggingFeatureNameIndex = new HashMap<String, Integer>();
		
		m_lengthThreshold = minDocLength;
		
		m_preDocs = new LinkedList<_Doc>(); // key: POS tag; value: index in the list		
	}	
	
	public void reset() {
		Arrays.fill(m_classMemberNo, 0);
		m_featureNames.clear();
		m_featureNameIndex.clear();
		m_featureStat.clear();
		m_corpus.reset();
	}
	
	public HashMap<String, Integer> getFeatureMap() {
		return m_featureNameIndex;
	}
	
	//Load the features from a file and store them in the m_featurNames.@added by Lin.
	protected boolean LoadCV(String filename){
		if (filename==null || filename.isEmpty())
			return false;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				
				if (line.startsWith("#")){
					if (line.startsWith("#NGram")) {//has to be decoded
						int pos = line.indexOf(':');
						m_Ngram = Integer.valueOf(line.substring(pos+1));
					}
						
				} else 
					expandVocabulary(line);
			}
			reader.close();
			
			System.out.format("%d feature words loaded from %s.\n", m_featureNames.size(), filename);
			m_isCVLoaded = true;
			
			return true;
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return false;
		}
	}
	
	public void LoadLCSFiles(String folder){
		if(folder == null || folder.isEmpty())
			return;
		
		m_LCSMap = new HashMap<_Pair, Integer>();
		File dir = new File(folder);
		for(File f: dir.listFiles()){
			long start = System.currentTimeMillis();
			if(f.isFile()){
				try {
					BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(f.getAbsolutePath()), "UTF-8"));
					String line;
					while ((line = reader.readLine()) != null) {
						String[] pair = line.split(",");
						
						if(pair.length != 3)
							System.out.println("Wrong LCS triple!");
						else
							m_LCSMap.put(new _Pair(Integer.parseInt(pair[0]), Integer.parseInt(pair[1].trim())), Integer.parseInt(pair[2].trim()));
					}
					reader.close();
					System.out.format("After reading %s in %.4f secs, %d LCS triples loaded totally.\n", f.getAbsolutePath(), (double)(System.currentTimeMillis()-start)/1000.0, m_LCSMap.size());
				} catch (IOException e) {
					System.err.format("[Error]Failed to open file %s!!", folder);
				}		
			}
		}
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
	
	public void setMinimumNumberOfSentences(int number){
		m_stnSizeThreshold = number;
	}
	
	abstract public void LoadDoc(String filename);
	
	//Add one more token to the current vocabulary.
	protected void expandVocabulary(String token) {
		m_featureNameIndex.put(token, m_featureNames.size()); // set the index of the new feature.
		m_featureNames.add(token); // Add the new feature.
		m_featureStat.put(token, new _stat(m_classNo));
	}
		
	//Return corpus without parameter and feature selection.
	public _Corpus returnCorpus(String finalLocation) throws FileNotFoundException {
		SaveCVStat(finalLocation);
		
		int sum = 0;
		for(int c:m_classMemberNo) {
			System.out.print(c + " ");
			sum += c;
		}
		System.out.println(", Total: " + sum);
		
		return getCorpus();
	}
	
	public _Corpus getCorpus() {
		//store the feature names into corpus
		m_corpus.setFeatures(m_featureNames);
		m_corpus.setFeatureStat(m_featureStat);
		m_corpus.setMasks(); // After collecting all the documents, shuffle all the documents' labels.
		m_corpus.setContent(!m_releaseContent);
		return m_corpus;
	}
	
	void rollBack(HashMap<Integer, Double> spVct, int y){
		if (!m_isCVLoaded) {
			for(int index: spVct.keySet()){
				String token = m_featureNames.get(index);
				_stat stat = m_featureStat.get(token);

				if(Utils.sumOfArray(stat.getDF())==1){//If the feature is the first time to show in feature set.
					m_featureNameIndex.remove(index);
					m_featureStat.remove(token);
					m_featureNames.remove(index);
				} else{//If the feature is not the first time to show in feature set.
					stat.minusOneDF(y);
					stat.minusNTTF(y, spVct.get(index));
				}
			}
		} else{//If CV is loaded and CV's statistics are loaded from file, no need to change it
			if (m_isCVStatLoaded)
				return;
			
			// otherwise, we can minus the DF and TTF directly.
			for(int index: spVct.keySet()){
				String token = m_featureNames.get(index);
				_stat stat = m_featureStat.get(token);
				stat.minusOneDF(y);
				stat.minusNTTF(y, spVct.get(index));
			}
		}
	}
	
	//Give the option, which would be used as the method to calculate feature value and returned corpus, calculate the feature values.
	public void setFeatureValues(String fValue, int norm) {
		ArrayList<_Doc> docs = m_corpus.getCollection(); // Get the collection of all the documents.
		int N = m_isCVStatLoaded ? m_TotalDF : docs.size();

		if (fValue.equals("TF")){
			//the original feature is raw TF
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N + 1) / DF);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("TFIDF")) {
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
//					double TF =  Math.log10(sf.getValue())+ 1 ;// normalized TF
//					double DF = Utils.sumOfArray(stat.getDF());
//					double IDF = Math.log10((25265)/DF)+ 1;
					double TF = sf.getValue() / temp.getTotalDocLength();// normalized TF
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N + 1) / DF);
					double TFIDF = TF * IDF;
					sf.setValue(TFIDF);
					avgIDF += IDF;
//					(1+Math.log10(r.m_VSM.get(key)))*(1+Math.log10(Config.NumberOfReviewsInTraining/m_Vocabs.get(key).getValue())));
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("TFIDF-sublinear")) {
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = 1 + Math.log10(sf.getValue());// sublinear TF
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = 1 + Math.log10(N / DF);
					double TFIDF = TF * IDF;
					sf.setValue(TFIDF);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("BM25")) {
			double k1 = 1.5; // [1.2, 2]
			double b = 0.75; // (0, 1000]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg, avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N - DF + 0.5) / (DF + 0.5));
					double BM25 = IDF * TF * (k1 + 1) / (k1 * (1 - b + b * n) + TF);
					sf.setValue(BM25);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else if (fValue.equals("PLN")) {
			double s = 0.5; // [0, 1]
			// Iterate all the documents to get the average document length.
			double navg = 0;
			for (int k = 0; k < N; k++)
				navg += docs.get(k).getTotalDocLength();
			navg /= N;

			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double n = temp.getTotalDocLength() / navg, avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double TF = sf.getValue();
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N + 1) / DF);
					double PLN = (1 + Math.log(1 + Math.log(TF)) / (1 - s + s * n)) * IDF;
					sf.setValue(PLN);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		} else {
			System.out.println("No feature value is set, keep the raw count of every feature.");
			//the original feature is raw TF
			for (int i = 0; i < docs.size(); i++) {
				_Doc temp = docs.get(i);
				_SparseFeature[] sfs = temp.getSparse();
				double avgIDF = 0;
				for (_SparseFeature sf : sfs) {
					String featureName = m_featureNames.get(sf.getIndex());
					_stat stat = m_featureStat.get(featureName);
					double DF = Utils.sumOfArray(stat.getDF());
					double IDF = Math.log((N + 1) / DF);
					avgIDF += IDF;
				}
				
				//compute average IDF
				temp.setAvgIDF(avgIDF/sfs.length);
			}
		}
		
		//rank the documents by product and time in all the cases
		//Collections.sort(m_corpus.getCollection());
		if (norm == 1){
			for(_Doc d:docs)			
				Utils.L1Normalization(d.getSparse());
		} else if(norm == 2){
			for(_Doc d:docs)			
				Utils.L2Normalization(d.getSparse());
		} else {
			System.out.println("No normalizaiton is adopted here or wrong parameters!");
		}
		System.out.format("Text feature generated for %d documents...\n", m_corpus.getSize());
	}
	
	//Select the features and store them in a file.
	public void featureSelection(String location, String featureSelection, double startProb, double endProb, int threshold) throws FileNotFoundException {
		FeatureSelector selector = new FeatureSelector(startProb, endProb, threshold);

		System.out.println("*******************************************************************");
		if (featureSelection.equals("DF"))
			selector.DF(m_featureStat);
		else if (featureSelection.equals("IG"))
			selector.IG(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("MI"))
			selector.MI(m_featureStat, m_classMemberNo);
		else if (featureSelection.equals("CHI"))
			selector.CHI(m_featureStat, m_classMemberNo);
		
		m_featureNames = selector.getSelectedFeatures();
		SaveCV(location, featureSelection, startProb, endProb, threshold); // Save all the features and probabilities we get after analyzing.
		System.out.println(m_featureNames.size() + " features are selected!");
		
		//clear memory for next step feature construction
//		reset();
//		LoadCV(location);//load the selected features
	}
	
	//Save all the features and feature stat into a file.
	protected void SaveCV(String featureLocation, String featureSelection, double startProb, double endProb, int threshold) throws FileNotFoundException {
		if (featureLocation==null || featureLocation.isEmpty())
			return;
		
		System.out.format("Saving controlled vocabulary to %s...\n", featureLocation);
		PrintWriter writer = new PrintWriter(new File(featureLocation));
		//print out the configurations as comments
		writer.format("#NGram:%d\n", m_Ngram);
		writer.format("#Selection:%s\n", featureSelection);
		writer.format("#Start:%f\n", startProb);
		writer.format("#End:%f\n", endProb);
		writer.format("#DF_Cut:%d\n", threshold);
		
		//print out the features
		for (int i = 0; i < m_featureNames.size(); i++)
			writer.println(m_featureNames.get(i));
		writer.close();
	}
	
	//Save all the features and feature stat into a file.
	public void SaveCVStat(String fvStatFile) {
		if (fvStatFile==null || fvStatFile.isEmpty())
			return;
		
		try {
			PrintWriter writer = new PrintWriter(new File(fvStatFile));
		
			for(int i = 0; i < m_featureNames.size(); i++){
				writer.print(m_featureNames.get(i));
				_stat temp = m_featureStat.get(m_featureNames.get(i));
				for(int j = 0; j < temp.getDF().length; j++)
					writer.print("\t" + temp.getDF()[j]);
				for(int j = 0; j < temp.getTTF().length; j++)
					writer.print("\t" + temp.getTTF()[j]);
				writer.println();
			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	//Return the number of features.
	public int getFeatureSize(){
		return m_featureNames.size();
	}
	
	//Sort the documents.
	public void setTimeFeatures(int window){//must be called before return corpus
		if (window<1) 
			return;
		
		//Sort the documents according to time stamps.
		ArrayList<_Doc> docs = m_corpus.getCollection();
		
		/************************time series analysis***************************/
		double norm = 1.0 / m_classMemberNo.length, avg = 0;
		int count = 0;//for computing the moving average
		String lastItemID = null;
		for(int i = 0; i < docs.size(); i++){
			_Doc doc = docs.get(i);			
			
			if (lastItemID == null)
				lastItemID = doc.getItemID();
			else if (lastItemID != doc.getItemID()) {
				m_preDocs.clear(); // reviews for a new category of products
				lastItemID = doc.getItemID();
				
				//clear for moving average
				avg = 0;
				count = 0;
			}
			
			avg += doc.getYLabel();
			count += 1;
			
			if(m_preDocs.size() < window){
				m_preDocs.add(doc);
				m_corpus.removeDoc(i);
				m_classMemberNo[doc.getYLabel()]--;
				i--;
			} else{
				doc.createSpVctWithTime(m_preDocs, m_featureNames.size(), avg/count, norm);
				m_preDocs.remove();
				m_preDocs.add(doc);
			}
		}
		System.out.format("Time-series feature set for %d documents!\n", m_corpus.getSize());
	}
	
	// added by Md. Mustafizur Rahman for Topic Modelling
	public double[] getBackgroundProb()
	{
		double back_ground_probabilty [] = new double [m_featureNameIndex.size()];
		
		for(int i = 0; i<m_featureNameIndex.size();i++)
		{
			String featureName = m_featureNames.get(i);
			_stat stat =  m_featureStat.get(featureName);
			back_ground_probabilty[i] = Utils.sumOfArray(stat.getTTF());
			
			if (back_ground_probabilty[i] < 0)
				System.err.println();
		}
		
		double sum = Utils.sumOfArray(back_ground_probabilty) + back_ground_probabilty.length;//add one smoothing
		for(int i = 0; i<m_featureNameIndex.size();i++)
			back_ground_probabilty[i] = (1.0 + back_ground_probabilty[i]) / sum;
		return back_ground_probabilty;
	}
	
	//added by Lin.
	public HashMap<_Pair, Integer> returnLCSMap(){
		System.out.format("LCS map has %d pairs.\n", m_LCSMap.size());
		return m_LCSMap;
	}
	
	public void saveTopicVectors(String filename) throws FileNotFoundException {
		if(filename == null || filename.isEmpty())
			return;
		PrintWriter writer = new PrintWriter(new File(filename));
		for(_Doc d: m_corpus.getCollection()){
			writer.write(d.getID()+"\t");
			for(double v: d.m_topics){
				writer.write(v+"\t");
			}
			writer.write("\n");
		}
		writer.close();
	}
	
	public void loadTopicVectors(String filename, int topicNu){
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			int index = 0;
			double[] topicVector;
			while ((line = reader.readLine()) != null) {
				String[] topics = line.split("\t");
				if(topics.length == (topicNu+1)){
					index = Integer.valueOf(topics[0]);
					topicVector = new double[topicNu];
					for(int i=1; i < topicNu; i++)
						topicVector[i-1] = Double.valueOf(topics[i]);
					m_corpus.getCollection().get(index).setTopics(topicVector);
				} else 
					System.out.println("The dimension of topic vector is wrong!");
			}
			reader.close();
			System.out.format("%d topic vectors loaded from %s...\n", index+1 , filename);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	// Added by Lin for loading the assigned sentiment labels for each sentence in the corpus.
	public HashMap<String, int[]> loadStnLabels(String filename){
		if (filename==null || filename.isEmpty())
			return null;
		
		try {
			HashMap<String, int[]> docStnLabelsMap = new HashMap<String, int[]>();
			int count = 0;
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line, docName;
			String[] strs;
			int[] labels;
			while ((line = reader.readLine()) != null) {
				strs = line.split("\t");
				docName = strs[0];
				labels = new int[strs.length-1];
				for(int i=1; i<strs.length; i++){
					labels[i-1] = Integer.valueOf(strs[i]);
				}
				docStnLabelsMap.put(docName, labels);
				count++;
			}
			reader.close();
			System.out.format("Sentence labels for %d documents are loaded from %s.\n", count, filename);
			return docStnLabelsMap;
 
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
			return null;
		}
	}
	
	// We get the overlapping of corpus and documents which have sentence labels.
	public void documentsMatch(HashMap<String, int[]> map, int k){
		ArrayList<Integer> lst = new ArrayList<Integer>();
		String docName;
		for(_Doc d: m_corpus.getCollection()){
			docName = d.getName();
			if(map.containsKey(docName)){
				d.setStnLabels(map.get(docName));
				d.setPosRatio(k);
			} else{
				lst.add(d.getID());
			}
		}
		// Sort the missing documents in descending order and remove them.
		Collections.sort(lst, Collections.reverseOrder());
		for(int id: lst)
			m_corpus.removeDoc(id);
	}
	
	
	// Assign the documents to different groups based on the pos/neg ratio.
	public void assignGroup(double threshold){
		for(_Doc d: m_corpus.getCollection()){
			if(d.getPosRatio() <= threshold || Math.abs(1-d.getPosRatio()) <= threshold)
				d.setClusterNo(0);
//				d.setGroupNo(0);
			else
				d.setClusterNo(1);
//				d.setGroupNo(1);
		}
	}
	//Return features.
	public ArrayList<String> getFeatures(){
		return m_featureNames;
	}
}
