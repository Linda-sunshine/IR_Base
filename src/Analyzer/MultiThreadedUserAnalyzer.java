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
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import structures.MyPriorityQueue;
import structures.TokenizeResult;
import structures._Doc;
import structures._RankItem;
import structures._Review;
import structures._Review.rType;
import structures._User;
import utils.Utils;

/**
 * @author Mohammad Al Boni
 * Multi-threaded extension of UserAnalyzer
 */
public class MultiThreadedUserAnalyzer extends UserAnalyzer {

	protected int m_numberOfCores;
	protected Tokenizer[] m_tokenizerPool;
	protected SnowballStemmer[] m_stemmerPool;
	protected Object m_allocReviewLock=null;
	protected Object m_corpusLock=null;
	protected Object m_rollbackLock=null;
	private Object m_featureStatLock=null;
	
	public MultiThreadedUserAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold, int numberOfCores, boolean b)
					throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold, b);
		
		m_numberOfCores = numberOfCores;
		
		// since DocAnalyzer already contains a tokenizer, then we can user it and define a pool with length of m_numberOfCores - 1
		m_tokenizerPool = new Tokenizer[m_numberOfCores-1]; 
		m_stemmerPool = new SnowballStemmer[m_numberOfCores-1];
		for(int i=0;i<m_numberOfCores-1;++i){
			m_tokenizerPool[i] = new TokenizerME(new TokenizerModel(new FileInputStream(tokenModel)));
			m_stemmerPool[i] = new englishStemmer();
		}
		
		m_allocReviewLock = new Object();// lock when collecting review statistics
		m_corpusLock = new Object(); // lock when collecting class statistics 
		m_rollbackLock = new Object(); // lock when revising corpus statistics
		m_featureStatLock = new Object();
	}
	
	//Load all the users.
	@Override
	public void loadUserDir(String folder){
		if(folder == null || folder.isEmpty())
			return;
		
		loadCategory();
		File dir = new File(folder);
		final File[] files=dir.listFiles();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<m_numberOfCores;++i){
			threads.add(  (new Thread() {
				int core;
				@Override
				public void run() {
					try {
						for (int j = 0; j + core <files.length; j += m_numberOfCores) {
							File f = files[j+core];
							// && f.getAbsolutePath().endsWith("txt")
							if(f.isFile()){//load the user								
								loadUser(f.getAbsolutePath(),core);
							}
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
				
				private Thread initialize(int core ) {
					this.core = core;
					return this;
				}
			}).initialize(i));
			
			threads.get(i).start();
		}
		for(int i=0;i<m_numberOfCores;++i){
			try {
				threads.get(i).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		} 
		
		// process sub-directories
		int count=0;
		for(File f:files ) 
			if (f.isDirectory())
				loadUserDir(f.getAbsolutePath());
			else
				count++;

		System.out.format("%d users are loaded from %s...\n", count, folder);
	}
	

	// Load one file as a user here. 
	private void loadUser(String filename, int core){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String userID = extractUserID(file.getName()); //UserId is contained in the filename.				
			
			// Skip the first line since it is user name.
			reader.readLine(); 

			String productID, source, category="";
			ArrayList<_Review> reviews = new ArrayList<_Review>();
//			ArrayList<Integer> ctgIndexes = new ArrayList<Integer>();
			_Review review;
			int ylabel;
			long timestamp=0;
			while((line = reader.readLine()) != null){
				productID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());
			
				m_ctgNames.add(category);
				
				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
					review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, productID, category, timestamp);
					if(AnalyzeDoc(review,core)){ //Create the sparse vector for the review.
						reviews.add(review);
//						ctgIndexes.add(m_ctgIndex.get(category));
					}
				}
			}

//			if(reviews.size() > 10){//at least one for adaptation and one for testing
//				synchronized (m_allocReviewLock) {
//					allocateReviews(reviews);				
//					m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
//				}
//			} else{// added by Lin, for those users with fewer than 2 reviews, ignore them.
//				for(_Review r: reviews){			
//					synchronized (m_rollbackLock) {
//						rollBack(Utils.revertSpVct(r.getSparse()), r.getYLabel());
//					}
//				}
//			}
			if(reviews.size() > 1){//at least one for adaptation and one for testing
				synchronized (m_allocReviewLock) {
					allocateReviews(reviews);	
					m_users.add(new _User(userID, m_classNo, reviews)); //create new user from the file.
//					m_users.add(new _User(userID, m_classNo, reviews, ctgIndexes)); //create new user from the file.
				}
			} else if(reviews.size() == 1){// added by Lin, for those users with fewer than 2 reviews, ignore them.
				review = reviews.get(0);
				synchronized (m_rollbackLock) {
					rollBack(Utils.revertSpVct(review.getSparse()), review.getYLabel());
				}
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//Tokenizing input text string
	private String[] Tokenizer(String source, int core){
		String[] tokens = getTokenizer(core).tokenize(source);
		return tokens;
	}
	
	//Snowball Stemmer.
	private String SnowballStemming(String token, int core){
		SnowballStemmer stemmer = getStemmer(core);
		stemmer.setCurrent(token);
		if(stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//Given a long string, tokenize it, normalie it and stem it, return back the string array.
	protected TokenizeResult TokenizerNormalizeStemmer(String source, int core){
		String[] tokens = Tokenizer(source, core); //Original tokens.
		TokenizeResult result = new TokenizeResult(tokens);

		//Normalize them and stem them.		
		for(int i = 0; i < tokens.length; i++)
			tokens[i] = SnowballStemming(Normalize(tokens[i]),core);
		
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
	
	/*Analyze a document and add the analyzed document back to corpus.*/
	protected boolean AnalyzeDoc(_Doc doc, int core) {
		TokenizeResult result = TokenizerNormalizeStemmer(doc.getSource(),core);// Three-step analysis.
		String[] tokens = result.getTokens();
		int y = doc.getYLabel();

		// Construct the sparse vector.
		HashMap<Integer, Double> spVct = constructSpVct(tokens, y, null);
		
		if (spVct.size()>m_lengthThreshold) {//temporary code for debugging purpose
			doc.createSpVct(spVct);
			doc.setStopwordProportion(result.getStopwordProportion());
			synchronized (m_corpusLock) {
				m_corpus.addDoc(doc);
				m_classMemberNo[y]++;
			}
			if (m_releaseContent)
				doc.clearSource();
			
			return true;
		} else {
			/****Roll back here!!******/
			synchronized (m_rollbackLock) {
				rollBack(spVct, y);
			}
			return false;
		}
	}
	
	//convert the input token sequence into a sparse vector (docWordMap cannot be changed)
	// Since multiple threads access the featureStat, we need lock for this variable.
	@Override
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
							if(m_featureStat.containsKey(0)){
								synchronized(m_featureStatLock){
									m_featureStat.get(token).addOneDF(y);
								}
							}
						}
					}
				} else {// indicate we allow the analyzer to dynamically expand the feature vocabulary
					expandVocabulary(token);// update the m_featureNames.
					index = m_featureNameIndex.get(token);
					spVct.put(index, 1.0);
					if(m_featureStat.containsKey(token)){
						synchronized(m_featureStatLock){
							m_featureStat.get(token).addOneDF(y);
						}
					}
				}
				if(m_featureStat.containsKey(token)){
					synchronized(m_featureStatLock){
						m_featureStat.get(token).addOneTTF(y);
					}
				}
			} else if (m_featureNameIndex.containsKey(token)) {// CV is loaded.
				index = m_featureNameIndex.get(token);
				if (spVct.containsKey(index)) {
					value = spVct.get(index) + 1;
					spVct.put(index, value);
				} else {
					spVct.put(index, 1.0);
					if (!m_isCVStatLoaded && (docWordMap==null || !docWordMap.containsKey(index))){
						synchronized(m_featureStatLock){
							m_featureStat.get(token).addOneDF(y);
						}
					}
				}
				
				if (!m_isCVStatLoaded){
					synchronized(m_featureStatLock){
						m_featureStat.get(token).addOneTTF(y);
					}
				}
			}
			// if the token is not in the vocabulary, nothing to do.
		}
		return spVct;
	}
	
	
	// return a tokenizer using the core number
	private Tokenizer getTokenizer(int index){
		if(index==m_numberOfCores-1)
			return m_tokenizer;
		else
			return m_tokenizerPool[index];
	}
	
	// return a stemmer using the core number
	private SnowballStemmer getStemmer(int index){
		if(index==m_numberOfCores-1)
			return m_stemmer;
		else
			return m_stemmerPool[index];
	}
	// Added by Lin for constructing the bow profile for each user.
	public void constructSparseVector4Users() {
		for (_User u : m_users)
			u.constructSparseVector();
	}

	protected HashMap<String, Integer> m_userIDIndex;
	// Added by Lin. Load user weights from learned models to construct neighborhood.
	public void loadUserWeights(String folder, String suffix){
		if(folder == null || folder.isEmpty())
			return;
		String userID;
		int userIndex, count = 0;
		double[] weights;
		constructUserIDIndex();
		File dir = new File(folder);
		
		if(!dir.exists()){
			System.err.print("[Info]Directory doesn't exist!");
		} else{
			for(File f: dir.listFiles()){
				if(f.isFile() && f.getName().endsWith(suffix)){
					int endIndex = f.getName().lastIndexOf(".");
					userID = f.getName().substring(0, endIndex);
					if(m_userIDIndex.containsKey(userID)){
						userIndex = m_userIDIndex.get(userID);
						weights = loadOneUserWeight(f.getAbsolutePath());
						m_users.get(userIndex).setSVMWeights(weights);
						count++;
					}
				}
			}
		}
		System.out.format("%d users weights are loaded!\n", count);
	}
	public void constructUserIDIndex(){
		m_userIDIndex = new HashMap<String, Integer>();
		for(int i=0; i<m_users.size(); i++)
			m_userIDIndex.put(m_users.get(i).getUserID(), i);
	}
	
	// Added by Lin. Load one user's weights.
	public double[] loadOneUserWeight(String fileName) {
		double[] weights = new double[getFeatureSize()];
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(fileName), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				String[] ws = line.split(",");
				if (ws.length != getFeatureSize() + 1)
					System.out.println("[error]Wrong dimension of the user's weights!");
				else {
					weights = new double[ws.length];
					for (int i = 0; i < ws.length; i++) {
						weights[i] = Double.valueOf(ws[i]);
					}
				}
			}
			reader.close();
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", fileName);
			e.printStackTrace();
		}
		return weights;
	}
	// assign some of the users for testing only.
	public void separateUsers(int k){
		int count = 0;
		double light = 0, medium = 0;
		while(count < k){
			if(m_users.get(count).getReviewSize() <= 10)
				light++;
			else if(m_users.get(count).getReviewSize() <= 50)
				medium++;
			for(_Review r: m_users.get(count++).getReviews()){
				if(r.getType() == rType.ADAPTATION){
					m_adaptSize--;
					r.setType(rType.TEST);
					m_testSize++;
					if(r.getYLabel() == 1){
						m_pCount[1]--;
						m_pCount[2]++;
					}
				}
			}
		}
		System.out.print(String.format("[Prob Info]Light: %.4f, medium: %.4f, heavy: %.4f\n", light/k, medium/k, (k-light-medium)/k));
	}
	
	
	/***Category related information.***/
	HashSet<String> m_ctgNames = new HashSet<String>();
	HashMap<String, Integer> m_ctgIndex = new HashMap<String, Integer>();
	String m_ctgFile = "./data/category_Amazon.txt";
	
	public void setCtgFile(String fv){
		m_ctgFile = fv;
	}
	//Load category information.
	public void loadCategory(){
		try{
			File file = new File(m_ctgFile);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			while((line = reader.readLine()) != null){
				m_ctgIndex.put(line, m_ctgIndex.size());
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
		System.out.println(m_ctgIndex.size() + " categories are loaded!");
	}
		
	// Get the number of reviews each category has.
	public void printCategoryInfo(){
		String[] names = new String[m_ctgIndex.size()];
		for(String s: m_ctgIndex.keySet())
			names[m_ctgIndex.get(s)] = s;
		
		int[] ctgStat = new int[m_ctgIndex.size()];
		for(_User u: m_users){
			for(_Review r: u.getReviews())
				ctgStat[m_ctgIndex.get(r.getCategory())]++;
		}
		MyPriorityQueue<_RankItem> q = new MyPriorityQueue<_RankItem>(m_ctgNames.size());
		for(int i=0; i<ctgStat.length; i++)
			q.add(new _RankItem(i, ctgStat[i]));
		
		for(_RankItem it: q)
			System.out.println(names[it.m_index] + "\t" + it.m_value);
	}
	
	public void saveCategory(String filename){
		try{
			PrintWriter writer = new PrintWriter(new File(filename));
			for(String c: m_ctgNames)
				writer.write(c+"\n");
			writer.close();
			System.out.println(m_ctgNames.size() + " categories are saved.\n");
		} catch(IOException e){
			e.printStackTrace();
		}
			
	}
	int[][] m_ctgStat = new int[2][2]; // global variable to record the ctg info.
	HashSet<String> train = new HashSet<String>();
	HashSet<String> test = new HashSet<String>();
	// Correglation analysis of the review categories.
	public void CtgCorrelation(){
		for(_User u: m_users){
			train.clear();
			test.clear();
			for(_Review r: u.getReviews()){
				if(r.getType() == rType.ADAPTATION){
					train.add(r.getCategory());
				} else{
					test.add(r.getCategory());
				}
			}
			for(_Review r: u.getReviews()){
				if(r.getType() == rType.ADAPTATION){
					if(test.contains(r.getCategory()))
						m_ctgStat[0][0]++;
					else
						m_ctgStat[0][1]++;
				} else{
					if(train.contains(r.getCategory()))
						m_ctgStat[0][0]++;
					else
						m_ctgStat[1][0]++;
				}
			}
		}
		
		for(int[] c: m_ctgStat){
			System.out.print(String.format("%d\t%d\n", c[0], c[1]));
		}
	}
	
	// key: user id; value: friends array.
	HashMap<String, String[]> m_neighborsMap = new HashMap<String, String[]>();
	public void buildFriendship(String filename){
		try{
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			String[] users, friends;
			while((line = reader.readLine()) != null){
				users = line.trim().split("\t");
				friends = Arrays.copyOfRange(users, 1, users.length);
				m_neighborsMap.put(users[0], friends);
			}
			reader.close();
			// map friends to users.
			for(_User u: m_users){
				if(m_neighborsMap.containsKey(u.getUserID()))
					u.setFriends(m_neighborsMap.get(u.getUserID()));
				else
					System.out.println("The user does not have any friends.");
			}
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void checkFriendship(){
		double sum = 0, miss = 0;
		for(String uid: m_neighborsMap.keySet()){
			for(String frd: m_neighborsMap.get(uid)){
				sum++;
				if(!m_neighborsMap.containsKey(frd))
					miss++;
			}
		}
		System.out.println("The avg friend size is: " + (sum-miss)/m_neighborsMap.size());
	}
}
