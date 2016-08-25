package Analyzer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.TreeMap;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import Classifier.supervised.modelAdaptation.CoAdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;

import opennlp.tools.tokenize.Tokenizer;
import opennlp.tools.tokenize.TokenizerME;
import opennlp.tools.tokenize.TokenizerModel;
import opennlp.tools.util.InvalidFormatException;
import structures.MyPriorityQueue;
import structures.TokenizeResult;
import structures._Doc;
import structures._RankItem;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import structures._stat;
import utils.Utils;

/**
 * @author Mohammad Al Boni
 * Multi-threaded extension of UserAnalyzer
 */
public class MultiThreadedUserAnalyzer extends UserAnalyzer {
	protected ArrayList<String> m_categories;
	protected boolean m_ctgFlag = false; // Whether category is loaded or not.
	protected int m_numberOfCores;
	protected Tokenizer[] m_tokenizerPool;
	protected SnowballStemmer[] m_stemmerPool;
	protected Object m_allocReviewLock=null;
	private Object m_corpusLock=null;
	private Object m_rollbackLock;
	protected int m_start = 0, m_end = Integer.MAX_VALUE; // Added by Lin for filtering reviews.
	protected double m_globalLen = 0, m_maxLen = 0;
	protected double[][] m_userWeights;
	protected HashMap<String, Integer> m_userIDIndex;
	protected int[] m_ctgCounts;
	protected int m_ctgThreshold = Integer.MAX_VALUE; // added by Lin, the category threshold for selecting users.
	
	public MultiThreadedUserAnalyzer(String tokenModel, int classNo,
			String providedCV, int Ngram, int threshold, int numberOfCores)
					throws InvalidFormatException, FileNotFoundException, IOException {
		super(tokenModel, classNo, providedCV, Ngram, threshold);
		
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
	
	public ArrayList<String> getCategory(){
		return m_categories;
	}

	// return a stemmer using the core number
	protected SnowballStemmer getStemmer(int index) {
		if (index == m_numberOfCores - 1)
			return m_stemmer;
		else
			return m_stemmerPool[index];
	}
	
	// return a tokenizer using the core number
	protected Tokenizer getTokenizer(int index) {
		if (index == m_numberOfCores - 1)
			return m_tokenizer;
		else
			return m_tokenizerPool[index];
	}
	
	//Load all the users.
	@Override
	public void loadUserDir(String folder){
		int max = 0;
		if(folder == null || folder.isEmpty())
			return;
		File dir = new File(folder);
		final File[] files=dir.listFiles();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int i=0;i<m_numberOfCores;++i){
			threads.add(  (new Thread() {
				int core;
				public void run() {
					try {
						for (int j = 0; j + core <files.length; j += m_numberOfCores) {
							File f = files[j+core];
							if(f.isFile()){//load the user								
								loadOneUser(f.getAbsolutePath(),core);
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
		int rvwCount = 0;
		for(_User u: m_users){
			rvwCount += u.getReviewSize();
			if(u.getReviewSize() > max)
				max = u.getReviewSize();
		}
		System.out.println("Max review size: " + max);
		System.out.format("Average review length over all users is %.3f, max average length is %.3f.\n", m_globalLen/rvwCount, m_maxLen);
		System.out.format("[Info]Start: %d, End: %d, (%d/%d) users and %d reviews are loaded from %s...\n", m_start, m_end, m_users.size(), count, rvwCount, folder);
	}
	
	// Load one file as a user here. 
	public void loadOneUser(String filename, int core){
		try {
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;			
			String userID = extractUserID(file.getName()); //UserId is contained in the filename.
			// Skip the first line since it is user name.
			reader.readLine(); 

			String productID, source, category;
			if(m_ctgFlag)
				m_ctgCounts = new int[m_categories.size()];
			ArrayList<_Review> reviews = new ArrayList<_Review>();
			_Review review;
			int ylabel;
			long timestamp;
			double localSize = 0, localLength = 0, localAvg = 0;
			while((line = reader.readLine()) != null){
				productID = line;
				source = reader.readLine(); // review content
				category = reader.readLine(); // review category
				if(m_categories.contains(category))
					m_ctgCounts[m_categories.indexOf(category)]++;
				
				ylabel = Integer.valueOf(reader.readLine());
				timestamp = Long.valueOf(reader.readLine());

				// Construct the new review.
				if(ylabel != 3){
					ylabel = (ylabel >= 4) ? 1:0;
					review = new _Review(m_corpus.getCollection().size(), source, ylabel, userID, productID, category, timestamp);
					if(AnalyzeDoc(review,core)){ //Create the sparse vector for the review.
						reviews.add(review);
						localLength += review.getDocLength();
						localSize++;
					}
				}
			}
			localAvg = localLength / localSize;
			
			// Added by Lin for debugging.
			if(reviews.size() > 1 && (localAvg < m_end) && (localAvg > m_start)){//at least one for adaptation and one for testing
				if( localAvg > m_maxLen)
					m_maxLen = localLength / localSize;
				m_globalLen += localLength;
				synchronized (m_allocReviewLock) {
					allocateReviews(reviews);			
					m_users.add(new _User(userID, m_classNo, reviews, m_ctgCounts));
				}
			}
			reader.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	//Snowball Stemmer.
	protected String SnowballStemming(String token, int core){
		SnowballStemmer stemmer = getStemmer(core);
		stemmer.setCurrent(token);
		if(stemmer.stem())
			return stemmer.getCurrent();
		else
			return token;
	}
	
	//Tokenizing input text string
	protected String[] Tokenizer(String source, int core){
		String[] tokens = getTokenizer(core).tokenize(source);
		return tokens;
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
	
	// Added by Lin for fitlering reviews.
	public void setRvwLenghRange(int start, int end){
		m_start = start;
		m_end = end;
	}
	
	// Added by Lin.
	public void loadCategory(String filename){
		m_categories = new ArrayList<String>();
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			while ((line = reader.readLine()) != null) {
				m_categories.add(line);
			}
			reader.close();
			System.out.println(m_categories.size() + " categories are loaded.");
			m_ctgCounts = new int[m_categories.size()];
			m_ctgFlag = true;
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
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
	
//	public void loadUserWeightsMultiThreads(String folder, String sfx){
//		if(folder == null || folder.isEmpty())
//			return;
//		File dir = new File(folder);
//		final File[] files=dir.listFiles();
//		final String userID, suffix = sfx;
//		final int userIndex, endIndex;
//		final double[] weights;
//		constructUserIDIndex();
//		
//		if(!dir.exists()){
//			System.err.print("[Info]Directory doesn't exist!");
//		} else{
//			ArrayList<Thread> threads = new ArrayList<Thread>();
//			for(int i=0;i<m_numberOfCores;++i){
//				threads.add(  (new Thread() {
//					int core;
//					public void run() {
//						try {
//							for (int j = 0; j + core <files.length; j += m_numberOfCores) {
//								File f = files[j+core];
//								if(f.isFile() && f.getName().endsWith(suffix)){
//									endIndex = f.getName().lastIndexOf(".");
//									userID = f.getName().substring(0, endIndex);
//									if(m_userIDIndex.containsKey(userID)){
//										userIndex = m_userIDIndex.get(userID);
//										if(f.isFile()){//load the user								
//											weights = loadOneUserWeight(f.getAbsolutePath());
//											m_users.get(userIndex).setSVMWeights(weights);
//										}
//									}
//								}
//							}
//						} catch(Exception ex) {
//							ex.printStackTrace(); 
//						}
//					}
//				
//					private Thread initialize(int core ) {
//						this.core = core;
//						return this;
//					}
//				}).initialize(i));
//				threads.get(i).start();
//			}
//			for(int i=0;i<m_numberOfCores;++i){
//				try {
//					threads.get(i).join();
//				} catch (InterruptedException e) {
//					e.printStackTrace();
//				} 
//			}
//		}
//	}
	
	// Added by Lin for neighborhood based on SVM weights.
	public void findSVMNeighbors(final int topK){
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;

				@Override
				public void run() {
					_User ui;
					try {
						for(int i=0; i + core<m_users.size(); i += numOfCores){
							ui = m_users.get(i+core);
							ui.initSVMNeighbors(topK);
							for(int j=0; j<m_users.size(); j++){
								if(j != i+core)
									ui.addSVMNeighbors(j, Utils.cosine(ui.getSVMWeights(), m_users.get(j).getSVMWeights()));
							}
						}
					} catch(Exception ex) {
						ex.printStackTrace(); 
					}
				}
			
				private Thread initialize(int core, int numOfCores) {
					this.core = core;
					this.numOfCores = numOfCores;
					return this;
				}
			}).initialize(k, numberOfCores));
		
			threads.get(k).start();
		}
	
		for(int k=0;k<numberOfCores;++k){
			try {
				threads.get(k).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			} 
		}
		System.out.format("[Info]SVM neighbors are constructed for %d users...\n", m_users.size());
	}
	
	// Added by Lin. Load one user's weights.
	public double[] loadOneUserWeight(String fileName){
		double[] weights = new double[getFeatureSize()];
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
			String line;
			while((line = reader.readLine()) != null){
				String[] ws = line.split(",");
				if(ws.length != getFeatureSize()+1)
					System.out.println("[error]Wrong dimension of the user's weights!");
				else{
					weights = new double[ws.length];
					for(int i=0; i<ws.length; i++){
						weights[i] = Double.valueOf(ws[i]);
					}
				}
			}
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", fileName);
			e.printStackTrace();
		}
		return weights;
	}
	
	public void constructUserIDIndex(){
		m_userIDIndex = new HashMap<String, Integer>();
		for(int i=0; i<m_users.size(); i++)
			m_userIDIndex.put(m_users.get(i).getUserID(), i);
	}
	
	public void saveSFVct(String folder){
		for(_User u:m_users) {
			try {
				BufferedWriter writer = new BufferedWriter(new FileWriter(folder+"/"+u.getUserID()+".txt"));
		        StringBuilder buffer = new StringBuilder(512);
		        _SparseFeature[] sfs = u.getBoWProfile();
		        for(int i=0; i<sfs.length; i++) {
		        	buffer.append(sfs[i].getIndex()+","+sfs[i].getValue());
		        	if (i<sfs.length-1)
		            	buffer.append(',');
		        }
		        writer.write(buffer.toString());
		        writer.close();
		    } catch (Exception e) {
		        e.printStackTrace(); 
		    } 
		}
		System.out.format("[Info]Save sparse features of users to %s.", folder);
	}
	
	// Added by Lin.
	public void printPosRatio(){
		PrintWriter writer;
		try{
			writer = new PrintWriter(new File("posRatio.txt"));
			int count = 0;
			for(_User u: m_users){
				writer.write(String.format("%d, %d, %.4f\n", count, u.getReviewSize(), u.getPosRatio()));
				count++;
			}
			writer.close();
		} catch(FileNotFoundException e){
			e.printStackTrace();
		}
	}
	
	// Added by Lin, calculate the overall pos/neg ratio of reviews.
	public double calcRatio(){
		double neg = 0, pos = 0;
		for(_User u: m_users){
			for(_Review r: u.getReviews()){
				if(r.getYLabel() == 0)
					neg++;
				else 
					pos++;
			}
		}
		return pos / neg;
	}
	
	// Added by Lin, print the counts of users with different categories.
	public void printCategoryStat(){
		PrintWriter writer;
		try{
			writer = new PrintWriter(new File("./data/ctgStat.txt"));
			for(_User u: m_users){
				for(int i: u.getCategory())
					writer.write(i+"\t");
				writer.write("\n");
			}
			writer.close();
		} catch(Exception e){
			e.printStackTrace();
		}
	}
	
	// Added by Lin, set the threshold for category counts.
	public void setCtgThreshold(int k){
		m_ctgThreshold = k;
	}
	
	// Added by Lin for constructing the 
	public void constructSparseVector4Users(){
		for(_User u: m_users)
			u.constructSparseVector();
	}
	 
	// added by Lin for accessing users in the same cluster.
	HashMap<Integer, ArrayList<Integer>> m_cIndexUIndex = new HashMap<Integer, ArrayList<Integer>>();

	// Group users based on clusters.
	public TreeMap<Integer, _User> groupUsers(int [] clusters){
		TreeMap<Integer, _User> groupedUsers = new TreeMap<Integer, _User>();
		int clusterNo;
		// i accesses all users.
		for(int i=0; i<clusters.length; i++){
			clusterNo = clusters[i];
			if(!groupedUsers.containsKey(clusterNo)){
				groupedUsers.put(clusterNo, new _User(clusterNo, m_classNo));
				m_cIndexUIndex.put(clusterNo, new ArrayList<Integer>());
			}
			groupedUsers.get(clusterNo).mergeReviews(m_users.get(i).getReviews());
			m_cIndexUIndex.get(clusterNo).add(i);
		}
		return groupedUsers;
	}
	
	public HashMap<Integer, ArrayList<Integer>> getCIndexUIndex(){
		return m_cIndexUIndex;
	}
}
