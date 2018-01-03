package Application;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures.Pair;
import structures._RankItem;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;

/***
 * @author lin
 * Content based Collaborative filtering.
 */
public class CollaborativeFiltering {
	// k is the number of neighbors
	protected int m_k, m_time;

	protected int m_featureSize;

	protected ArrayList<_User> m_users;
	protected HashMap<Integer, _User> m_userMap;
	// All the reviews for ranking
	protected ArrayList<_Review> m_totalReviews;
	
	//Given a review, find its index in total reviews
	HashMap<_Review, Integer> m_reviewIndexMap;
	//Given a user ID, access the index of the user.
	HashMap<String, Integer> m_userIDIndex;
	//Given a itemID, find all the users who have purchased this item.
	HashMap<String, ArrayList<Integer>> m_itemIDUserIndex;
	//Given a userID, find his/her neighbors
	HashMap<String, ArrayList<Integer>> m_userIDRdmNeighbors;
	
	protected double m_avgNDCG, m_avgMAP;
	//Assume we have a cache containing all the similarities of all pairs of users.
	protected double[] m_similarity, m_NDCGs, m_MAPs;
	// the group affinity matrix for similarity calculation
	protected double[][] m_userWeights;
	
	protected int[][] m_ranks;
	protected Pair[][] m_realRanks; 

	//The flag is used to decide whether we take all users' average as ranking score or not.
	boolean m_avgFlag = false;
	// The flag used to decide whether we perform weighted score or not in calculating ranking score.
	boolean m_equalWeight = false; 
	// default neighborhood by BoW
	SimType m_sType = SimType.ST_BoW;
	
	// lock when collecting review statistics
	private Object m_userWeightsLock = null;
	private Object m_similarityLock = null;
	private Object m_NDCGMAPLock = null;
	
	public CollaborativeFiltering(ArrayList<_User> users){
		convert2UserMap(users);
		
		m_featureSize = 0;
		m_totalReviews = new ArrayList<_Review>();
		m_similarityLock = new Object();
		m_userWeightsLock = new Object();
		m_NDCGMAPLock = new Object();
		init();
	}
	
	public CollaborativeFiltering(ArrayList<_User> users, int fs) {
		convert2UserMap(users);
		m_featureSize = fs;
		
		m_totalReviews = new ArrayList<_Review>();
		m_similarityLock = new Object();
		m_userWeightsLock = new Object();
		m_NDCGMAPLock = new Object();
		init();
	}
	
	public CollaborativeFiltering(ArrayList<_User> users, int fs, int k){
		convert2UserMap(users);
		m_featureSize = fs;
		m_k = k;
		
		m_totalReviews = new ArrayList<_Review>();
		m_similarityLock = new Object();
		m_userWeightsLock = new Object();
		m_NDCGMAPLock = new Object();
		init();
	}
		
	// The function for calculating all NDCGs and MAPs.
	public void calculateAllNDCGMAP(){
		System.out.println("[Info]Start calculating NDCG and MAP...\n");
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run() {
					_User u;
					try {
						for (int i = 0; i + core <m_users.size(); i += numOfCores) {
							if(i%500==0) System.out.print(".");
							u = m_users.get(i+core);
							if(m_userMap.containsKey(i+core))
								calculateNDCGMAP(u);
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
	}
	
	public void calculateAvgNDCGMAP(){
		double sumNDCG = 0, sumMAP = 0;
		for(int i=0; i < m_users.size(); i++){
			sumNDCG += m_NDCGs[i];
			if(Double.isNaN(m_NDCGs[i]))
				System.out.print("*");
			sumMAP += m_MAPs[i];
			if(Double.isNaN(m_MAPs[i]))
				System.out.print("*");
		}
		m_avgNDCG = sumNDCG/m_userMap.size();
		m_avgMAP = sumMAP/m_userMap.size();
	}
	
	// calculate the nDCG and MAP for each user
	public void calculateNDCGMAP(_User u){
			
		int rdmIndex = 0;
		int reviewSize = u.getReviewSize();
		int userIndex = m_userIDIndex.get(u.getUserID());
		double iDCG = 0, DCG = 0, PatK = 0, AP = 0, count = 0;
			
		_Review review;
		ArrayList<Integer> rdmIndexes = m_userIDRdmNeighbors.get(u.getUserID());
		int totalReviewSize = rdmIndexes.size() + reviewSize;
		
		int[] rank = new int[totalReviewSize];
		Pair[] realRank = new Pair[totalReviewSize];
			
		//Calculate the ideal rank and real rank.
		for(int i=0; i<totalReviewSize; i++){
			if(i < reviewSize){
				review = u.getReviews().get(i);
				rank[i] = review.getYLabel() + 1;
				realRank[i]=new Pair(rank[i], calculateRankScore(u, review));
			} else{
				rdmIndex = rdmIndexes.get(i-reviewSize);
				review = m_totalReviews.get(rdmIndex);
				rank[i] = 0;
				realRank[i] = new Pair(rank[i], calculateRankScore(u, review));
			}
		}
		// sort the array in descending order
		sortPrimitivesDescending(rank);
		// sort the calculated rank based on each pair's value
		Arrays.sort(realRank, new Comparator<Pair>(){
			@Override
			public int compare(Pair p1, Pair p2){
				if(p1.getValue() < p2.getValue())
					return 1;
				else if(p1.getValue() > p2.getValue())
					return -1;
				else 
					return 0;
				
			}
		});
					
		//Calculate DCG and iDCG, nDCG = DCG/iDCG.
		for(int i=0; i<rank.length; i++){
			iDCG += (Math.pow(2, rank[i])-1)/(Math.log(i+2));//log(i+1), since i starts from 0, add 1 more.
			DCG += (Math.pow(2, realRank[i].getLabel())-1)/(Math.log(i+2));
			if(realRank[i].getLabel() >= 1){
				PatK = (count+1)/((double)i+1);
				AP += PatK;
				count++;
			}
		}
		
		// put the calculated nDCG into the array for average calculation
		synchronized(m_NDCGMAPLock){
			m_NDCGs[userIndex] = DCG/iDCG;
			m_MAPs[userIndex] = AP/count;
			
			m_ranks[userIndex] = rank;
			m_realRanks[userIndex] = realRank;
		}
		u.setNDCG(m_NDCGs[userIndex]);	
		u.setMAP(m_MAPs[userIndex]);
	}

	// calculate the ranking score for each review of each user.
	// The ranking score is calculated based on the set of users who have reviewed the item
	public double calculateRankScore(_User u, _Review r){
		int userIndex = m_userIDIndex.get(u.getUserID());
		double rankSum = 0;
		double simSum = 0;
		String itemID = r.getItemID();
			
		//select top k users who have purchased this item.
		ArrayList<Integer> candidates = m_itemIDUserIndex.get(itemID);
		if(m_avgFlag){
			for(int c: candidates){
				if(!m_userMap.containsKey(c)) continue;
				double label = m_userMap.get(c).getItemIDRating().get(itemID)+1;
				rankSum += label;
				simSum++;
			}
			return rankSum/ simSum;
		} else{
			MyPriorityQueue<_RankItem> neighbors;
			if(candidates.size() < m_k)
				neighbors = new MyPriorityQueue<_RankItem>(candidates.size());
			else
				neighbors = new MyPriorityQueue<_RankItem>(m_k);
			//collect k nearest neighbors for each item of the user.
			for(int c: candidates){
				if(c != userIndex)
					neighbors.add(new _RankItem(c, getSimilarity(userIndex, c)));
			}
			//Calculate the value given by the neighbors and similarity;
			for(_RankItem ri: neighbors){
				int label = m_userMap.get(ri.m_index).getItemIDRating().get(itemID)+1;
				rankSum += m_equalWeight ? label:ri.m_value*label;//If equal weight, add label, otherwise, add weighted label.
				simSum += m_equalWeight ? 1: ri.m_value;
			}
		}
		if( simSum == 0) 
			return 0;
		else
			return rankSum/simSum;
	}
	
	protected double calculateSimilarity(double[] ui, double[] uj){
		return Utils.cosine(ui, uj);
	}

	//<Item, <UserIndex>>, inside each user, <item, rating>
	public void constructItemUserIndex(){
			
		ArrayList<Integer> userIndexes;
		m_itemIDUserIndex = new HashMap<String, ArrayList<Integer>>();
		m_reviewIndexMap = new HashMap<_Review, Integer>();
		
		// Traverse all users and set the item-userID map.
		for (int index: m_userMap.keySet()) {
			_User user = m_userMap.get(index);
			for (_Review r : user.getReviews()) {
				String itemID = r.getItemID();
				user.addOneItemIDRatingPair(itemID, r.getYLabel());
				// If the product is in the hashmap.
				if (!m_itemIDUserIndex.containsKey(itemID))
					m_itemIDUserIndex.put(itemID, new ArrayList<Integer>());

				m_itemIDUserIndex.get(itemID).add(index);
			}
		}
			
		System.out.format("[Info]%d users before removal.\n", m_userMap.size());
		ArrayList<String> prodIDs = new ArrayList<String>();
		// Remove the items that are only purchased by one user.
		for (String prodID : m_itemIDUserIndex.keySet()) {
			userIndexes = m_itemIDUserIndex.get(prodID);
			if (userIndexes.size() == 1) {
				int index = userIndexes.get(0);
				m_userMap.get(index).removeOneReview(prodID);
				if(m_userMap.get(index).getReviewSize() == 0)
					m_userMap.remove(index);
				prodIDs.add(prodID);
			}
		}
		// remove zero-reivew users in user map.
		for(int i=0; i<m_users.size(); i++){
			if(m_users.get(i).getReviewSize() == 0)
				m_userMap.remove(i);
				
		}
		System.out.format("[Info]%d/%d products are left after removal.\n", m_itemIDUserIndex.size()-prodIDs.size(), m_itemIDUserIndex.size());
		
		// Remove products with <=1 purchases.
		for(String prodID: prodIDs)
			m_itemIDUserIndex.remove(prodID);
		System.out.format("[Info]%d users are left after removal.\n", m_userMap.size());
	
		// Collect all the reviews of all the users.
		for (int index: m_userMap.keySet()){
			_User user = m_userMap.get(index);
			for(_Review r: user.getReviews()){
				m_totalReviews.add(r);
				m_reviewIndexMap.put(r, index++);
			}
		}
	}
	
	// calculate the similarity between each pair of users
	public void constructNeighborhood() {
		System.out.println("\n[Info]Construct user neighborhood...");
		m_similarity = new double[m_users.size() * (m_users.size()-1)/2];
		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		for(int k=0; k<numberOfCores; ++k){
			threads.add((new Thread() {
				int core, numOfCores;
				@Override
				public void run() {
					double[] ui, uj;
					try {
						for (int i = 0; i + core <m_users.size(); i += numOfCores) {
							if(!m_userMap.containsKey(i+core)) continue;
							ui = m_userWeights[i+core];
							for(int j=0; j<i+core; j++) {
								if (j == i+core || !m_userMap.containsKey(j))
									continue;
								uj = m_userWeights[j];
								double simi = calculateSimilarity(ui, uj);
								synchronized(m_similarityLock){
									m_similarity[getIndex(i+core, j)] = simi;
								}
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

		System.out.format("[Info]Neighborhood graph based on %s constructed for %d users.\n", m_sType, m_userMap.size());
	}	
	
	// For each user, construct candidate items for ranking
	// candidate size = time * review size
	public void constructRandomNeighbors(int t, HashMap<String, ArrayList<Integer>> userIDRdmNeighbors){
		m_time = t;
		_Review review;
		ArrayList<Integer> indexes;
		for(int index: m_userMap.keySet()){
			_User user = m_userMap.get(index);
			indexes = new ArrayList<Integer>();
			for(int i=user.getReviewSize(); i<user.getReviewSize()*m_time; i++){
				int randomIndex = (int) (Math.random() * m_totalReviews.size());
				review = m_totalReviews.get(randomIndex);
				while(user.getReviews().contains(review)){
					randomIndex = (int) (Math.random() * m_totalReviews.size());
					review = m_totalReviews.get(randomIndex);
				}
				indexes.add(randomIndex);
			}
			userIDRdmNeighbors.put(user.getUserID(), indexes);
		}
	}

	// for one item of a user, find the other users who have reviewed this item.
	// collection their other purchased items for ranking.
	public void constructRandomNeighborsAll(HashMap<String, ArrayList<Integer>> userIDRdmNeighbors){
		for(int index: m_userMap.keySet()){
			_User user = m_userMap.get(index);
			ArrayList<Integer> indexes = new ArrayList<Integer>();
			for(int i=0; i<user.getReviewSize(); i++){
				String itemID = user.getReviews().get(i).getItemID();
				// access all the users who have purchased this item
				for(int userIndex: m_itemIDUserIndex.get(itemID)){
					_User nei = m_userMap.get(userIndex);
					// the users' other purchased items will be considered as candidate item for ranking
					for(_Review r: nei.getReviews()){
						if(!r.getItemID().equals(itemID)){
							indexes.add(m_reviewIndexMap.get(r));
						}
					}
				}
			}
			userIDRdmNeighbors.put(user.getUserID(), indexes);
		}
	}
	
	protected void convert2UserMap(ArrayList<_User> users){
		m_userMap = new HashMap<Integer, _User>();
		for(int i=0; i<users.size(); i++){
			m_userMap.put(i, users.get(i));
		}
		m_users = users;
	}

	//Access the index of similarity.
	int getIndex(int i, int j) {
		if (i<j) {//swap
			int t = i;
			i = j;
			j = t;
		} else if(i == j){
			System.out.println("The pair has the same indexes!");
			return 0;
		} 
		return i*(i-1)/2+j;//lower triangle for the square matrix, index starts from 1 in liblinear
	}
	
	public double getSimilarity(int i, int j){
		int index = getIndex(i, j);
		if(index == 47516626)
			System.out.println("bug here.");
		return m_similarity[index];
	}
	
	public double getAvgNDCG(){
		return m_avgNDCG;
	}
	
	public double getAvgMAP(){
		return m_avgMAP;
	}
	
	public double getItemStat(){
		double avg = 0;
		for(String itemID: m_itemIDUserIndex.keySet()){
			avg += m_itemIDUserIndex.get(itemID).size();
		}
		return avg/ m_itemIDUserIndex.size();
	}

	public void init(){
		
		sanityCheck();
		
		m_userIDIndex = new HashMap<String, Integer>();
		for(int i=0; i<m_users.size(); i++){
			m_userIDIndex.put(m_users.get(i).getUserID(), i);
		}
		constructItemUserIndex();
				
		m_NDCGs = new double[m_users.size()];
		m_MAPs = new double[m_users.size()];
		m_ranks = new int[m_users.size()][];
		m_realRanks = new Pair[m_users.size()][];
		
		m_avgNDCG = 0;
		m_avgMAP = 0;
	}

	public void loadWeights(String weightFile, String suffix1, String suffix2){
		loadUserWeights(weightFile, suffix1, suffix2);
		constructNeighborhood();
	}
	
	public void loadUserWeights(String folder, final String suffix1, final String suffix2){
		m_userWeights = new double[m_users.size()][];
		final File dir = new File(folder);
		final File[] files;

		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		if(!dir.exists()){
			System.err.print("[Info]BoW is used as user weights.");
			loadSVMWeights();
		} else{
			files = dir.listFiles();
			for(int k=0; k<numberOfCores; ++k){
				threads.add((new Thread() {
					int core, numOfCores;
					@Override
					public void run() {
						double[] weights;
						String userID;
						int userIndex;
						try {
							for (int i = 0; i + core <dir.listFiles().length; i += numOfCores) {
								File f = files[i+core];
								if(f.isFile() && (f.getName().endsWith(suffix1) || f.getName().endsWith(suffix2))){
									int endIndex = f.getName().lastIndexOf(".");
									userID = f.getName().substring(0, endIndex);
									if(m_userIDIndex.containsKey(userID)){
										userIndex = m_userIDIndex.get(userID);
										weights = loadOneUser(f.getAbsolutePath());
										synchronized(m_userWeightsLock){
											m_userWeights[userIndex] = weights;
										}
									}
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
		}
		System.out.format("[Info]%d users weights are loaded from %s.", m_userWeights.length, folder);
	}
	
	//If not weights provided, use BoW weights.
	public void loadSVMWeights(){
		
		_User u;
		m_userWeights = new double[m_users.size()][];
		
		for(int i=0; i<m_users.size(); i++){
			m_userWeights[i] = new double[m_featureSize];
			u = m_users.get(i);
			for(_SparseFeature fv: u.getBoWProfile())
				m_userWeights[i][fv.getIndex()] = fv.getValue();
		}
	}
	
	public double[] loadOneUser(String fileName){
		double[] weights = new double[m_featureSize];
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
			String line;
			while((line = reader.readLine()) != null){
				String[] ws = line.split(",");
				if(ws.length != m_featureSize)
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
	
	public void sanityCheck(){
		int counter1 = 0;
		int counter2 = 0;
		int counter3 = 0;
		for (int index: m_userMap.keySet()) {
			_User u = m_userMap.get(index);
			if(u.getReviewSize() == 1)
				counter1++;
			if(u.getReviewSize() == 2)
				counter2++;
			else if(u.getReviewSize() == 3)
				counter3++;
		}
		System.out.format("[Info]%d users have 1 reviews, %d users have 2 review, %d users have 3 reviews.\n", counter1, counter2, counter3);
	}
	
	public void sortPrimitivesDescending(int[] rank){
		Arrays.sort(rank);
		// then reverse the array
		for(int i=0; i<rank.length/2; i++){
			int tmp = rank[rank.length-1-i];
			rank[rank.length-i-1] = rank[i];
			rank[i] = tmp;
		}
	}
	
	public void savePerf(String filename){
		PrintWriter writer;
		try{
			writer = new PrintWriter(new File(filename));
			for(int i=0; i<m_NDCGs.length; i++){
				if(!m_userMap.containsKey(i)) continue;
				writer.write(String.format("%s\t%.4f\t%.4f\n", m_users.get(i).getUserID(), m_NDCGs[i], m_MAPs[i]));
			}
			writer.close();
			
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void setUserIDRdmNeighbors(HashMap<String, ArrayList<Integer>> userIDRdmNeighbors){
		m_userIDRdmNeighbors = userIDRdmNeighbors;
	}

	public void setAvgFlag(boolean b){
		m_avgFlag = b;
	}
	public void setEqualWeightFlag(boolean a){
		m_equalWeight = a;
	}
}
