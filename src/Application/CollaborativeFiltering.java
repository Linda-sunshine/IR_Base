package Application;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import structures.MyPriorityQueue;
import structures.Pair;
import structures._CFUser;
import structures._RankItem;
import structures._Review;
import structures._Review.rType;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;
//import structures._User;

/***
 * @author lin
 * Content based Collaborative filtering.
 */
public class CollaborativeFiltering {
	// k is the number of neighbors
	protected int m_k, m_time, m_featureSize;
	protected int m_validUser = 0;
	protected ArrayList<_CFUser> m_users;
	protected HashMap<String, _CFUser> m_userMap;
	// All the reviews for ranking
	protected ArrayList<_Review> m_trainReviews;
	
	//Given a user ID, access the index of the user.
	protected HashMap<String, Integer> m_userIDIndex;
	
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
	
	// key: item id, value: train user id
	HashMap<String, ArrayList<String>> m_trainMap;
	// key: item id, value: test user id

	HashMap<String, ArrayList<String>> m_testMap;
	
	// lock when collecting review statistics
	private Object m_userWeightsLock = null;
	private Object m_similarityLock = null;
	private Object m_NDCGMAPLock = null;
	
	// constructor for cf with all neighbors
	public CollaborativeFiltering(ArrayList<_CFUser> users, int fs){
		m_userMap = new HashMap<String, _CFUser>();
		for(int i=0; i<users.size(); i++){
			_CFUser user = users.get(i);
			m_userMap.put(user.getUserID(), user);
		}
		
		m_users = users;		
		m_featureSize = fs;
		
		m_trainReviews = new ArrayList<_Review>();
		m_similarityLock = new Object();
		m_userWeightsLock = new Object();
		m_NDCGMAPLock = new Object();
		init();
	}
	// constructor for cf with topK neighbors
	public CollaborativeFiltering(ArrayList<_CFUser> users, int fs, int k){
		m_userMap = new HashMap<String, _CFUser>();
		for(int i=0; i<users.size(); i++){
			_CFUser user = users.get(i);
			m_userMap.put(user.getUserID(), user);
		}
		
		m_users = users;		
		m_featureSize = fs;
		m_k = k;
		
		m_trainReviews = new ArrayList<_Review>();
		m_similarityLock = new Object();
		m_userWeightsLock = new Object();
		m_NDCGMAPLock = new Object();
		init();
	}
	
	// constructor for getting ranking items
	public CollaborativeFiltering(ArrayList<_User> users, int fs, int k, int t){
		convert2UserMap(users);
		m_featureSize = fs;
		m_k = k;
		m_time = t;
		
		m_trainReviews = new ArrayList<_Review>();
		m_similarityLock = new Object();
		m_userWeightsLock = new Object();
		m_NDCGMAPLock = new Object();
		init();
	}
		
	// constructor for getting ranking items with all neighbors
	public CollaborativeFiltering(ArrayList<_User> users){
		convert2UserMap(users);
		
		m_trainReviews = new ArrayList<_Review>();
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
					_CFUser u;
					try {
						for (int i = 0; i + core <m_users.size(); i += numOfCores) {
							if(i%500==0) System.out.print(".");
							u = m_users.get(i+core);
							if(u.isValid())
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
		m_avgNDCG = sumNDCG/m_validUser;
		m_avgMAP = sumMAP/m_validUser;
	}
	
	// calculate the nDCG and MAP for each user
	public void calculateNDCGMAP(_CFUser u){
		if(u.getUserID().equals("8l3psjdJO7twdMQGB4MTmw"))
			System.out.println("debug!");
		int userIndex = m_userIDIndex.get(u.getUserID());
		double iDCG = 0, DCG = 0, PatK = 0, AP = 0, count = 0;
			
		String[] items = u.getRankingItems();
		int[] rank = new int[items.length];
		Pair[] realRank = new Pair[items.length];
			
		//Calculate the ideal rank and real rank.
		for(int i=0; i<items.length; i++){
			String item = items[i];
			if(u.containsRvw(item)){
				rank[i] = u.getItemRating(item) + 1;
				realRank[i]=new Pair(rank[i], calculateRankScore(u, item));
			} else{
				rank[i] = 0;
				realRank[i] = new Pair(rank[i], calculateRankScore(u, item));
			}
		}
	
		// sort the array in descending order
		sortPrimitivesDescending(rank);
		// sort the calculated rank based on each pair's value
		realRank = mergeSort(realRank);
		if(u.getUserID().equals("8l3psjdJO7twdMQGB4MTmw")){
			for(int r: rank)
				System.out.println(r);
			for(Pair p: realRank){
				System.out.println(p.getLabel()+","+p.getValue());
			}
		}
//		Arrays.sort(realRank, new Comparator<Pair>(){
//			@Override
//			public int compare(Pair p1, Pair p2){
//				if(p1.getValue() < p2.getValue())
//					return 1;
//				else if(p1.getValue() > p2.getValue())
//					return -1;
//				else 
//					return 0;
//			}
//		});
					
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
	}

	// calculate the ranking score for each review of each user.
	// The ranking score is calculated based on the set of users who have reviewed the item
	public double calculateRankScore(_CFUser u, String item){
		int userIndex = m_userIDIndex.get(u.getUserID());
		double rankSum = 0;
		double simSum = 0;
			
		//select top k users who have purchased this item.
		ArrayList<String> neighbors = m_trainMap.get(item);
		if(m_avgFlag){
			for(String nei: neighbors){
				if(!nei.equals(u.getUserID())){
					int index = m_userIDIndex.get(nei);
					double label = m_users.get(index).getItemRating(item)+1;
					rankSum += label;
					simSum++;
				}
			}
			if(simSum == 0){
				System.err.println("bug in candidate!");
				return 0;
			} else
			return rankSum/simSum;
		} else{
			MyPriorityQueue<_RankItem> topKNeighbors;
			if(neighbors.size() < m_k)
				topKNeighbors = new MyPriorityQueue<_RankItem>(neighbors.size());
			else
				topKNeighbors = new MyPriorityQueue<_RankItem>(m_k);
			//collect k nearest neighbors for each item of the user.
			for(String nei: neighbors){
				if(!nei.equals(u.getUserID())){
					int neiIndex = m_userIDIndex.get(nei);
					topKNeighbors.add(new _RankItem(neiIndex, getSimilarity(userIndex, neiIndex)));
				} 
			}
			//Calculate the value given by the neighbors and similarity;
			for(_RankItem ri: topKNeighbors){
				int label = m_users.get(ri.m_index).getItemRating(item);
				if(label == -1)
					System.out.println("[error]Wrong neighbor!");
				else{
					label++;
					rankSum += m_equalWeight ? label:ri.m_value*label;//If equal weight, add label, otherwise, add weighted label.
					simSum += m_equalWeight ? 1: ri.m_value;
				}
			}
		}
		if(simSum == 0){
			System.err.println("bug in candidate!");
			return 0;
		} else
			return rankSum/simSum;
	}
	
	protected double calculateSimilarity(double[] ui, double[] uj){
		return Utils.cosine(ui, uj);
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
							ui = m_userWeights[i+core];
							for(int j=0; j<i+core; j++) {
								if (j == i+core)
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
	public void constructRankingNeighbors(){		
		double sum = 0;
		double avgRvwSize = 0, rvwSize = 0;
		for(_CFUser user: m_users){
			rvwSize = 0;
			Set<String> items = new HashSet<String>();
			for(int i=0; i<user.getTestReviewSize()*m_time; i++){
				if(i< user.getTestReviewSize()){
					_Review r = user.getTestReviews().get(i);
					if(!m_trainMap.containsKey(r.getItemID()))
						continue;
					items.add(r.getItemID());
					rvwSize++;
				} else if(items.size() > 0){
					int randomIndex = (int) (Math.random() * m_trainReviews.size());
					_Review review = m_trainReviews.get(randomIndex);
					while(items.contains(review.getItemID())){
						randomIndex = (int) (Math.random() * m_trainReviews.size());
						review = m_trainReviews.get(randomIndex);
					}
					items.add(review.getItemID());
				}
			}
			user.setRankingItems(items);
			sum += items.size();
			if(items.size()!= 0){
				m_validUser++;
				avgRvwSize += rvwSize;
			}
		}
		System.out.format("[Stat]Valid user: %s, avg candidate item: %.2f, avg rvw size: %.2f.\n", m_validUser, sum/m_validUser, avgRvwSize/m_validUser);
	}
	
//	// For each user, construct candidate items for ranking
//	// candidate size = time * review size
//	public void constructRankingNeighbors(){
//		double sum = 0;
//		double avgRvwSize = 0, rvwSize = 0;
//		for(_CFUser user: m_users){
//			rvwSize = 0;
//			Set<String> items = new HashSet<String>();
//			for(int i=0; i<user.getTrainReviewSize()*m_time; i++){
//				if(i< user.getTrainReviewSize()){
//					_Review r = user.getTrainReviews().get(i);
//					if(!m_trainMap.containsKey(r.getItemID()))
//						continue;
//					items.add(r.getItemID());
//					rvwSize++;
//				} else if(items.size() > 0){
//					int randomIndex = (int) (Math.random() * m_trainReviews.size());
//					_Review review = m_trainReviews.get(randomIndex);
//					while(items.contains(review.getItemID())){
//						randomIndex = (int) (Math.random() * m_trainReviews.size());
//						review = m_trainReviews.get(randomIndex);
//					}
//					items.add(review.getItemID());
//				}
//			}
//			user.setRankingItems(items);
//			sum += items.size();
//			if(items.size()!= 0){
//				m_validUser++;
//				avgRvwSize += rvwSize;
//			}
//		}
//		System.out.format("[Stat]Valid user: %s, avg candidate item: %.2f, avg rvw size: %.2f.\n", m_validUser, sum/m_validUser, avgRvwSize/m_validUser);
//	}
	// we want to get the basic statistics of user-item statistic in training/testing
	//<Item, <UserIndex>>, inside each user, <item, rating>
	public void constructItemUserIndex(){
		m_trainMap = new HashMap<String, ArrayList<String>>();
		m_testMap = new HashMap<String, ArrayList<String>>();
		for(_CFUser u: m_users){
			for(_Review r: u.getReviews()){
				// if it is adaptation review
				if(r.getType() == rType.ADAPTATION){
					String itemID = r.getItemID();
					if(!m_trainMap.containsKey(itemID)){
						m_trainMap.put(itemID, new ArrayList<String>());
					}
					m_trainMap.get(itemID).add(u.getUserID());
				// if it is testing review
				} else{
					String itemID = r.getItemID();
					if(!m_testMap.containsKey(itemID)){
						m_testMap.put(itemID, new ArrayList<String>());
					}
					m_testMap.get(itemID).add(u.getUserID());
				}
				u.addOneItemIDRatingPair(r.getItemID(), r.getYLabel());
			}
		}
		int trainMiss = 0, testMiss = 0;
		for(String itemID: m_trainMap.keySet()){
			if(!m_testMap.containsKey(itemID))
				testMiss++;
		}
		for(String itemID: m_testMap.keySet()){
			if(!m_trainMap.containsKey(itemID))
				trainMiss++;
		}
		for(_CFUser u: m_users){
			for(_Review r: u.getTrainReviews()){
				m_trainReviews.add(r);
			}
		}
		System.out.format("There are %d items in training set.\n", m_trainMap.size());
		System.out.format("There are %d items in testing set.\n", m_testMap.size());
		System.out.format("There are %d items in training while not in testing.\n", testMiss);
		System.out.format("There are %d items in testing while not in training.\n", trainMiss);

	}

	// pass the users from analyzer into cf for later use
	protected void convert2UserMap(ArrayList<_User> users){
		m_userMap = new HashMap<String, _CFUser>();
		m_users = new ArrayList<_CFUser>();
		for(int i=0; i<users.size(); i++){
			_User user = users.get(i);
			_CFUser cfUser = new _CFUser(user);
			m_userMap.put(user.getUserID(), cfUser);
			m_users.add(cfUser);
		}
	}
	
	public ArrayList<_CFUser> getUsers(){
		return m_users;
	}

	public int getValidUserSize(){
		return m_validUser;
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
//		if(index == 47516626)
//			System.out.println("bug here.");
		return m_similarity[index];
	}
	
	public double getAvgNDCG(){
		return m_avgNDCG;
	}
	
	public double getAvgMAP(){
		return m_avgMAP;
	}

	public void init(){
				
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
			System.err.format("[error] %s does not exist! BoW is used as user weights.\n", dir);
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
									} else{
										System.err.format("[error]Bug in loading weights.");
										System.err.println("[error]"+f.getName());
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
		
		_CFUser u;
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
	
	public Pair[] mergeSort(Pair[] rank){
		ArrayList<Pair[]> collection = new ArrayList<Pair[]>();
		for(int i=0; i<rank.length; i=i+2){
			//If the list has odd members.
			if((i+1)>(rank.length-1)){
				Pair[] tmp = new Pair[]{rank[i]};
				collection.add(tmp);
			} else{
				Pair v1 = rank[i], v2 = rank[i+1];
				if(v1.getValue() < v2.getValue()){
					Pair[] tmp = new Pair[]{v2, v1};
					collection.add(tmp);
				} else{
					Pair[] tmp = new Pair[]{v1, v2};
					collection.add(tmp);
				}
			}
		}
		while(collection.size()>1){
			ArrayList<Pair[]> current = new ArrayList<Pair[]>();
			for(int i=0; i<collection.size();i+=2){
				if((i+1) <= collection.size()-1){
					Pair[] merge = merge(collection.get(i), collection.get(i+1));
					current.add(merge);
				} else
					current.add(collection.get(i));
			}
			collection.clear();
			collection.addAll(current);
		}
		return collection.get(0);
	}
	
	public Pair[] merge(Pair[] a, Pair[] b){
		Pair[] res = new Pair[a.length + b.length];
		int pointer1 = 0, pointer2 = 0, count = 0;
		while(pointer1 < a.length && pointer2 < b.length){
			if(a[pointer1].getValue() < b[pointer2].getValue()){
				res[count++] = b[pointer2++];
			} else{
				res[count++] = a[pointer1++];
			}
		}
		while(pointer1 < a.length)
			res[count++] = a[pointer1++];
			
		while(pointer2 < b.length)
			res[count++] = b[pointer2++];
		return res;
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
				writer.write(String.format("%s\t%d\t%d\t%.4f\t%.4f\n", m_users.get(i).getUserID(), m_users.get(i).getTestReviewSize(), m_users.get(i).getRankingItemSize(), m_NDCGs[i], m_MAPs[i]));
			}
			writer.close();
			
		} catch(IOException e){
			e.printStackTrace();
		}
	}

	public void setAvgFlag(boolean b){
		m_avgFlag = b;
	}
	public void setValidUserSize(int sz){
		m_validUser = sz;
	}
	public void setEqualWeightFlag(boolean a){
		m_equalWeight = a;
	}
}
