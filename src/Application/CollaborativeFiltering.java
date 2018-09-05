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
import java.util.HashSet;
import java.util.Set;

import structures.MyPriorityQueue;
import structures.Pair;
import structures._Item;
import structures._RankItem;
import structures._Review;
import structures._Doc.rType;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;

/***
 * @author lin
 * Content based Collaborative Filtering.
 */
public class CollaborativeFiltering {
	// k is the number of neighbors
	protected int m_k, m_featureSize;
	protected int m_validUser = 0;
	protected ArrayList<_User> m_users;
	protected ArrayList<_Item> m_items;
	protected HashMap<Integer, _User> m_userMap;
	// All the reviews for ranking
	protected ArrayList<_Review> m_trainReviews;
	
	//Given a user ID, access the index of the user.
	protected HashMap<String, Integer> m_userIDIndex;
	protected HashMap<String, Integer> m_itemIDIndex;

	protected double m_avgNDCG, m_avgMAP;
	//Assume we have a cache containing all the similarities of all pairs of users.
	protected double[] m_similarity, m_NDCGs, m_MAPs;
	// the group affinity matrix for similarity calculation
	protected double[][] m_userWeights;

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
	// key: item id, value: corresponding item
	HashMap<String, _Item> m_itemMap;
	
	// lock when collecting review statistics
	private Object m_userWeightsLock = null;
	private Object m_similarityLock = null;
	private Object m_NDCGMAPLock = null;
	
	// constructor for cf with all neighbors
	public CollaborativeFiltering(ArrayList<_User> users, int fs){
		m_userMap = new HashMap<Integer, _User>();
		for(int i=0; i<users.size(); i++){
			m_userMap.put(i, users.get(i));
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
	public CollaborativeFiltering(ArrayList<_User> users, int fs, int k){
		m_userMap = new HashMap<Integer, _User>();
		for(int i=0; i<users.size(); i++){
			m_userMap.put(i, users.get(i));
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
							if(m_userMap.containsKey(i+core)){
								calculateNDCGMAP(u);	
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
	
	// for each item, calculate the popularity of the item
	public void calculatePopularity(){
		double pop = 0;
		for(String item: m_trainMap.keySet()){
			pop += m_trainMap.get(item).size();
		}
		System.out.println("Avg pop is : " + pop/m_trainMap.size());
	}
	
	public void calculateAvgNDCGMAP(){
		double sumNDCG = 0, sumMAP = 0;
		int valid = 0;
		for(int i=0; i < m_users.size(); i++){
			if(Double.isNaN(m_NDCGs[i]) || Double.isNaN(m_MAPs[i]) || m_NDCGs[i] == -1 || m_MAPs[i] == -1)
				continue;
			else{
				sumNDCG += m_NDCGs[i];
				sumMAP += m_MAPs[i];
				valid++;
			}
		}
		m_avgNDCG = sumNDCG/valid;
		m_avgMAP = sumMAP/valid;
		System.out.format("\n[Info]Pre-calculated %d valid users, real %d valid user.", m_validUser, valid);
	}
	
	// calculate the nDCG and MAP for each user
	public void calculateNDCGMAP(_User u){
		int userIndex = m_userIDIndex.get(u.getUserID());
		double iDCG = 0, DCG = 0, PatK = 0, AP = 0, count = 0;
			
		if(u.getRankingItems() == null)
			return;
		
		String[] items = u.getRankingItems();
		int[] rank = new int[items.length];
		Pair[] realRank = new Pair[items.length];
			
		//Calculate the ideal rank and real rank.
		for(int i=0; i<items.length; i++){
			String item = items[i];
			if(u.containsTestRvw(item)){
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
		Arrays.sort(realRank, new Comparator<Pair>(){
			@Override
			public int compare(Pair p1, Pair p2){
				if(p1.getValue() < p2.getValue())
					return 1;
				else if(p1.getValue() > p2.getValue())
					return -1;
				else{
//					return (int) (-p1.getLabel() + p2.getLabel()); //potential problematic
					return 0;
				}
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

		}
	}
	
	// calculate the ranking score for each review of each user.
	// The ranking score is calculated based on the set of users who have reviewed the item
	public double calculateRankScore(_User u, String item){
		int userIndex = m_userIDIndex.get(u.getUserID());
		double rankSum = 0;
		double simSum = 0;
			
		if(!m_trainMap.containsKey(item)){
			return 0;
		}
		//select top k users who have purchased this item.
		ArrayList<String> neighbors = m_trainMap.get(item);
		if(m_avgFlag){
			for(String nei: neighbors){
				int neiIndex = m_userIDIndex.get(nei);
				if(neiIndex == userIndex) continue;
				double label = m_users.get(neiIndex).getItemRating(item)+1;
				rankSum += label;
				simSum++;
			}
			if(simSum == 0){
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
				int neiIndex = m_userIDIndex.get(nei);
				if(neiIndex == userIndex) continue;
				topKNeighbors.add(new _RankItem(neiIndex, getSimilarity(userIndex, neiIndex)));
			}
			//Calculate the value given by the neighbors and similarity;
			for(_RankItem ri: topKNeighbors){
				int label = m_users.get(ri.m_index).getItemRating(item)+1;
				rankSum += m_equalWeight ? label:ri.m_value*label;//If equal weight, add label, otherwise, add weighted label.
				simSum += m_equalWeight ? 1: ri.m_value;
			}
		}
		if(simSum == 0){
			return 0;
		} else
			return rankSum/simSum;
	}
	
	protected double calculateSimilarity(double[] ui, double[] uj){
//		return Utils.cosine(ui, uj);
		return Utils.euclideanDistance(ui, uj);
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
	// 1st param: the neighbor selection method
	// 2nd param: the threshold, the number of times or the popularity
	public void constructRankingNeighbors(String selection, int threshold){
		
		// randomly select ranking neighbors
		// and the candidate size = (time -1) * review size
		if(selection.equals("times")){
			constructRankingNeighborsTimes(threshold);
		// select all the ranking neighbors/item with certain popularity
		} else if(selection.equals("all")){
			constructRankingNeighborsAll(threshold);
		} else{
			System.out.println("[error] The neighbor selection method is not developed!");
		}
	}
	
	// randomly select ranking neighbors and the candidate size = (time -1) * review size
	public void constructRankingNeighborsTimes(int time){
		
		double sum = 0;
		double avgRvwSize = 0, rvwSize = 0;
		Set<String> items = new HashSet<String>();
		Set<String> rankItems = new HashSet<String>();
		for(_User user: m_users){
			rvwSize = 0;
			items.clear();
			rankItems.clear();
			for(_Review r: user.getReviews()){
				items.add(r.getItemID());
			}
			
			for(_Review r: user.getTestReviews()){
				if(rankItems.contains(r.getItemID()))
					System.err.println("bug!!");
				rankItems.add(r.getItemID());
			}
			
			for(int i=user.getReviewSize(); i<user.getReviewSize()*time; i++){
				int randomIndex = (int) (Math.random() * m_trainReviews.size());
				_Review review = m_trainReviews.get(randomIndex);
				while(items.contains(review.getItemID())){
					randomIndex = (int) (Math.random() * m_trainReviews.size());
					review = m_trainReviews.get(randomIndex);
				}
				if(items.contains(review.getItemID()))
					System.err.println("bug!!");
				if(rankItems.contains(review.getItemID()))
					System.err.println("bug!!");

				items.add(review.getItemID());
				rankItems.add(review.getItemID());
				rvwSize++;
			}
			user.setRankingItems(rankItems);
			HashMap<String, Integer> map = new HashMap<String, Integer>();
			for(String item: rankItems){
				if(map.containsKey(item)){
					int val = map.get(item)+1;
					map.put(item, val);
				} else {
					map.put(item, 1);
				}
			}
			for(String item: map.keySet()){
				if(map.get(item) > 1)
					System.out.format("(%s, %d)\t", item, map.get(item));
			}
			sum += rankItems.size();
			if(rankItems.size()!= 0){
				m_validUser++;
				avgRvwSize += rvwSize;
			}
		}
		System.out.format("[Stat]Valid user: %s, avg candidate item: %.2f, avg rvw size: %.2f.\n", m_validUser, sum/m_validUser, avgRvwSize/m_validUser);
	}
	
	// select all the ranking neighbors/item with certain popularity
	public void constructRankingNeighborsAll(int pop){
		double sum = 0;
		double avgRvwSize = 0, rvwSize = 0;

		for(_User user: m_users){
			rvwSize = 0;
			ArrayList<_Review> testReviews = user.getTestReviews();
			Set<String> items = new HashSet<String>();
			
			// for each test review, construct the ranking array
			for(_Review tr: testReviews){
				
				String itemID = tr.getItemID();
				// if this item is not in the training map, ignore it.
				if(!m_trainMap.containsKey(itemID))
					continue;
				
				// add the user's own reviewed item
				items.add(tr.getItemID());
				rvwSize++;
				
				// and access all the users who have rated this item in training
				ArrayList<String> trainNeis = m_trainMap.get(itemID);
				for(String nid: trainNeis){
					if(nid.equals(user.getUserID())){
//						System.out.println("The user has rated the same item twice!");
						continue;
					}
					int nIdx = m_userIDIndex.get(nid);
					_User nei = m_userMap.get(nIdx);
					for(_Review r: nei.getTrainReviews()){
						if(m_trainMap.get(r.getItemID()).size() < pop)
							continue;
						items.add(r.getItemID());
					}
				}
			}
			user.setRankingItems(items);
			sum += items.size();
			if(items.size()>=1){
				m_validUser++;
				avgRvwSize += rvwSize;
			}
		}
		System.out.format("[Stat]Pop: %d, Valid user: %s, avg candidate item: %.2f, avg rvw size: %.2f.\n", pop, m_validUser, sum/m_validUser, avgRvwSize/m_validUser);
	}
	// we want to get the basic statistics of user-item statistic in training/testing
	//<Item, <UserIndex>>, inside each user, <item, rating>
	public void constructItemUserIndex(){
		m_trainMap = new HashMap<String, ArrayList<String>>();
		m_testMap = new HashMap<String, ArrayList<String>>();
		m_itemMap = new HashMap<String, _Item>();
		for(_User u: m_users){
			for(_Review r: u.getReviews()){
				// if it is adaptation review
				if(r.getType() == rType.ADAPTATION){
					String itemID = r.getItemID();
					if(!m_trainMap.containsKey(itemID)){
						m_trainMap.put(itemID, new ArrayList<String>());
						m_itemMap.put(itemID, new _Item(itemID));
					}
					m_trainMap.get(itemID).add(u.getUserID());
					m_itemMap.get(itemID).addOneReview(r);
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
		for(_User u: m_users){
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
		m_userMap = new HashMap<Integer, _User>();
		for(int i=0; i<users.size(); i++){
			m_userMap.put(i, users.get(i));
		}
		m_users = users;	
	}
	
	public void constructItems(String model){
		for(String itemID: m_itemMap.keySet()){
			_Item item = m_itemMap.get(itemID);
			item.buildProfile(model);
			item.normalizeProfile();
		}
	}
	public ArrayList<_User> getUsers(){
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
//			System.out.println("The pair has the same indexes!");
			return 0;
		} 
		return i*(i-1)/2+j;//lower triangle for the square matrix, index starts from 1 in liblinear
	}
	
	public double getSimilarity(int i, int j){
		int index = getIndex(i, j);
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
		
		Arrays.fill(m_NDCGs, -1);
		Arrays.fill(m_MAPs, -1);
		
		m_avgNDCG = 0;
		m_avgMAP = 0;
	}

	public void constructRankingCandidates(){
        for(_User u: m_users){
            String[] rankingItems = u.getRankingItems();
            if(rankingItems == null)
                continue;

            for(String item: rankingItems){
                String uid = u.getUserID();
                int userIndex = m_userIDIndex.get(uid);
                if(!m_userMap.containsKey(userIndex))
                    System.err.println("The user does not exist!");
                m_userMap.get(userIndex).addOneCandidate(item);
            }
            u.setRankingItems(m_trainMap);
        }
    }
	
	// load pre-selected ranking candidates
	public void loadRankingCandidates(String filename){
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			// skip the first line
			reader.readLine();
			while((line = reader.readLine()) != null){
				String[] ws = line.split(",");
				String uid = ws[0];
				String item = ws[1];
				int userIndex = m_userIDIndex.get(uid);
				if(!m_userMap.containsKey(userIndex))
					System.err.println("The user does not exist!");
				m_userMap.get(userIndex).addOneCandidate(item);
			}
			reader.close();
			for(_User u: m_users){
				u.setRankingItems(m_trainMap);
			}
			System.out.format("Finish loading ranking candidates for %d users.", m_users.size());
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", filename);
			e.printStackTrace();
		}
	}

	// load user weights and construct the neighborhood
	public void loadWeights(String weightFile, String model, String suffix1, String suffix2){
		loadUserWeights(weightFile, model, suffix1, suffix2);
		constructNeighborhood();
		printSimilarity(model);
	}
	
	public void printSimilarity(String model){
		try{
			PrintWriter writer = new PrintWriter(new File(model+"_sim.txt"));
			for(int i=0; i<m_similarity.length; i++){
				writer.format("%.3f\t", m_similarity[i]);
			}
			writer.close();
		} catch(IOException e){
			e.printStackTrace();
		}
		
	}
	public void loadUserWeights(String folder, String model, final String suffix1, final String suffix2){
		m_userWeights = new double[m_users.size()][];
		final File dir = new File(folder);
		final File[] files;

		int numberOfCores = Runtime.getRuntime().availableProcessors();
		ArrayList<Thread> threads = new ArrayList<Thread>();
		
		// if the directory does not exist, then we load the bow weight of each user
		if(!dir.exists()){
			System.err.format("[error] %s does not exist! BoW is used as user weights.\n", dir);
			loadBoWWeights(model);
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
	
	//If not weights provided, use BoW weights, we can choose bow based on LR features or LM features
	public void loadBoWWeights(String model){
		System.out.format("[Info]The BoW is based on %s features.\n", model);
		_User u;
		m_userWeights = new double[m_users.size()][];
		
		for(int i=0; i<m_users.size(); i++){
			m_userWeights[i] = new double[m_featureSize];
			u = m_users.get(i);
			u.buildProfile(model);
			u.normalizeProfile();
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

	public void setFeatureSize(int fs){
		m_featureSize = fs;
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
	
	// save the user-item pair to graphlab for model training.
	public void saveUserItemPairs(String dir){
		int trainUser = 0, testUser = 0, trainPair = 0, testPair = 0;
		try{
			PrintWriter trainWriter = new PrintWriter(new File(dir+"train.csv"));
			PrintWriter testWriter = new PrintWriter(new File(dir+"test.csv"));
			trainWriter.write("user_id,item_id,rating\n");
			testWriter.write("user_id,item_id,rating\n");
			for(_User u: m_users){
				trainUser++;
				// print out the training pairs
				for(_Review r: u.getTrainReviews()){
					trainPair++;
					trainWriter.write(String.format("%s,%s,%d\n", u.getUserID(), r.getItemID(), u.getItemRating(r.getItemID())+1));
				}
				String[] rankingItems = u.getRankingItems();
				if(rankingItems == null)
					continue;
				testUser++;
				for(String item: rankingItems){
					testPair++;
					testWriter.write(String.format("%s,%s,%d\n", u.getUserID(), item, u.getItemRating(item)+1));
				}
			}
			trainWriter.close();
			testWriter.close();
			System.out.format("[Info]Finish writing (%d,%d) training users/pairs, (%d,%d) testing users/pairs.\n", trainUser, trainPair, testUser, testPair);
		} catch(IOException e){
			e.printStackTrace();
		}
		
	}
	
	// incorporate text information for matrix factorization
	public void saveUsersWithText(String dir, int topk){
		int trainUser = 0, testUser = 0, trainPair = 0, testPair = 0;
		String zeroStr = "";
		for(int i=0; i<topk; i++){
			if(i == topk -1)
				zeroStr += "0";
			else 
				zeroStr += "0,";
		}
		try{
			PrintWriter trainWriter = new PrintWriter(new File(dir+"train.csv"));
			PrintWriter testWriter = new PrintWriter(new File(dir+"test.csv"));
			// construct the string for annotation
			String note = "user_id,item_id,rating,";
			for(int i=0; i<topk-1; i++){
				note += String.format("f_%d,", i);
			}
			note += String.format("f_%d\n", topk-1);
			
			trainWriter.write(note);
			testWriter.write(note);
			for(_User u: m_users){
				trainUser++;
				// print out the training pairs
				for(_Review r: u.getTrainReviews()){
					trainPair++;
					trainWriter.write(String.format("%s,%s,%d,", u.getUserID(), r.getItemID(), u.getItemRating(r.getItemID())+1));
					double[] vct = normalize(r.getLMSparse(), topk);
					// print the lm feature value
					for(int i=0; i<vct.length; i++){
						if(i != vct.length-1)
							trainWriter.write(vct[i]+",");
						else {
							trainWriter.write(vct[i]+"\n");
						}
					}
				}
				String[] rankingItems = u.getRankingItems();
				if(rankingItems == null)
					continue;
				testUser++;
				for(String item: rankingItems){
					testPair++;
					// if it is a relevant item
					if(u.containsTestRvw(item)){
						_Review r = u.getTestReview(item);
						testWriter.write(String.format("%s,%s,%d,", u.getUserID(), item, u.getItemRating(item)+1));
						double[] vct = normalize(r.getLMSparse(), topk);
						// print the lm feature value
						for(int i=0; i<vct.length; i++){
							if(i != vct.length-1)
								testWriter.write(vct[i]+",");
							else {
								testWriter.write(vct[i]+"\n");
							}
						}
					// if it is an irrelevant item
					} else{
						testWriter.write(String.format("%s,%s,%d,%s\n", u.getUserID(), item, u.getItemRating(item)+1, zeroStr));
					}
					
				}
			}
			trainWriter.close();
			testWriter.close();
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	// normalize the lm feature vector
	public double[] normalize(_SparseFeature[] fvs, int topk){
		double sum = 0;
		double[] vct = new double[topk];
		for(_SparseFeature fv: fvs){
			vct[fv.getIndex()] = fv.getValue();
			sum += fv.getValue();
		}
		for(_SparseFeature fv: fvs){
			vct[fv.getIndex()] = fv.getValue()/sum;
		}
		return vct;
	}
}
