package Application;

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
import java.util.HashMap;

import structures.MyPriorityQueue;
import structures._RankItem;
import structures._Review;
import structures._SparseFeature;
import structures._User;
import utils.Utils;
import Classifier.supervised.modelAdaptation._AdaptStruct.SimType;

/***
 * @author lin
 * Collaborative filtering.
 */
public class CollaborativeFiltering {
	// k is the number of neighbors
	// time is the number of random reviews = m_time*(reviews.size()-1) 
	int m_k, m_featureSize, m_time; 
	
	ArrayList<_User> m_users;
	ArrayList<_Review> m_totalReviews; // All the reviews.

	String[] m_userIDs;//Given a user index, access ID of the user.
	HashMap<String, Integer> m_userIDIndex; //Given a user ID, access the index of the user.
	HashMap<String, ArrayList<Integer>> m_itemIDUserIndex;
	HashMap<String, ArrayList<Integer>> m_userIDRdmNeighbors;
	private Object m_userWeightsLock = null, m_similarityLock = null, m_NDCGMAPLock = null;// lock when collecting review statistics
	
	double m_avgNDCG, m_avgMAP;
	//Assume we have a cache containing all the similarities of all pairs of users.
	double[] m_similarity, m_NDCGs, m_MAPs;
	double[][] m_userWeights;
	// the group affinity matrix for similarity calculation
	double[][] m_B;
	
	int[][] m_ranks;
	Pair[][] m_realRanks; 
	
	// The flag for considering weight or not.
	boolean m_equalWeight; 
	//The flag is used to decide whether we take all users' average as ranking score or not.
	boolean m_avgFlag; 
	// whether we use mixture as similarity measure or not.
	boolean m_mixFlag = false;
	
	SimType m_sType = SimType.ST_BoW;// default neighborhood by BoW

	// Structure: pair for stroing real rand and ideal rank.
	public class Pair {
		double m_label;
		double m_rankValue;
		
		public Pair(){
			m_label = 0;
			m_rankValue = 0;
		}
		
		public Pair(double l, double rv){
			m_label = l;
			m_rankValue = rv;
		}
		
		public double getLabel(){
			return m_label;
		}
		public double getValue(){
			return m_rankValue;
		}
		//rank by predicted score
		public int compareTo (Pair p){
			if (this.m_rankValue > p.m_rankValue)
				return -1;
			else if (this.m_rankValue < p.m_rankValue)
				return 1;
			else 
				return 0;
		}
	}
	
	public CollaborativeFiltering(ArrayList<_User> users, int time){
		m_users = users;
		m_featureSize = 0;
		m_equalWeight = false;
		m_avgFlag = false;
		m_time = time;
		m_totalReviews = new ArrayList<_Review>();
//		m_model = "BoW";
		m_similarityLock = new Object();
		m_userWeightsLock = new Object();
		m_NDCGMAPLock = new Object();
		init();
	}
	
	public CollaborativeFiltering(ArrayList<_User> users, int fs, int k, int time, String model){
		m_users = users;
		m_featureSize = fs;
		
		m_k = k;
		m_time = time;
		m_equalWeight = false;
		m_avgFlag = false;
		m_totalReviews = new ArrayList<_Review>();
//		m_model = model;
		m_similarityLock = new Object();
		m_userWeightsLock = new Object();
		m_NDCGMAPLock = new Object();
		init();
	}
	
	//The ranking score is based on the set of users. For each item, there is a set of neighbors.
	public double calculateRankScore(_User u, _Review r){
		int userIndex = m_userIDIndex.get(u.getUserID());
		double rankSum = 0;
		double simSum = 0;
		String itemID = r.getItemID();
			
		//select top k users who have purchased this item.
		ArrayList<Integer> candidates = m_itemIDUserIndex.get(itemID);
		if(m_avgFlag){
			for(int c: candidates){
				double label = m_users.get(c).getItemIDRating().get(itemID)+1;
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
				int label = m_users.get(ri.m_index).getItemIDRating().get(itemID)+1;
				rankSum += m_equalWeight ? label:ri.m_value*label;//If equal weight, add label, otherwise, add weighted label.
				simSum += m_equalWeight ? 1: ri.m_value;
			}
		}
		if( simSum == 0) 
			return 0;
		else
			return rankSum/simSum;
	}

	public void calculatenDCGMAP(_User u){
			
		int rdmIndex = 0;
		int reviewSize = u.getReviewSize();
		int userIndex = m_userIDIndex.get(u.getUserID());
		double iDCG = 0, DCG = 0, PatK = 0, AP = 0, count = 0;
			
		_Review review;
		int totalReviewSize = reviewSize*m_time;
		int[] rank = new int[totalReviewSize];
		Pair[] realRank = new Pair[totalReviewSize];
		ArrayList<Integer> rdmIndexes = m_userIDRdmNeighbors.get(u.getUserID());
			
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
			
		rank = sortPrimitives(rank);
		realRank = mergeSort(realRank);
			
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
		
		synchronized(m_NDCGMAPLock){
			m_NDCGs[userIndex] = DCG/iDCG;
			m_MAPs[userIndex] = AP/count;
			
			m_ranks[userIndex] = rank;
			m_realRanks[userIndex] = realRank;
		}
		u.setNDCG(m_NDCGs[userIndex]);	
		u.setMAP(m_MAPs[userIndex]);
	}
		

	// The function for calculating all NDCGs and MAPs.
	public void calculatAllNDCGMAP(){
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
							calculatenDCGMAP(u);
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

	public void calcuateSaveAvgNDCGMAP(String fileName) throws FileNotFoundException{
		System.out.println("Start writing NDCG and MAPs.");
		PrintWriter writer = new PrintWriter(new File(fileName));
		double sumNDCG = 0, sumMAP = 0;
		String itemID;
		_User u;
		writer.format("UserIndex\tNoReviews\tNDCG\tMAP\n");
		for(int i=0; i < m_users.size(); i++){
			u = m_users.get(i);
			writer.format("%s\t%d\t%.4f\t%.4f\n", u.getUserID(), u.getReviewSize(), m_NDCGs[i], m_MAPs[i]);
			for(_Review r: u.getReviews())
				writer.format("%d\t", m_itemIDUserIndex.get(r.getItemID()).size());
			writer.write("\n");
			for(int rdmIndex: m_userIDRdmNeighbors.get(u.getUserID())){
				itemID = m_totalReviews.get(rdmIndex).getItemID();
				writer.write(m_itemIDUserIndex.get(itemID).size()+"\t");
			}
			writer.write("\n---------------------------------\n");
			for(int r: m_ranks[i])
				writer.write(r+"\t");
			writer.write("\n");
			for(Pair p: m_realRanks[i])
				writer.write(p.m_label+"\t");
			writer.write("\n==================================\n");
			
			sumNDCG += m_NDCGs[i];
			if(Double.isNaN(m_NDCGs[i]))
				System.out.print("*");
			sumMAP += m_MAPs[i];
			if(Double.isNaN(m_MAPs[i]))
				System.out.print("*");
		}
		m_avgNDCG = sumNDCG/m_users.size();
		m_avgMAP = sumMAP/m_users.size();
		writer.format("avg\t%.4f\t%.4f\n", m_avgNDCG, m_avgMAP);
		writer.close();
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
		m_avgNDCG = sumNDCG/m_users.size();
		m_avgMAP = sumMAP/m_users.size();
	}
	
	// calculate the similarity between two users based on the product of mixture and group affinity.
	protected double calcMixtureSimi(double[] ui, double[] uj){
		if(ui.length != m_B.length)
			System.out.println("Wrong dimension of user mixture!");
		double sim = 0;
		for(int i=0; i<ui.length; i++){
			for(int j=0; j<uj.length; j++){
				sim += ui[i]*uj[j]*m_B[i][j];
			}
		}
		return sim;
	}
	
	public void checkSimi(){
		int[] arr = new int[]{33, 54, 100, 400, 577};
		double sim1, sim2;
		for(int i: arr){
			for(int j: arr){
				if(i == j)
					continue;
				sim1 = m_similarity[getIndex(i,j)];
				sim2 = Euclidean(m_userWeights[i], m_userWeights[j]);
				if(sim1 == sim2)
					System.out.print("=");
				else
					System.out.print("x");
			}
		}
	}
	//<Item, <UserIndex>>, inside each user, <item, rating>
	public void constructItemUserIndex(){
			
		int userIndex;
		String itemID;
		ArrayList<Integer> userIndexes;
		m_itemIDUserIndex = new HashMap<String, ArrayList<Integer>>();
			
		// Traverse all users and set the item-userID map.
		for (_User u : m_users) {
			userIndex = m_userIDIndex.get(u.getUserID());
			for (_Review r : u.getReviews()) {
				itemID = r.getItemID();
				u.addOneItemIDRatingPair(itemID, r.getYLabel());
				// If the product is in the hashmap.
				if (!m_itemIDUserIndex.containsKey(itemID))
					m_itemIDUserIndex.put(itemID, new ArrayList<Integer>());

				m_itemIDUserIndex.get(itemID).add(userIndex);
			}
		}
			
		System.out.format("[Info]%d products in total before removal/", m_itemIDUserIndex.size());
		ArrayList<String> prodIDs = new ArrayList<String>();
		ArrayList<Integer> rmUserIndexes = new ArrayList<Integer>();
		// Remove the items that are only purchased by one user.
		for (String prodID : m_itemIDUserIndex.keySet()) {
			userIndexes = m_itemIDUserIndex.get(prodID);
			if (userIndexes.size() == 1) {
				for (int index : userIndexes){
					m_users.get(index).removeOneReview(prodID);
					if(m_users.get(index).getReviewSize() == 0)
						rmUserIndexes.add(index);
				}
				prodIDs.add(prodID);
			}
		}
		// Remove products with <=1 purchases.
		for(String prodID: prodIDs)
			m_itemIDUserIndex.remove(prodID);
		
		// Remove users with no reviews.
		Collections.sort(rmUserIndexes, Collections.reverseOrder());
		for(int rmUserIndex: rmUserIndexes)
			m_users.remove(rmUserIndex);
		
		// Collect all the reviews of all the users.
		for (_User u : m_users)
			m_totalReviews.addAll(u.getReviews());
		System.out.format("%d are left after removal.\n", m_itemIDUserIndex.size());
	}
	
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
								double simi = m_mixFlag ? calcMixtureSimi(ui, uj):Utils.cosine(ui, uj);
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

		System.out.format("[Info]Neighborhood graph based on %s constructed for %d users.\n", m_sType, m_users.size());
	}	
	
	// For each user, construct neighbors.
	public HashMap<String, ArrayList<Integer>> constructRandomNeighbors(){
		HashMap<String, ArrayList<Integer>> userIDRdmNeighbors = new HashMap<String, ArrayList<Integer>>();
		for(_User u: m_users)
			userIDRdmNeighbors.put(u.getUserID(), getRandomNeighbors(u));
		return userIDRdmNeighbors;
	}
	
	//If the euclidean distance is small, then the pair is similar.
	public double Euclidean(double[] a, double[] b){
		double res = 0;
		if(a.length != b.length)
			return res;
		else{
			for(int i=0; i<a.length; i++){
				res += (a[i]-b[i])*(a[i]-b[i]);
			}
		}
		if(1/Math.sqrt(res) == 0)
			System.out.println("Sim 0 in Euclidean!");
		return 1/Math.sqrt(res);
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
	
	// Get neighbor indexes of the users.
	public ArrayList<Integer> getRandomNeighbors(_User u){
		ArrayList<Integer> indexes = new ArrayList<Integer>();
		_Review review;
		for(int i=u.getReviewSize(); i<u.getReviewSize()*m_time; i++){
			int randomIndex = (int) (Math.random() * m_totalReviews.size());
			review = m_totalReviews.get(randomIndex);
			while(u.getReviews().contains(review)){
				randomIndex = (int) (Math.random() * m_totalReviews.size());
				review = m_totalReviews.get(randomIndex);
			}
			indexes.add(randomIndex);
		}
		return indexes;
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
		String userID;
		m_userIDIndex = new HashMap<String, Integer>();

		for(int i=0; i<m_users.size(); i++){
			userID = m_users.get(i).getUserID();
			m_userIDIndex.put(userID, i);
		}
		
		constructItemUserIndex();
		m_userIDs = new String[m_users.size()];
		for(int i=0; i<m_users.size(); i++){
			m_userIDs[i] = m_users.get(i).getUserID();
		}
		
		m_NDCGs = new double[m_users.size()];
		m_MAPs = new double[m_users.size()];
		m_ranks = new int[m_users.size()][];
		m_realRanks = new Pair[m_users.size()][];
		
		m_avgNDCG = 0;
		m_avgMAP = 0;
	}
	
	// load B_0 and B_1 and calculate the MLE of B
	public void calcMLEB(String fileB0, String fileB1){
		double b = 0;
		// store B_0
		double[][] m_B_0 = loadBFile(fileB0);
		// store B_1 first, later on store MLE_B
		m_B = loadBFile(fileB1);
		if(m_B_0.length != m_B.length){
			System.out.println("[Error]The dimension of B_0 and B_1 does not match!");
			return;
		}
		int kBar = m_B.length;
		m_featureSize = kBar;
		for(int i=0; i<kBar; i++){
			for(int j=i; j<kBar; j++){
				if(m_B[i][j] == 0) continue;
				b = m_B[i][j] / (m_B[i][j] + m_B_0[i][j]);
				m_B[i][j] = b;
				m_B[j][i] = b;
			}
		}
	}

	public double[][] loadBFile(String fileName){
		int kBar = -1;
		double[][] weights = null;
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
			String line;
			int index = 0;
			while((line = reader.readLine()) != null){
				String[] ws = line.split("\t");
				// get the kBar
				if(kBar == -1 && ws.length > 0){
					kBar = ws.length;
					weights = new double[kBar][kBar];
				}
				if(ws.length != kBar)
					System.out.println("[error]The dimension of B does not match kBar!");
				else{
					for(int i=0; i<ws.length; i++){
						weights[index][i] = Double.valueOf(ws[i]);
					}
					index++;
				}
			}
			reader.close();
		} catch(IOException e){
			System.err.format("[Error]Failed to open file %s!!", fileName);
			e.printStackTrace();
		}
		System.out.println("[Info]Finish loading B file " + fileName);
		return weights;
	}
	
	public void loadWeights(String weightFile, String suffix1, String suffix2){
		loadUserWeights(weightFile, suffix1, suffix2);
		constructNeighborhood();
//		checkSimi();
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
	public int[] sortPrimitives(int[] rank){
		int[] res = new int[rank.length];
		Arrays.sort(rank);
		for(int i=0; i<rank.length; i++){
			res[i] = rank[rank.length-1-i];
		}
		return res;
	}
	
	public void savePerf(String filename){
		PrintWriter writer;
		try{
			writer = new PrintWriter(new File(filename));
			for(int i=0; i<m_NDCGs.length; i++){
				writer.write(String.format("%s\t%.4f\t%.4f\n", m_userIDs[i], m_NDCGs[i], m_MAPs[i]));
			}
			writer.close();
			
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void setUserIDRdmNeighbors(HashMap<String, ArrayList<Integer>> userIDRdmNeighbors){
		m_userIDRdmNeighbors = userIDRdmNeighbors;
	}
	
	public void setMixtureFlag(boolean b){
		m_mixFlag = b;
	}

	public void setAvgFlag(boolean b){
		m_avgFlag = b;
	}
	public void setEqualWeightFlag(boolean a){
		m_equalWeight = a;
	}
}
