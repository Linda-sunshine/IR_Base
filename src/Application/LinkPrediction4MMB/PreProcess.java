package Application.LinkPrediction4MMB;

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

public class PreProcess {
	class _LinkFriend{
		String m_uid;
		String[] m_trainFriends;
		String[] m_testFriends;
		String[] m_testNonFriends;
		
		public _LinkFriend(String uid, String[] frds){
			m_trainFriends = Arrays.copyOf(frds, frds.length);
		}
		
		public void setTestFriends(String[] frds){
			m_testFriends = Arrays.copyOf(frds, frds.length);
		}
		
		public void setTestNonFriends(String[] nonfrds){
			m_testNonFriends = Arrays.copyOf(nonfrds, nonfrds.length);
		}
		
		public String[] getTrainFriends(){
			return m_trainFriends;
		}
		public String[] getTestFriends(){
			return m_testFriends;
		}
		public String[] getTestNonFriends(){
			return m_testNonFriends;
		}
		
		public int getTestNonFrinedSize(){
			if(m_testNonFriends == null)
				return 0;
			else
				return m_testNonFriends.length;
		}
		public boolean hasTestFriends(){
			return m_testFriends != null;
		}
		
		public boolean hasTestNonFriends(){
			return m_testNonFriends != null;
		}
		
	}
	ArrayList<String> m_userIDs, m_itemIDs;
	HashMap<String, Integer> m_userMap;
	HashMap<String, Integer> m_itemMap;
	int m_trainPairSize = 0, m_testPairSize = 0;

	
	public PreProcess(){
		m_userMap = new HashMap<String, Integer>();
		m_itemMap = new HashMap<String, Integer>();
		m_userIDs = new ArrayList<String>();
		m_itemIDs = new ArrayList<String>();
	}
	
	// load the user-item data and get the size of users and items
	public void buildUserItemMap(String filename){
		m_trainPairSize = 0;
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			// skip the first line
			line = reader.readLine();
			while ((line = reader.readLine()) != null) {
				String[] strs = line.split(",");
				String userID = strs[0];
				String itemID = strs[1];
				if(!m_userMap.containsKey(userID)){
					m_userMap.put(userID, m_userMap.size());
					m_userIDs.add(userID);
				}
				if(!m_itemMap.containsKey(itemID)){
					m_itemMap.put(itemID, 1);
					m_itemIDs.add(itemID);
				} else{
					int val = m_itemMap.get(itemID);
					m_itemMap.put(itemID, val+1);
				}
				m_trainPairSize++;
			}
			reader.close();
			System.out.format("(%d, %d) users/items are loaded from %s...\n", m_userMap.size(), m_itemIDs.size(), filename);
			double sum = 0;
			for(String item: m_itemMap.keySet()){
				sum += m_itemMap.get(item);
			}
			sum /= m_itemMap.size();
			System.out.println("avg pop: " + sum);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	// load training friends
	HashMap<String, _LinkFriend> m_userFriendMap = new HashMap<String, _LinkFriend>();
	public void loadTrainFriends(String filename){
		m_trainPairSize = 0;
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			// skip the first line
			while ((line = reader.readLine()) != null) {
				String[] strs = line.split("\t");
				String userID = strs[0];
				if(!m_userFriendMap.containsKey(userID)){
					m_userFriendMap.put(userID, new _LinkFriend(userID, Arrays.copyOfRange(strs, 1, strs.length)));
				} else{
					System.out.println("[error]The user shows twice!");
				}
				m_trainPairSize += strs.length - 1;
			}
			reader.close();
			System.out.format("(%d, %d) users/pairs are loaded from %s...\n", m_userFriendMap.size(), m_trainPairSize, filename);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	// load testing friends
	public void loadTestFriends(String filename){
		int testSize = 0;
		int testFriendSize = 0;
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			// skip the first line
			while ((line = reader.readLine()) != null) {
				String[] strs = line.split("\t");
				String userID = strs[0];
				if(!m_userFriendMap.containsKey(userID)){
					System.out.println(String.format("[error]The testing user does not exist-%s", userID));
				} else{
					testSize++;
					m_userFriendMap.get(userID).setTestFriends(Arrays.copyOfRange(strs, 1, strs.length));
				}
				testFriendSize += strs.length - 1;
			}
			reader.close();
			System.out.format("(%d, %d) test users/friend-pairs are loaded from %s...\n", testSize, testFriendSize, filename);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	// load the non-friends for the testing users.
	public void loadTestNonFriends(String filename){
		int testSize = 0;
		int testNonFriendSize = 0;
		if (filename==null || filename.isEmpty())
				return;
			
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			//skip the first line
			while ((line = reader.readLine()) != null) {
				String[] strs = line.split("\t");
				String userID = strs[0];
				if(!m_userFriendMap.containsKey(userID)){
					System.out.println(String.format("[error]The testing user does not exist-%s", userID));
				} else{
					testSize++;
					m_userFriendMap.get(userID).setTestNonFriends(Arrays.copyOfRange(strs, 1, strs.length));
				}
				testNonFriendSize += strs.length - 1;
			}
			reader.close();
			System.out.format("(%d, %d) test users/friend-pairs are loaded from %s...\n", testSize, testNonFriendSize, filename);
			} catch (IOException e) {
				System.err.format("[Error]Failed to open file %s!!", filename);
			}
		}
	
	public void calcTestPairSize(String filename){
		m_testPairSize = 0;
		if (filename==null || filename.isEmpty())
			return;
		Set<String> userSet = new HashSet<>();
		Set<String> itemSet = new HashSet<>();
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			// skip the first line
			line = reader.readLine();
			while ((line = reader.readLine()) != null) {
				String[] strs = line.split(",");
				String userID = strs[0];
				String itemID = strs[1];
				userSet.add(userID);
				itemSet.add(itemID);
				m_testPairSize++;
			}
			reader.close();
			System.out.format("There are %d total testing pairs, %d test users, %d items.\n", m_testPairSize, userSet.size(), itemSet.size());
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}

	public int getTrainPairSize(){
		return m_trainPairSize;
	}
	public int getTestPairSize(){
		return m_testPairSize;
	}
	
	// save the user-user pairs to graphlab for model training.
	public void saveTrainTestFiles(String trainFile, String testFile){
		int testMiss = 0, testNonMiss = 0, test = 0;
		try{
			PrintWriter trainWriter = new PrintWriter(new File(trainFile));
			PrintWriter testWriter = new PrintWriter(new File(testFile));
			trainWriter.write("user_id,item_id,rating\n");
			testWriter.write("user_id,item_id,rating\n");
			for(String uid: m_userFriendMap.keySet()){
				_LinkFriend linkFriend = m_userFriendMap.get(uid);
				for(String frd: linkFriend.getTrainFriends()){
					trainWriter.write(String.format("%s,%s,%d\n", uid, frd, 1));
				}
				// for test users, we also need to write out non-friends
				if(!linkFriend.hasTestFriends()){
					testMiss++;
					continue;
				}
				if(!linkFriend.hasTestNonFriends()){
					testNonMiss++;
					continue;
				}
				test++;
				for(String frd: linkFriend.getTestFriends()){
					testWriter.write(String.format("%s,%s,%d\n", uid, frd, 1));
				}
				for(String nonfrd: linkFriend.getTestNonFriends()){
					testWriter.write(String.format("%s,%s,%d\n", uid, nonfrd, 0));
				}
			}
			trainWriter.close();
			testWriter.close();
			System.out.format("\n[Info]%d users don't have testing friends, %d users don't have non-friends, %d users are recorded. \n", testMiss, testNonMiss, test);
		} catch(IOException e){
			e.printStackTrace();
		}	
	}
	// transfer the user-item matrix to mm file for graphchi
	public void transfer2MMFile(String filename, String outputfile, int pairSize){
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			PrintWriter writer = new PrintWriter(new File(outputfile));
			writer.write("%%MatrixMarket matrix coordinate real general\n% Generated Jan, 2018\n");
			writer.write(String.format("%d\t%d\t%d\n", m_userIDs.size(), m_itemIDs.size(), pairSize));
			// skip the first line
			int count = 0;
			line = reader.readLine();
			count++;
			while ((line = reader.readLine()) != null) {
				count++;
				String[] strs = line.split(",");
				String userID = strs[0];
				String itemID = strs[1];
				if(!m_userMap.containsKey(userID))
					System.out.println("bug");
				int userIdx = m_userMap.get(userID)+1;
				if(!m_itemMap.containsKey(itemID))
					continue;
				int itemIdx = m_itemMap.get(itemID)+1;
				int rating = Integer.valueOf(strs[2]);
				writer.write(String.format("%d\t%d\t%d\n", userIdx, itemIdx, rating));
			}
			reader.close();
			writer.close();
			System.out.format("[Info]Finish transferring %d lines to MM File.\n", count);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	// transfer the user-item matrix to mm file for graphchi
	public void transfer2MMFil4Link(String filename, String outputfile, int pairSize){
		if (filename==null || filename.isEmpty())
			return;
			
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			PrintWriter writer = new PrintWriter(new File(outputfile));
			writer.write("%%MatrixMarket matrix coordinate real general\n% Generated Jan, 2018\n");
			writer.write(String.format("%d\t%d\t%d\n", m_userIDs.size(), m_itemIDs.size(), pairSize));
			// skip the first line
			int count = 0;
			line = reader.readLine();
			count++;
			while ((line = reader.readLine()) != null) {
				count++;
				String[] strs = line.split(",");
				String userID = strs[0];
				String itemID = strs[1];// user_2 too
				if(!m_userMap.containsKey(userID))
					System.out.print("x.");
				int userIdx = m_userMap.get(userID)+1;
				if(!m_userMap.containsKey(itemID)){
					System.out.print("x-");
					continue;
				}
				int itemIdx = m_userMap.get(itemID)+1;
				int rating = Integer.valueOf(strs[2]);
				writer.write(String.format("%d\t%d\t%d\n", userIdx, itemIdx, rating));
			}
			reader.close();
			writer.close();
			System.out.format("\n[Info]Finish transferring %d lines to MM File.\n", count);
		} catch (IOException e) {
			System.err.format("\n[Error]Failed to open file %s!!", filename);
		}
	}
	// transfer the user-item matrix to mm file for graphchi
	public void transfer2MMFileWithText(String filename, String outputfile, int pairSize){
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			PrintWriter writer = new PrintWriter(new File(outputfile));
			writer.write("%%MatrixMarket matrix coordinate real general\n% Generated Jan, 2018\n");
			writer.write(String.format("%d\t%d\t%d\n", m_userIDs.size(), m_itemIDs.size(), pairSize));
			// skip the first line
			int count = 0;
			line = reader.readLine();
			count++;
			while ((line = reader.readLine()) != null) {
				count++;
				String[] strs = line.split(",");
				String userID = strs[0];
				String itemID = strs[1];
				int userIdx = m_userMap.get(userID)+1;
				if(!m_itemMap.containsKey(itemID))
					continue;
				int itemIdx = m_itemMap.get(itemID)+1;
				int rating = Integer.valueOf(strs[2]);
				writer.write(String.format("%d\t%d\t%d,", userIdx, itemIdx, rating));
				int start = strs[0].length() + strs[1].length() + strs[2].length() + 3;
				writer.write(line.substring(start));
			}
			reader.close();
			writer.close();
			System.out.format("[Info]Finish transferring %d lines to MM File.\n", count);
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	public static void main(String[] args){
		String dataset = "YelpNew";
		
//		for(int p: new int[]{3, 5, 10}){
//			SVDPreProcess process = new SVDPreProcess();
//			String trainFile = String.format("./data/cfData/fm/%s_cf_all_nei_pop_%d_train.csv", dataset, p);
//			String testFile = String.format("./data/cfData/fm/%s_cf_all_nei_pop_%d_test.csv", dataset, p);
//			String trainMMFile = String.format("./data/cfData/svd/%s_cf_all_nei_pop_%d_train.mm", dataset, p);
//			String testMMFile = String.format("./data/cfData/svd/%s_cf_all_nei_pop_%d_test.mm", dataset, p);
//				
//			process.buildUserItemMap(trainFile);
//			process.calcTestPairSize(testFile);
//			
//			//process.transfer2MMFile(trainFile, trainMMFile, process.getTrainPairSize());
//			//process.transfer2MMFile(testFile, testMMFile, process.getTestPairSize());
//			System.out.println("---------------------------");
//				
//			/****
//			SVDPreProcess process = new SVDPreProcess();
//			String trainFile = String.format("./data/cfData/fm_text/%s_cf_time_%d_topk_%d_text_train.csv", dataset, t, k);
//			String testFile = String.format("./data/cfData/fm_text/%s_cf_time_%d_topk_%d_text_test.csv", dataset, t, k);
//			String trainMMFile = String.format("./data/cfData/svd_text/%s_cf_time_%d_topk_%d_text_train.mm", dataset, t, k);
//			String testMMFile = String.format("./data/cfData/svd_text/%s_cf_time_%d_topk_%d_text_test.mm", dataset, t, k);
//				
//			// transfer csv data to mm data
//			process.buildUserItemMap(trainFile);
//			process.calcTestPairSize(testFile);
//			process.transfer2MMFileWithText(trainFile, trainMMFile, process.getTrainPairSize());
//			process.transfer2MMFileWithText(testFile, testMMFile, process.getTestPairSize());
//			****/
//		}
			
		for(int t: new int[]{2}) {
		PreProcess process = new PreProcess();
		String trainFriend = String.format("./data/DataEUB/CV4Edges/%sCVIndex4Interaction_fold_0_train.txt", dataset);
		String testFriend = String.format("./data/DataEUB/CV4Edges/%sCVIndex4Interaction_fold_0_test.txt", dataset);
		String testNonFriend = String.format("./data/DataEUB/CV4Edges/%sCVIndex4NonInteraction_time_%d_fold_0.txt", dataset, t);

		String trainFile = String.format("./data/DataEUB/%s_link_pred_fold_0_train.csv", dataset);
		String testFile = String.format("./data/DataEUB/%s_link_pred_time_%d_fold_0_test.csv", dataset, t);

//		// load the train/test matrix
		process.loadTrainFriends(trainFriend);
		process.loadTestFriends(testFriend);
		process.loadTestNonFriends(testNonFriend);

		process.saveTrainTestFiles(trainFile, testFile);
		
		String trainMMFile = String.format("./data/linkPredData/%s_link_pred_fold_0_train.mm", dataset);
		String testMMFile = String.format("./data/linkPredData/%s_link_pred_time_%d_fold_0_test.mm", dataset, t);
		
		process.buildUserItemMap(trainFile);
		process.calcTestPairSize(testFile);
		
		process.transfer2MMFil4Link(trainFile, trainMMFile, process.getTrainPairSize());
		process.transfer2MMFil4Link(testFile, testMMFile, process.getTestPairSize());
		
//		process.transfer2MMFile(trainFile, trainMMFile, process.getTrainPairSize());
//		process.transfer2MMFile(testFile, testMMFile, process.getTestPairSize());
		}
	}
}
