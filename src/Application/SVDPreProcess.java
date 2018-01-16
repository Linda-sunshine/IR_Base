package Application;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;

public class SVDPreProcess {
	ArrayList<String> m_userIDs, m_itemIDs;
	HashMap<String, Integer> m_userMap;
	HashMap<String, Integer> m_itemMap;
	int m_trainPairSize = 0, m_testPairSize = 0;
	
	// low-rank representation of users 
	double[][] m_U;
	// low-rank representation of items 
	double[][] m_V;
	
	public SVDPreProcess(){
		m_userMap = new HashMap<String, Integer>();
		m_itemMap = new HashMap<String, Integer>();
		m_userIDs = new ArrayList<String>();
		m_itemIDs = new ArrayList<String>();
	}
	
	// load the user-item data and get the size of users and items
	public void buildUserItemMap(String filename){
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
					m_itemMap.put(itemID, m_itemMap.size());
					m_itemIDs.add(itemID);
				}
				m_trainPairSize++;
			}
			reader.close();
			System.out.format("(%d, %d) users/items are loaded from %s...\n", m_userIDs.size(), m_itemIDs.size(), filename);
			
		} catch (IOException e) {
			System.err.format("[Error]Failed to open file %s!!", filename);
		}
	}
	
	public void calcTestPairSize(String filename){
		if (filename==null || filename.isEmpty())
			return;
		
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			String line;
			// skip the first line
			line = reader.readLine();
			while ((line = reader.readLine()) != null) {
				m_testPairSize++;
			}
			reader.close();
			System.out.format("There are %d total testing pairs.", m_testPairSize);
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
	
	public static void main(String[] args){
		String dataset = "YelpNew";
		for(int t: new int[]{2,3,4,5}){
			for(int k: new int[]{4,6,8,10}){
				SVDPreProcess process = new SVDPreProcess();
				String trainFile = String.format("./data/cfData/fm/%s_cf_time_%d_topk_%d_train.csv", dataset, t, k);
				String testFile = String.format("./data/cfData/fm/%s_cf_time_%d_topk_%d_test_valid.csv", dataset, t, k);
				String trainMMFile = String.format("./data/cfData/svd/%s_cf_time_%d_topk_%d_train.mm", dataset, t, k);
				String testMMFile = String.format("./data/cfData/svd/%s_cf_time_%d_topk_%d_test.mm", dataset, t, k);
				// transfer csv data to mm data
				process.buildUserItemMap(trainFile);
				process.calcTestPairSize(testFile);
				
				process.transfer2MMFile(trainFile, trainMMFile, process.getTrainPairSize());
				process.transfer2MMFile(testFile, testMMFile, process.getTestPairSize());
			}
		}	
	}
	
}
