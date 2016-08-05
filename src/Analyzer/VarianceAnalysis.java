package Analyzer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;

import structures._User;

public class VarianceAnalysis {
	int m_featureSize;
	double[][] m_userWeights;
	ArrayList<_User> m_users;
	HashMap<String, Integer> m_userIDIndex; //Given a user ID, access the index of the user.
	String[] m_userIDs;//Given a user index, access ID of the user.

	public VarianceAnalysis(ArrayList<_User> users, int featureSize){
		m_users = users;
		m_featureSize = featureSize;
	}
	
	public void init(){
		String userID;
		m_userIDIndex = new HashMap<String, Integer>();

		for(int i=0; i<m_users.size(); i++){
			userID = m_users.get(i).getUserID();
			m_userIDIndex.put(userID, i);
		}
		
//		constructItemUserIndex();
		m_userIDs = new String[m_users.size()];
		for(int i=0; i<m_users.size(); i++){
			m_userIDs[i] = m_users.get(i).getUserID();
		}
	}
	
	public void loadUserWeights(String folder, String suffix){
		String userID;
		int userIndex;
		double[] weights;
		m_userWeights = new double[m_users.size()][];
		File dir = new File(folder);
		
		if(!dir.exists()){
			System.err.print("[Info]BoW is used as user weights.");
//			loadVSMWeights();
		} else{
			for(File f: dir.listFiles()){
				if(f.isFile() && f.getName().endsWith(suffix)){
					int endIndex = f.getName().lastIndexOf(".");
					userID = f.getName().substring(0, endIndex);
					if(m_userIDIndex.containsKey(userID)){
						userIndex = m_userIDIndex.get(userID);
						weights = loadOneUser(f.getAbsolutePath());
						m_userWeights[userIndex] = weights;
					}
				}
			}
		}
		System.out.format("%d users weights are loaded!", m_userWeights.length);
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
	public double[] calculateVar(){
		double[] vars = new double[m_featureSize];
		
		return vars;
	}
}
