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

import structures._User;
import Classifier.supervised.modelAdaptation._AdaptStruct;
import Classifier.supervised.modelAdaptation.MMB._MMBAdaptStruct;


public class LinkPredictionWithSVMSplit extends LinkPredictionWithSVM {

	HashMap<String, _MMBAdaptStruct> m_userMap;
	
	public LinkPredictionWithSVMSplit(double c){
		super(c);
		m_userMap = new HashMap<String, _MMBAdaptStruct>();
	}
	
	@Override
	public void calcTrainTestSize(){
		m_kBar = m_mmbModel.getKBar();
		m_allUserSize = m_mmbModel.getUserSize();
		
		m_trainSet = new ArrayList<_MMBAdaptStruct>();
		m_testSet = new ArrayList<_MMBAdaptStruct>();
		for(_AdaptStruct user: m_mmbModel.getUsers()){
			if(user.getTestSize() != 0){
				m_testSize++;
				m_testSet.add((_MMBAdaptStruct) user);
			}
			else{
				m_trainSize++;
				m_trainSet.add((_MMBAdaptStruct) user);
			}
		}
	
		if(m_trainSize + m_testSize != m_allUserSize)
			System.out.println("The user size does not match!!");
	}
	//we always encounter exceeds java heap size in training svm.
	// thus we split the training of user mixture and training of svm
	public void linkPrediction_Prep(String trainFile, String testFile){
		calcTrainTestSize();
		m_mmbModel.calculateMixturePerUser();
		
		saveTrainUsers(trainFile);
		saveTestUsers(testFile);
	}
	
	// calculate training/testing size, construct training set/testing set
	@Override
	public void initLinkPred(){
		// The train user and test user may not exist in order, thus we still set the friend 
		// matrix's size as the total number of users for convenient indexing. As their dim 
		// is different, we cannot put them in one array
		m_frdTrainMtx = new int[m_trainSize][m_trainSize-1];
		m_frdTestMtx = new int[m_testSize][m_allUserSize-1];
		m_simMtx = new double[m_allUserSize][m_allUserSize];	
	}
	
	@Override
	public void linkPrediction(){
		initLinkPred();
		trainSVM();
		_MMBAdaptStruct ui;
		// for each training user, rank their neighbors.
		for(int i=0; i<m_trainSize; i++){
			ui = m_trainSet.get(i);
			linkPrediction4TrainUsers(i, ui);
		}
		// for each testing user, rank their neighbors.
		for(int i=0; i<m_testSize; i++){
			ui = m_testSet.get(i);
			linkPrediction4TestUsers(i, ui);
		}
	}
	
	// perform link prediction in multi-threading
	@Override
	public void linkPrediction_MultiThread(){
		initLinkPred();				
		trainSVM();
					
		// use a boolean flag to decide whether it is training set or testing set
		linkPrediction_MultiThread_Split(m_trainSet, true);
		System.out.format("[Info]Finish link prediction on %d training users.\n", m_trainSize);

		linkPrediction_MultiThread_Split(m_testSet, false);
		System.out.format("[Info]Finish link prediction on %d testing users.\n", m_testSize);
	}		
		
	public void loadData(String trainFile, String testFile, String friendFile){
		loadTrainUserMixture(trainFile);
		loadTestUserMixture(testFile);
		loadFriends(friendFile);
	}
	
	// load training users
	public void loadTrainUserMixture(String trainFile){
		double[] mixture;
		String line, userID;
		_MMBAdaptStruct tmpUser;
		m_trainSet = new ArrayList<_MMBAdaptStruct>();
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(trainFile), "UTF-8"));
			while ((line = reader.readLine()) != null) {
				String[] strs = line.split(",");
				userID = strs[0];
				mixture = new double[strs.length-1];
				if(m_kBar == 0)
					m_kBar = strs.length - 1;
				
				for(int i = 1; i < strs.length; i++) {
					mixture[i-1] = Double.valueOf(strs[i]);
				}
				tmpUser = new _MMBAdaptStruct(new _User(userID), mixture);
				m_userMap.put(userID, tmpUser);
				m_trainSet.add(tmpUser);
				m_trainSize++; m_allUserSize++;
			}
			reader.close();
			System.out.print(String.format("[Info]%d train users mixture are loaded!\n", m_trainSize));
			
		} catch (IOException e) {
			System.err.format("[Error]Failed to open mixture files %s!!", trainFile);
			e.printStackTrace();
		}
	}
	
	// load testing users
	public void loadTestUserMixture(String testFile){
		double[] mixture;
		String line, userID;
		_MMBAdaptStruct tmpUser;
		m_testSet = new ArrayList<_MMBAdaptStruct>();
		try {
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(testFile), "UTF-8"));
			while ((line = reader.readLine()) != null) {
				String[] strs = line.split(",");
				userID = strs[0];
				mixture = new double[strs.length-1];
							
				for(int i = 1; i < strs.length; i++) {
					mixture[i-1] = Double.valueOf(strs[i]);
				}
				tmpUser = new _MMBAdaptStruct(new _User(userID), mixture);
				m_userMap.put(userID, tmpUser);
				m_testSet.add(tmpUser);
				m_testSize++;m_allUserSize++;
			}
			reader.close();
			System.out.print(String.format("[Info]%d test users mixture are loaded!\n", m_testSize));
		} catch (IOException e) {
			System.err.format("[Error]Failed to open mixture files %s!!", testFile);
			e.printStackTrace();
		}
	}	
	
	/** Construct user network for analysis****/
	public void loadFriends(String filename){
		try{
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8"));
			String line;
			String[] user, friends;
			while((line = reader.readLine()) != null){
				user = line.trim().split("\t");
				friends = Arrays.copyOfRange(user, 1, user.length);
				if(friends.length < 1){
					System.out.println("[Error]The user does not have any friends!");
					continue;
				}
				if(m_userMap.containsKey(user[0])){
					m_userMap.get(user[0]).getUser().setFriends(friends);
				} else {
					System.out.print("x");
				}
			}
			reader.close();
			System.out.println("\n[Info]Finish loading friends for users!");
		} catch(IOException e){
			e.printStackTrace();
		}
	}

	
	// save training users 
	public void saveTrainUsers(String trainFile){
		_MMBAdaptStruct ui;
		double[] mix;
		try{
			PrintWriter writer = new PrintWriter(trainFile);
			for(int i=0; i<m_trainSize; i++){
				ui = m_trainSet.get(i);
				writer.write(ui.getUserID()+",");
				mix = ui.getMixture();
				for(int v=0; v<mix.length-1; v++){
					writer.write(mix[v]+",");
				}
				writer.write(mix[mix.length-1]+"\n");
			}
			writer.close();
			System.out.println("[Info]Finish writing training users!");
		} catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public void saveTestUsers(String testFile){
		_MMBAdaptStruct ui;
		double[] mix;
		try{
			PrintWriter writer = new PrintWriter(testFile);
			for(int i=0; i<m_testSize; i++){
				ui = m_testSet.get(i);
				writer.write(ui.getUserID()+",");
				mix = ui.getMixture();
				for(int v=0; v<mix.length-1; v++){
					writer.write(mix[v]+",");
				}
				writer.write(mix[mix.length-1]+"\n");
			}
			writer.close();
			System.out.println("[Info]Finish writing testing users!");
		} catch(IOException e){
			e.printStackTrace();
		}
	}
}