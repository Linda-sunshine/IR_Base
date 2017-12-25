package Classifier.supervised.modelAdaptation.MMB;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;


public class SVMBasedLinkPredictionSplit extends SVMBasedLinkPrediction{

	public SVMBasedLinkPredictionSplit(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, betas);
	}
	
	//we always encounter exceeds java heap size in training svm.
	// thus we split the training of user mixture and training of svm
	public void linkPrediction_Prep(String trainFile, String testFile){
		calcTrainTestSize();
		calculateMixturePerUser();
		saveTrainUsers(trainFile);
		saveTestUsers(testFile);
	}
	
	@Override
	public void linkPrediction(){
		
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
	
	public void loadData(String trainFile, String testFile, String friendFile){
		loadTrainUsersMixture(trainFile);
		loadTestUserMixture(testFile);
		loadFriends(friendFile);
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