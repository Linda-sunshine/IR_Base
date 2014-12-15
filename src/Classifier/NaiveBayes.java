package Classifier;

import java.util.ArrayList;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;

public class NaiveBayes extends BaseClassifier {
	protected double[][] m_model; //The model contains total frequency for features /presences of features.
	protected double[][] m_sstat; //The probabilities of values in model.
	private double[] m_classProb;//p(c)
	private double[] m_classMember;//count(d|c)
	
	//Constructor.
	public NaiveBayes(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		m_model = new double [m_classNo][featureSize];
		m_sstat = new double [m_classNo][featureSize];
		m_classProb = new double [m_classNo];
		m_classMember = new double [m_classNo];
	}
	//Return the probs for differernt classes.
	public double[] getClassProbs() {
		return m_classProb;
	}
	//Train the data set.
	public void train(){
		for(_Doc doc: m_trainSet){
			int label = doc.getYLabel();
			m_classMember[label]++;
			for(_SparseFeature sf: doc.getSparse()){
				m_model[label][sf.getIndex()] += sf.getValue();
			}
		}
		calculateStat(m_model);
	}
	
	public void train(ArrayList<_Doc> trainSet){
		for(_Doc doc: trainSet){
			int label = doc.getYLabel();
			m_classMember[label]++;
			for(_SparseFeature sf: doc.getSparse()){
				m_model[label][sf.getIndex()] += sf.getValue();
			}
		}
		calculateStat(m_model);
	}
	
	//Train the data set with term presence????
	//Is the numerator the total number of document in one class????
	//If is, I need to set different counters for different classes.
	public void trainPresence(){
		for(_Doc doc: m_trainSet){
			int label = doc.getYLabel();
			for(_SparseFeature sf: doc.getSparse()){
				m_model[label][sf.getIndex()] += 1;
			}
		}
		calculateStat(m_model);
	}
	
	//Calculate the probabilities for different features in m_model;
	public void calculateStat(double[][] model){
		
		for(int i = 0; i < m_classNo; i++){
			m_classProb[i] = m_classMember[i]/m_trainSet.size();
		}
		for(int i = 0; i < model.length; i++){
			int sum = 0;
			for(int j = 0; j < model[i].length; j++){
				sum += model[i][j];
			}
			for(int j = 0; j < model[i].length; j++){
				m_sstat[i][j] = (m_model[i][j] + 1)/ (sum + m_featureSize);//add one smoothing
			}
		}
	}
	//Test the data set.
	public void test(){
		double[] probs = new double[m_classNo];
		
		for(_Doc doc: m_testSet){
			for(int i = 0; i < m_classNo; i++){
				double probability = Math.log(m_classProb[i]);
				double[] sparseProbs = new double[doc.getSparse().length];
				double[] sparseValues = new double[doc.getSparse().length];
				
				//Construct probs array and values array first.
				for(int j = 0; j < doc.getSparse().length; j++){
					int index = doc.getSparse()[j].getIndex();
					sparseValues[j] = doc.getSparse()[j].getValue();
					sparseProbs[j] = m_sstat[i][index];
				}
				probability += Utils.sumLog(sparseProbs, sparseValues);
				probs[i] = probability;
			}
			doc.setPredictLabel(Utils.maxOfArrayIndex(probs)); //Set the predict label according to the probability of different classes.
			m_TPTable[doc.getPredictLabel()][doc.getYLabel()] +=1; //Compare the predicted label and original label, construct the TPTable.
		}
		m_PreRecOfOneFold = calculatePreRec(m_TPTable);
		m_precisionsRecalls.add(m_PreRecOfOneFold);
		m_classProb = new double [m_classNo];
		m_classMember = new double [m_classNo];
	}
	
	//Predict the label for one document.
	public int predictOneDoc(_Doc d){
		int label = 0;
		double[] probs = new double[m_classNo];
		for(int i = 0; i < m_classNo; i++){
			double probability = Math.log(m_classProb[i]);
			double[] sparseProbs = new double[d.getSparse().length];
			double[] sparseValues = new double[d.getSparse().length];
			//Construct probs array and values array first.
			for(int j = 0; j < d.getSparse().length; j++){
				int index = d.getSparse()[j].getIndex();
				sparseValues[j] = d.getSparse()[j].getValue();
				sparseProbs[j] = m_sstat[i][index];
			}
			probability += Utils.sumLog(sparseProbs, sparseValues);
			probs[i] = probability;
		}
		label = Utils.maxOfArrayIndex(probs);
		return label;
	}
	
	//Save the parameters for classification.
	public void saveModel(String modelLocation){
		
	}
}
