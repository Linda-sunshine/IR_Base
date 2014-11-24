package Classifier;

import java.util.ArrayList;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;

public class NaiveBayes extends BaseClassifier {
	
	private double[] m_classProb;//p(c)
	private double[] m_classMember;//count(d|c)
	
	//Constructor.
	public NaiveBayes(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		this.m_model = new double [classNumber][featureSize];
		this.m_sstat = new double [classNumber][featureSize];
		this.m_classProb = new double [this.m_classNo];
		this.m_classMember = new double [this.m_classNo];
	}
	
	//Train the data set.
	public void train(ArrayList<_Doc> train_set){
		for(_Doc doc: train_set){
			int label = doc.getYLabel();
			this.m_classMember[label]++;
			for(_SparseFeature sf: doc.getSparse()){
				this.m_model[label][sf.getIndex()] += sf.getValue();
			}
		}
		calculateStat(this.m_model);
	}
	
	//Train the data set with term presence????
	//Is the numerator the total number of document in one class????
	//If is, I need to set different counters for different classes.
	public void trainPresence(ArrayList<_Doc> train_set){
		
		for(_Doc doc: train_set){
			int label = doc.getYLabel();
			for(_SparseFeature sf: doc.getSparse()){
				this.m_model[label][sf.getIndex()] += 1;
			}
		}
		calculateStat(this.m_model);
	}
	
	//Calculate the probabilities for different features in m_model;
	public void calculateStat(double[][] model){
		
		for(int i = 0; i < this.m_classNo; i++){
			this.m_classProb[i] = this.m_classMember[i]/this.m_trainSet.size();
		}
		for(int i = 0; i < model.length; i++){
			int sum = 0;
			for(int j = 0; j < model[i].length; j++){
				sum += model[i][j];
			}
			for(int j = 0; j < model[i].length; j++){
				this.m_sstat[i][j] = (this.m_model[i][j] + 1)/ (sum + this.m_featureSize);//add one smoothing
			}
		}
	}
	
	//Test the data set.
	public void test(ArrayList<_Doc> testSet){
		double[][] TPTable = new double [this.m_classNo][this.m_classNo];
		double[][] PreRecOfOneFold = new double[this.m_classNo][2];
		double[] probs = new double[this.m_classNo];
		
		for(_Doc doc: testSet){
			for(int i = 0; i < this.m_classNo; i++){
				double probability = Math.log(this.m_classProb[i]);
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
			TPTable[doc.getPredictLabel()][doc.getYLabel()] +=1; //Compare the predicted label and original label, construct the TPTable.
		}
		PreRecOfOneFold = calculatePreRec(TPTable);
		this.m_precisionsRecalls.add(PreRecOfOneFold);
		this.m_classProb = new double [this.m_classNo];
		this.m_classMember = new double [this.m_classNo];
	}
}
