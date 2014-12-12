package Classifier;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import structures._Corpus;
import structures._Doc;
import utils.Utils;

public class SemiSupervised extends BaseClassifier{
	protected double m_alpha; //Weight coefficient between unlabeled node and labeled node.
	protected double m_beta; //Weight coefficient between unlabeled node and unlabeled node.
	protected double m_M; //Influence of label to node.
	protected double m_k; // k labeled nodes.
	protected double m_kPrime;//k' unlabeled nodes.
	
	protected ArrayList<_Doc> m_labeled;
	protected DoubleMatrix2D m_Wij;
	protected DoubleMatrix2D m_Dii; //The diagonal degree matrix.
	protected DoubleMatrix2D m_delta;
	protected DoubleMatrix2D m_Cii;
	protected DoubleMatrix2D m_y;
	protected DoubleMatrix2D m_f;
	protected Algebra m_Algebra;
	
	protected NaiveBayes m_NB; //Multiple learner.
	
	//Randomly pick 10% of all the training documents.
	public SemiSupervised(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		this.m_alpha = 0;
		this.m_beta = 0;
		this.m_M = 0;
		this.m_k = 0;
		this.m_kPrime = 0;	
		this.m_labeled = new ArrayList<_Doc>();
		this.m_NB = new NaiveBayes(m_corpus, m_classNo, m_featureSize);
	}
	//Train the data set.
	public void train(){	
		//m_NB.train(m_trainSet);
		//Randomly pick some training documents as the labeled documents.
		for (int i = 0; i < m_trainSet.size(); i++){
			Random r = new Random();
			if(r.nextInt()%9 == 0){
				this.m_labeled.add(m_trainSet.get(i));
			}
		}
	}
	//Test the data set.
	public void test(){
		double similarity = 0;
		int size = this.m_labeled.size() + this.m_testSet.size();
		this.m_Wij = new DenseDoubleMatrix2D(size, size);
		//DoubleMatrix2D m_WijTranspose = new DenseDoubleMatrix2D(size, size);
		this.m_Dii = new DenseDoubleMatrix2D(size, size);
		this.m_Cii = new DenseDoubleMatrix2D(size, size);
		this.m_y = new DenseDoubleMatrix2D(size, 1);
		this.m_f = new DenseDoubleMatrix2D(size, 1);

		/***Construct the Wij matrix.****/
		//set the part of unlabeled nodes. U-U
		for(int i = 0; i < this.m_testSet.size(); i++){
			this.m_Wij.set(i, i, 0);
			for(int j = 0; j < i; j++){
				similarity = Utils.calculateSimilarity(m_testSet.get(i), m_testSet.get(j));
				similarity = this.m_beta * similarity;
				this.m_Wij.set(i, j, similarity);
				this.m_Wij.set(j, i, similarity);
			}	
		}
		//Set the part of labeled and unlabeled nodes. L-U and U-L
		for(int i = this.m_testSet.size(); i < size; i++){
			this.m_Wij.set(i, i, 0);
			for(int j = 0; j < size; j++){
				if(j < i){
					similarity = Utils.calculateSimilarity(m_labeled.get(i - m_testSet.size()), m_testSet.get(j));
					this.m_Wij.set(i, j, similarity);
					this.m_Wij.set(j, i, similarity);
				} else{
					//Set the part of labeled nodes. L-L
					this.m_Wij.set(j, i, 0);
				}
			}
		}
		//m_WijTranspose = this.m_Wij.viewDice();
		//or m_WijTranspose = m_Algebra.transpose(m_Wij);
		//this.m_Wij = max(this.m_Wij, m_WijTranspose);
		/****Construct the Dii matrix.****/
		for(int i = 0; i < size; i++){
			double sum = this.m_Wij.viewColumn(i).zSum();
			this.m_Dii.set(i, i, sum);
		}
		/****Construct the delta matrix, = Wij - Dii****/
		this.m_delta = new DenseDoubleMatrix2D(size, size);
		for(int i = 0; i < size; i++){
			for(int j = 0; j < size; j++){
				double temp = this.m_Wij.get(j, i) - this.m_Dii.get(j, i);
				this.m_delta.set(j, i, temp);
			}
		}
		/***Construct the Cii matrix.****/
		for(int i = 0; i < size; i++){
			if(i < this.m_testSet.size())
				this.m_Cii.set(i, i, 1);
			else this.m_Cii.set(i, i, m_M);
		}
		/***Construct the y matrix.****/
		for(int i = 0; i < size; i++){
			int tempLabel = 0;
			if(i < this.m_testSet.size()){
				tempLabel = m_NB.predictOneDoc(this.m_testSet.get(i)); //Multiple learner.
				this.m_y.set(i, 1, tempLabel);
			} else {
				tempLabel = this.m_labeled.get(i-this.m_testSet.size()).getYLabel();
				this.m_y.set(i, 1, tempLabel);
			}
		}
		//Predict the labels for all the labeled and unlabeled nodes.
		DoubleMatrix2D tempM = new DenseDoubleMatrix2D(size, size);
		double temp = 0;
		for(int i = 0; i < size; i++){
			for(int j = 0; j < size; j++){
				temp = m_Cii.get(j, i) + m_alpha/(m_k + m_beta*m_kPrime) * m_delta.get(j, i);
				tempM.set(j, i, temp);
			}
		}
		tempM = m_Algebra.inverse(tempM);
		this.m_f = m_Algebra.mult(m_Algebra.mult(tempM, m_Cii), m_y);
		
		/*******Show results*********/
		for(int i = 0; i < m_testSet.size(); i++){
			m_TPTable[(int) m_f.get(i, 1)][m_testSet.get(i).getYLabel()] += 1;
		}
		m_PreRecOfOneFold = calculatePreRec(m_TPTable);
		this.m_precisionsRecalls.add(m_PreRecOfOneFold);
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c) {
		c.shuffle(k);
		int[] masks = c.getMasks();
		HashMap<Integer, ArrayList<_Doc>> k_folder = new HashMap<Integer, ArrayList<_Doc>>();

		// Set the hash map with documents.
		for (int i = 0; i < masks.length; i++) {
			_Doc doc = c.getCollection().get(i);
			if (k_folder.containsKey(masks[i])) {
				ArrayList<_Doc> temp = k_folder.get(masks[i]);
				temp.add(doc);
				k_folder.put(masks[i], temp);
			} else {
				ArrayList<_Doc> docs = new ArrayList<_Doc>();
				docs.add(doc);
				k_folder.put(masks[i], docs);
			}
		}
		// Set the train set and test set.
		for (int i = 0; i < k; i++) {
			this.setTestSet(k_folder.get(i));
			for (int j = 0; j < k; j++) {
				if (j != i) {
					this.addTrainSet(k_folder.get(j));
				}
			}
			if (i == 0) m_NB.train();
			// Train the data set to get the parameter.
			train();
			test();
			this.m_trainSet.clear();
			// this.m_testSet.clear();
		}
		this.calculateMeanVariance(this.m_precisionsRecalls);
	}
}
