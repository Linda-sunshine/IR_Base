package Classifier;

import java.util.ArrayList;
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
		m_alpha = 0;
		m_beta = 0;
		m_M = 100000;
		m_k = 0;
		m_kPrime = 0;	
		m_labeled = new ArrayList<_Doc>();
		m_NB = new NaiveBayes(m_corpus, m_classNo, m_featureSize);
	}
	//Train the data set.
	public void train(){	
		//m_NB.train(m_trainSet);
		//Randomly pick some training documents as the labeled documents.
		for (int i = 0; i < m_trainSet.size(); i++){
			Random r = new Random();
			if(r.nextInt()%9 == 0){
				m_labeled.add(m_trainSet.get(i));
			}
		}
	}
	//Test the data set.
	public void test(){
		double similarity = 0;
		int size = m_labeled.size() + m_testSet.size();
		m_Wij = new DenseDoubleMatrix2D(size, size);
		//DoubleMatrix2D m_WijTranspose = new DenseDoubleMatrix2D(size, size);
		m_Dii = new DenseDoubleMatrix2D(size, size);
		m_Cii = new DenseDoubleMatrix2D(size, size);
		m_y = new DenseDoubleMatrix2D(size, 1);
		m_f = new DenseDoubleMatrix2D(size, 1);

		/***Construct the Wij matrix.****/
		//set the part of unlabeled nodes. U-U
		for(int i = 0; i < m_testSet.size(); i++){
			m_Wij.set(i, i, 0);
			for(int j = 0; j < i; j++){
				similarity = Utils.calculateSimilarity(m_testSet.get(i), m_testSet.get(j));
				similarity = m_beta * similarity;
				m_Wij.set(i, j, similarity);
				m_Wij.set(j, i, similarity);
			}	
		}
		//Set the part of labeled and unlabeled nodes. L-U and U-L
		for(int i = m_testSet.size(); i < size; i++){
			m_Wij.set(i, i, 0);
			for(int j = 0; j < size; j++){
				if(j < i){
					similarity = Utils.calculateSimilarity(m_labeled.get(i - m_testSet.size()), m_testSet.get(j));
					m_Wij.set(i, j, similarity);
					m_Wij.set(j, i, similarity);
				} else{
					//Set the part of labeled nodes. L-L
					m_Wij.set(j, i, 0);
				}
			}
		}
		//m_WijTranspose = this.m_Wij.viewDice();
		//or m_WijTranspose = m_Algebra.transpose(m_Wij);
		//this.m_Wij = max(this.m_Wij, m_WijTranspose);
		/****Construct the Dii matrix.****/
		for(int i = 0; i < size; i++){
			double sum = m_Wij.viewColumn(i).zSum();
			m_Dii.set(i, i, sum);
		}
		/****Construct the delta matrix, = Wij - Dii****/
		m_delta = new DenseDoubleMatrix2D(size, size);
		for(int i = 0; i < size; i++){
			for(int j = 0; j < size; j++){
				double temp = m_Wij.get(j, i) - m_Dii.get(j, i);
				m_delta.set(j, i, temp);
			}
		}
		/***Construct the Cii matrix.****/
		for(int i = 0; i < size; i++){
			if(i < m_testSet.size())
				m_Cii.set(i, i, 1);
			else m_Cii.set(i, i, m_M);
		}
		/***Construct the y matrix.****/
		for(int i = 0; i < size; i++){
			int tempLabel = 0;
			if(i < m_testSet.size()){
				tempLabel = m_NB.predictOneDoc(m_testSet.get(i)); //Multiple learner.
				m_y.set(i, 1, tempLabel);
			} else {
				tempLabel = m_labeled.get(i-m_testSet.size()).getYLabel();
				m_y.set(i, 1, tempLabel);
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
		m_f = m_Algebra.mult(m_Algebra.mult(tempM, m_Cii), m_y);
		
		/*******Show results*********/
		for(int i = 0; i < m_testSet.size(); i++){
			m_TPTable[(int) m_f.get(i, 1)][m_testSet.get(i).getYLabel()] += 1;
		}
		m_PreRecOfOneFold = calculatePreRec(m_TPTable);
		m_precisionsRecalls.add(m_PreRecOfOneFold);
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c) {
		c.shuffle(k);
		int[] masks = c.getMasks();
		ArrayList<_Doc> docs = c.getCollection();
		//Use this loop to iterate all the ten folders, set the train set and test set.
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < masks.length; j++) {
				if( masks[j]==i ) m_testSet.add(docs.get(j));
				else m_trainSet.add(docs.get(j));
			}
			m_NB.train(m_trainSet);
			train();
			test();
			m_trainSet.clear();
			m_testSet.clear();
		}
		calculateMeanVariance(m_precisionsRecalls);	
	}
}
