package Classifier;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import structures._Corpus;
import structures._Doc;
import utils.Utils;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

public class SemiSupervised extends BaseClassifier{
	protected double m_alpha; //Weight coefficient between unlabeled node and labeled node.
	protected double m_beta; //Weight coefficient between unlabeled node and unlabeled node.
	protected double m_M; //Influence of labeled node.
	protected double m_k; // k labeled nodes.
	protected double m_kPrime;//k' unlabeled nodes.
	
	protected ArrayList<_Doc> m_labeled; // a subset of training set
	protected double m_labelRatio; // percentage of training data for semi-supervised learning
	
	protected BaseClassifier m_classifier; //Multiple learner.
	
	//Randomly pick 10% of all the training documents.
	public SemiSupervised(_Corpus c, int classNumber, int featureSize){
		super(c, classNumber, featureSize);
		
		m_labelRatio = 0.1;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 100000;
		m_k = 0;
		m_kPrime = 0;	
		m_labeled = new ArrayList<_Doc>();
		m_classifier = new NaiveBayes(m_corpus, m_classNo, m_featureSize);
	}	
	
	public SemiSupervised(_Corpus c, int classNumber, int featureSize, double ratio){
		super(c, classNumber, featureSize);
		
		m_labelRatio = ratio;
		m_alpha = 1.0;
		m_beta = 0.1;
		m_M = 100000;
		m_k = 0;
		m_kPrime = 0;	
		m_labeled = new ArrayList<_Doc>();
		m_classifier = new NaiveBayes(m_corpus, m_classNo, m_featureSize);
	}
	
	@Override
	protected void init() {
		m_labeled.clear();
	}
	
	//Train the data set.
	public void train(Collection<_Doc> trainSet){	
		init();
		
		m_classifier.train(trainSet);
		
		//Randomly pick some training documents as the labeled documents.
		Random r = new Random();
		for (_Doc doc:trainSet){
			if(r.nextDouble()<m_labelRatio){
				m_labeled.add(doc);
			}
		}
	}
	
	//Test the data set.
	public void test(){
		double similarity = 0;
		int L = m_labeled.size(), U = m_testSet.size();
		double[][] Wij = new double[U+L][U+L];
		
		/***Set up K and K'.****/
		m_k = L;
		m_kPrime = U; // right now we used all
		
		/***Construct the Wij matrix.****/
		for(int i = 0; i < U; i++){
			//set the part of unlabeled nodes. U-U
			for(int j = 0; j < i; j++){//not including i-self
				similarity = m_beta * Utils.calculateSimilarity(m_testSet.get(i), m_testSet.get(j));
				Wij[i][j] = similarity;
				Wij[j][i] = similarity;
			}	
			
			//Set the part of labeled and unlabeled nodes. L-U and U-L
			for(int j = 0; j < L; j++){
				similarity = m_alpha * Utils.calculateSimilarity(m_testSet.get(i), m_labeled.get(j));
				Wij[i][j+U] = similarity;
				Wij[j+U][i] = similarity;
			}
		}
		
		/****Construct the C+scale*\Delta matrix and Y vector.****/
		double scale = -m_alpha / (m_k + m_beta*m_kPrime);
		double[] Y = new double[U+L];
		for(int i = 0; i < U+L; i++) {
			Wij[i][i] = -Utils.sumOfArray(Wij[i]);
			Utils.scaleArray(Wij[i], scale);
			
			if (i<U) {
				Wij[i][i] += 1.0;
				Y[i] = m_classifier.predict(m_testSet.get(i)); //Multiple learner.
			} else {
				Wij[i][i] += m_M;
				Y[i] = m_M * m_labeled.get(i-U).getYLabel();
			}
		}
		
		/***Perform matrix inverse.****/
		DenseDoubleMatrix2D mat = new DenseDoubleMatrix2D(Wij);
		Algebra alg = new Algebra();
		DoubleMatrix2D result = alg.inverse(mat);
		
		/*******Show results*********/
		for(int i = 0; i < U; i++){
			double pred = 0;
			for(int j=0; j<U+L; j++)
				pred += result.getQuick(i, j) * Y[j];
			
			m_TPTable[getLabel(pred)][m_testSet.get(i).getYLabel()] += 1;
		}
		m_PreRecOfOneFold = calculatePreRec(m_TPTable);
		m_precisionsRecalls.add(m_PreRecOfOneFold);
	}
	
	//get the closest int
	private int getLabel(double pred) {
		for(int i=0; i<m_classNo; i++)
			m_cProbs[i] = -Math.abs(i-pred); //-|c-p(c)|
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	@Override
	public int predict(_Doc doc) {
		return -1; //we don't support this
	}
}
