package Classifier.supervised;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;

import structures.MyPriorityQueue;
import structures._Corpus;
import structures._Doc;
import structures._RankItem;
import structures._SparseFeature;
import utils.Utils;
import Classifier.BaseClassifier;

public class NaiveBayes extends BaseClassifier {
	private double[][] m_Pxy; // p(X|Y)
	private double[] m_pY;//p(Y)
	private boolean m_presence;
	private double m_deltaY; // for smoothing p(Y) purpose;
	private double m_deltaXY; // for smoothing p(X|Y) purpose;
	
	//Constructor.
	public NaiveBayes(_Corpus c){
		super(c);
		m_Pxy = new double [m_classNo][m_featureSize];
		m_pY = new double [m_classNo];
		
		m_presence = false;
		m_deltaY = 0.1;
		m_deltaXY = 0.1;
	}
	
	//Constructor.
	public NaiveBayes(int classNo, int featureSize){
		super(classNo, featureSize);
		m_Pxy = new double [m_classNo][m_featureSize];
		m_pY = new double [m_classNo];
		
		m_presence = false;
		m_deltaY = 0.1;
		m_deltaXY = 0.1;
	}
	
	//Constructor.
	public NaiveBayes(_Corpus c, boolean presence, double deltaY, double deltaXY){
		super(c);
		m_Pxy = new double [m_classNo][m_featureSize];
		m_pY = new double [m_classNo];
		
		m_presence = presence;
		m_deltaY = deltaY;
		m_deltaXY = deltaXY;
	}
	
	@Override
	public String toString() {
		return String.format("Naive Bayes[C:%d, F:%d]", m_classNo, m_featureSize);
	}
	
	protected void init() {
		for(int i=0; i<m_classNo; i++) {
			Arrays.fill(m_Pxy[i], 0);
			m_pY[i] = 0;
		}
	}
	
	//Train the data set.
	public void train(Collection<_Doc> trainSet){
		init();
		
		for(_Doc doc: trainSet){
			int label = doc.getYLabel();
			m_pY[label] ++;
			for(_SparseFeature sf: doc.getSparse())
				m_Pxy[label][sf.getIndex()] += m_presence?1.0:sf.getValue();
		}
		
		//normalization
		for(int i = 0; i < m_classNo; i++){
			m_pY[i] = Math.log(m_pY[i] + m_deltaY);//up to a constant since normalization of this is not important
			double sum = Math.log(Utils.sumOfArray(m_Pxy[i]) + m_featureSize*m_deltaXY);
			for(int j = 0; j < m_featureSize; j++)
				m_Pxy[i][j] = Math.log(m_deltaXY+m_Pxy[i][j]) - sum;
		}
		
		//printTopFeatures(5);
	}
		
	//Predict the label for one document.
	@Override
	public int predict(_Doc d){
		for(int i = 0; i < m_classNo; i++){
			m_cProbs[i] = m_pY[i];
			for(_SparseFeature f:d.getSparse())
				m_cProbs[i] += m_Pxy[i][f.getIndex()] * (m_presence?1.0:f.getValue());
		}
		return Utils.maxOfArrayIndex(m_cProbs);
	}
	
	@Override
	public double score(_Doc d, int label){
		for(int i = 0; i < m_classNo; i++){
			m_cProbs[i] = m_pY[i];
			for(_SparseFeature f:d.getSparse())
				m_cProbs[i] += m_Pxy[i][f.getIndex()] * (m_presence?1.0:f.getValue()); // in log space
		}
		return m_cProbs[label] - Utils.logSumOfExponentials(m_cProbs);
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation) {
		
	}
	
	// ranking the features in desceding order from Naive Bayes p(X|Y=1) and p(X|Y=0) 
	// positive features at the top and negative features at the bootom of the list
	// 1 means cons and 0 means pros
	public void printTopFeaturesSet(int topK, ArrayList<String> features) {
		MyPriorityQueue<_RankItem> queue = new MyPriorityQueue<_RankItem>(topK, false);
		double logRatio = 0.0;
		for(int n=0; n<m_featureSize; n++) {
			for(int i=0; i<m_classNo; i++)
				m_cProbs[i] = Math.exp(m_Pxy[i][n]); // converting to original space since m_Pxy in logSpace
			logRatio = Math.log(m_cProbs[1]) - Math.log(m_cProbs[0]); // log Ratio since we have only two classes
			queue.add(new _RankItem(n, logRatio));
		}
		
		System.out.print("Most discriminative features:\n");
		int i = 0;
		for(_RankItem item:queue) {
			System.out.format("%d:%s(%f)\n", i++, features.get(item.m_index), item.m_value);
		}
		System.out.println();
	}

	@Override
	protected void debug(_Doc d) {
		//to be implemented
	}
}
