package Classifier.metricLearning;

import java.util.ArrayList;
import java.util.Collection;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import Classifier.BaseClassifier;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;

public class LinearSVMMetricLearning extends BaseClassifier {
	protected ArrayList<_Doc> m_labeled; // a subset of training set
	
	protected Model m_libModel;
	
	//Default constructor without any default parameters.
	public LinearSVMMetricLearning(_Corpus c, int classNumber, int featureSize, String classifier){
		super(c, classNumber, featureSize);
	}

	@Override
	public String toString() {
		return "LinearSVM based Metric Learning";
	}
	
	@Override
	protected void init() {
		m_labeled.clear();
	}
	
	public void train(Collection<_Doc> trainSet){
		//Train the m_LinearWeight first with libliear.
		m_libModel = trainLibLinear(3);
	}
	
	//In this training process, we want to get the weight of all pairs of samples.
	public Model trainLibLinear(int bound){
		int mustLink = 0, cannotLink = 0;
		
		//In the problem, the size of feature size is m*m.
		ArrayList<Feature[]> featureArray = new ArrayList<Feature[]>();
		ArrayList<Double> targetArray = new ArrayList<Double>();
		for(int i = 0; i < m_trainSet.size(); i++){
			_Doc d1 = m_trainSet.get(i);
			for(int j = i+1; j < m_trainSet.size(); j++){
				_Doc d2 = m_trainSet.get(j);
				if(d1.getYLabel() == d2.getYLabel() && (d1.getYLabel()==0 || d1.getYLabel()==4)){//start from the extreme case?
					featureArray.add(createLinearFeature(d1, d2));
					targetArray.add(1.0); //If similiar, 1 + 2 = 3
					mustLink ++;
				} else if(Math.abs(d1.getYLabel() - d2.getYLabel())>bound){
					featureArray.add(createLinearFeature(d1, d2));
					targetArray.add(0.0); //If dissimilar, -1 + 2 = 1
					cannotLink ++;
				} 
			}
		}
		System.out.format("Generating %d must-links and %d cannot links.\n", mustLink, cannotLink);
		
		Feature[][] featureMatrix = new Feature[featureArray.size()][];
		double[] targetMatrix = new double[targetArray.size()];
		for(int i = 0; i < featureArray.size(); i++){
			featureMatrix[i] = featureArray.get(i);
			targetMatrix[i] = targetArray.get(i);
		}
		
		double C = 1.0, eps = 0.01;
		Parameter libParameter = new Parameter(SolverType.L2R_LR, C, eps);
		
		Problem libProblem = new Problem();
		libProblem.l = targetMatrix.length;
		libProblem.n = m_featureSize * (1+m_featureSize)/2;
		libProblem.x = featureMatrix;
		libProblem.y = targetMatrix;
		Model model = Linear.train(libProblem, libParameter);
		return model;
	}
	
	//Calculate the new sample according to two documents.
	//Since cross-product will be symmetric, we don't need to store the whole matrix 
	public Feature[] createLinearFeature(_Doc d1, _Doc d2){
		_SparseFeature[] diffVct = Utils.diffVector(d1.getSparse(), d2.getSparse());
		
		Feature[] features = new Feature[diffVct.length*(1+diffVct.length)/2];
		int pi, pj, spIndex = 0;
		double value = 0;
		for(int i = 0; i < diffVct.length; i++){
			pi = diffVct[i].getIndex();
			for(int j = 0; j < i; j++){
				pj = diffVct[j].getIndex();
				
				//Currently, we use one dimension array to represent V*V features 
				value = 2 * diffVct[i].getValue() * diffVct[j].getValue();
				features[spIndex++] = new FeatureNode(encode(pi, pj), value);
			}
			value = diffVct[i].getValue() * diffVct[i].getValue();
			features[spIndex++] = new FeatureNode(encode(pi, pi), value);
		}
		return features;
	}
	
	int encode(int i, int j) {
		if (i<j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		return i*(i+1)/2+j;//lower triangle for the square matrix
	}
	
	@Override
	protected void debug(_Doc d){} // no easy way to debug
	
	@Override
	public int predict(_Doc doc) {
		return -1; //we don't support this
	}
	
	//Save the parameters for classification.
	@Override
	public void saveModel(String modelLocation){
		
	}
}
