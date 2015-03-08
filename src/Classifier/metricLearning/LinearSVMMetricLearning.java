package Classifier.metricLearning;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Random;

import structures._Corpus;
import structures._Doc;
import structures._SparseFeature;
import utils.Utils;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;

public class LinearSVMMetricLearning extends GaussianFieldsByRandomWalk {
	protected Model m_libModel;
	int m_bound;
	double m_contSamplingRate;
	
	HashMap<Integer, Integer> m_selectedFVs;
	
	//Default constructor without any default parameters.
	public LinearSVMMetricLearning(_Corpus c, int classNumber, int featureSize, String classifier, int bound){
		super(c, classNumber, featureSize, classifier);
		m_bound = bound;
		m_contSamplingRate = 0.001; // a conservative setting
	}
	
	public LinearSVMMetricLearning(_Corpus c, int classNumber, int featureSize, String classifier, 
			double ratio, int k, int kPrime, double alhpa, double beta, double delta, double eta, boolean storeGraph, 
			int bound, double cSampleRate) {
		super(c, classNumber, featureSize, classifier, ratio, k, kPrime, alhpa, beta, delta, eta, storeGraph);
		m_bound = bound;
		m_contSamplingRate = cSampleRate;
	}

	@Override
	public String toString() {
		return "LinearSVM based Metric Learning for Gaussian Fields by Random Walk";
	}
	
	@Override
	protected double getSimilarity(_Doc di, _Doc dj) {
		Feature[] fv = createLinearFeature(di, dj);
		if (fv == null)
			return 0;
		else
			return Math.exp(Linear.predictValue(m_libModel, fv));//to make sure this is positive
	}
	
	@Override
	protected void init() {
		super.init();
		m_libModel = trainLibLinear(m_bound);
	}
	
	//debugging code
	void saveFv(int bound) {
		try {
			Feature[] fv = null;
			int mustLink = 0, cannotLink = 0, label;
			Random rand = new Random();
			
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream("metric.dat")));
			for(int i = 0; i < m_trainSet.size(); i++){
				_Doc d1 = m_trainSet.get(i);
				for(int j = i+1; j < m_trainSet.size(); j++){
					_Doc d2 = m_trainSet.get(j);
					if(d1.getYLabel() == d2.getYLabel())//start from the extreme case?  && (d1.getYLabel()==0 || d1.getYLabel()==4)
						label = 1;
					else if(Math.abs(d1.getYLabel() - d2.getYLabel())>bound)
						label = 0;
					else
						label = -1;
					
					if (label!=-1 && rand.nextDouble() < m_labelRatio) {
						fv = createLinearFeature(d1, d2);
						writer.write(String.format("%d", label));
						for(Feature f:fv){
							writer.write(String.format(" %d:%f", f.getIndex(), f.getValue()));//index starts from 1
						}
						writer.write('\n');
						
						if (label==1)
							mustLink ++;
						else
							cannotLink ++;
					}
				}
			}
			writer.close();
			System.out.format("Generating %d must-links and %d cannot links.\n", mustLink, cannotLink);
			
			System.exit(-1);
		} catch (IOException e) {
			e.printStackTrace();
		} 
	}
	
	//using L1 SVM to select a subset of features
	void selFeatures(Collection<_Doc> trainSet, double C) {
		Feature[][] fvs = new Feature[trainSet.size()][];
		double[] y = new double[trainSet.size()];
		
		int fid = 0;
		for(_Doc d:trainSet) {
			fvs[fid] = Utils.createLibLinearFV(d);
			y[fid] = d.getYLabel();
			fid ++;
		}
		
		Problem libProblem = new Problem();
		libProblem.l = fid;
		libProblem.n = m_featureSize;
		libProblem.x = fvs;
		libProblem.y = y;
		m_libModel = Linear.train(libProblem, new Parameter(SolverType.L1R_L2LOSS_SVC, C, 0.001));//use L1 regularization to reduce the feature size
		
		m_selectedFVs = new HashMap<Integer, Integer>();
		double[] w = m_libModel.getWeights();
		for(int i=0; i<m_featureSize; i++) {
			for(int c=0; c<m_classNo; c++) {
				if (w[i*m_classNo+c]!=0) {//a non-zero feature
					m_selectedFVs.put(i, m_selectedFVs.size());
					break;
				}	
			}
		}
		
		System.out.format("Selecting %d non-zero features by L1 regularization...\n", m_selectedFVs.size());
	}
	
	//In this training process, we want to get the weight of all pairs of samples.
	public Model trainLibLinear(int bound){
		selFeatures(m_trainSet, 0.5);
		
		int mustLink = 0, cannotLink = 0, label;
		Random rand = new Random();
		
		//In the problem, the size of feature size is m'*m'. (m' is the reduced feature space by L1-SVM)
		Feature[] fv;
		ArrayList<Feature[]> featureArray = new ArrayList<Feature[]>();
		ArrayList<Integer> targetArray = new ArrayList<Integer>();
		for(int i = 0; i < m_trainSet.size(); i++){//directly using m_trainSet should not be a good idea!
			_Doc d1 = m_trainSet.get(i);
			for(int j = i+1; j < m_trainSet.size(); j++){
				_Doc d2 = m_trainSet.get(j);
				if(d1.getYLabel() == d2.getYLabel())//start from the extreme case?  && (d1.getYLabel()==0 || d1.getYLabel()==4)
					label = 1;
				else if(Math.abs(d1.getYLabel() - d2.getYLabel())>bound)
					label = 0;
				else
					label = -1;
				
				if (label!=-1 && rand.nextDouble() < m_contSamplingRate) {
					fv = createLinearFeature(d1, d2);
					if (fv==null)
						continue;
					
					featureArray.add(fv);
					targetArray.add(label);
					
					if (label==1)
						mustLink ++;
					else
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
		Parameter libParameter = new Parameter(SolverType.L2R_L2LOSS_SVC_DUAL, C, eps);
		
		Problem libProblem = new Problem();
		libProblem.l = targetMatrix.length;
		libProblem.n = m_selectedFVs.size() * (1+m_selectedFVs.size())/2;
		libProblem.x = featureMatrix;
		libProblem.y = targetMatrix;
		Model model = Linear.train(libProblem, libParameter);
		return model;
	}
	
	//Calculate the new sample according to two documents.
	//Since cross-product will be symmetric, we don't need to store the whole matrix 
	Feature[] createLinearFeature(_Doc d1, _Doc d2){
		_SparseFeature[] diffVct = Utils.diffVector(d1.getSparse(), d2.getSparse());
		
		ArrayList<Feature> features = new ArrayList<Feature>();
		int pi, pj;
		double value = 0;
		for(int i = 0; i < diffVct.length; i++){
			pi = diffVct[i].getIndex();
			if (!m_selectedFVs.containsKey(pi))
				continue;//feature being discarded
			else
				pi = m_selectedFVs.get(pi);//map this feature to the reduced feature space
			
			for(int j = 0; j < i; j++){
				pj = diffVct[j].getIndex();
				if (!m_selectedFVs.containsKey(pj))
					continue;//feature being discarded
				else
					pj = m_selectedFVs.get(pj);//map this feature to the reduced feature space
				
				//Currently, we use one dimension array to represent V*V features 
				value = 2 * diffVct[i].getValue() * diffVct[j].getValue(); // this might be too small to count
				features.add(new FeatureNode(getIndex(pi, pj), value));
			}
			value = diffVct[i].getValue() * diffVct[i].getValue(); // this might be too small to count
			features.add(new FeatureNode(getIndex(pi, pi), value));
		}
		
		if (features.isEmpty())
			return null;//might hit nothing
		else
			return features.toArray(new Feature[features.size()]);
	}
	
	int getIndex(int i, int j) {
		if (i<j) {//swap
			int t = i;
			i = j;
			j = t;
		}
		return 1+i*(i+1)/2+j;//lower triangle for the square matrix, index starts from 1
	}
}
