package Classifier.supervised.modelAdaptation.MMB;

import java.util.ArrayList;
import java.util.HashMap;

import Classifier.supervised.SVM;
import Classifier.supervised.liblinear.Feature;
import Classifier.supervised.liblinear.FeatureNode;
import Classifier.supervised.liblinear.Linear;
import Classifier.supervised.liblinear.Model;
import Classifier.supervised.liblinear.Parameter;
import Classifier.supervised.liblinear.Problem;
import Classifier.supervised.liblinear.SolverType;

public class SVMBasedLinkPrediction extends MTCLinAdaptWithMMB4LinkPrediction{

	double m_C = 1;
	Model m_libModel;
	
	public SVMBasedLinkPrediction(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, String globalModel,
			String featureGroupMap, String featureGroup4Sup, double[] betas) {
		super(classNo, featureSize, featureMap, globalModel, featureGroupMap,
				featureGroup4Sup, betas);
	}
	@Override
	public void linkPrediction(){
		initLinkPred();
		calculateMixturePerUser();

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
		calculateMixturePerUser();
			
		trainSVM();
				
		// use a boolean flag to decide whether it is training set or testing set
		linkPrediction_MultiThread_Split(m_trainSet, true);
		System.out.format("[Info]Finish link prediction on %d training users.\n", m_trainSize);

		linkPrediction_MultiThread_Split(m_testSet, false);
		System.out.format("[Info]Finish link prediction on %d testing users.\n", m_testSize);
	}		
		
	// we construct user pair as training instances and input into svm for training
	public void trainSVM(){
		// train svm model
		int trainSize = m_trainSize*(m_trainSize-1)/2;
		Feature[][] fvs = new Feature[trainSize][];
		double[] ys = new double[trainSize];
		
		constructXsYs(fvs, ys);
		Problem libProblem = new Problem();
		libProblem.l = trainSize;
		libProblem.x = fvs;
		libProblem.y = ys;
		libProblem.n = m_kBar*m_kBar+1;
		libProblem.bias = 1;// bias term in liblinear.

		SolverType type = SolverType.L2R_L1LOSS_SVC_DUAL;//solver type: SVM
		m_libModel = Linear.train(libProblem, new Parameter(type, m_C, SVM.EPS));

	}
	
	// calculate the similarity between two users based on trained sv
	@Override
	protected double calcSimilarity(_MMBAdaptStruct ui, _MMBAdaptStruct uj){
		return Linear.predictValue(m_libModel, constructOneX(ui, uj), 1);
	}
	
	protected void constructXsYs(Feature[][] fvs, double[] ys){
		// construct training instances
		double eij = 0;
		_MMBAdaptStruct ui, uj;
		int index = 0;
		for(int i=0; i<m_trainSize; i++){
			ui = m_trainSet.get(i);
			for(int j=i+1; j<m_trainSize; j++){
				uj = m_trainSet.get(j);
				fvs[index] = constructOneX(ui, uj);
				eij = ui.getUser().hasFriend(uj.getUserID()) ? 1 : 0;
				ys[index] = eij;
				index++;
			}
		}
	}
	
	protected Feature[] constructOneX(_MMBAdaptStruct ui, _MMBAdaptStruct uj){
		ArrayList<Feature> fv = new ArrayList<Feature>();
		double[] mixI = ui.getMixture(), mixJ = uj.getMixture();
		for(int k=0; k<mixI.length; k++){
			if(mixI[k] == 0) continue;
			for(int l=0; l<mixJ.length; l++){
				if(mixJ[l] == 0) continue;
				fv.add(new FeatureNode(k*m_kBar+l+1, mixI[k]*mixJ[l]));
			}
		}
		Feature[] x = new Feature[fv.size()];
		for(int i=0; i<fv.size(); i++){
			x[i] = fv.get(i);
		}
		return x;
	}
}
