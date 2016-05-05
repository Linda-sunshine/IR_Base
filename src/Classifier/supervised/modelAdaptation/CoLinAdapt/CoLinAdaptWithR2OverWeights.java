package Classifier.supervised.modelAdaptation.CoLinAdapt;

import java.util.HashMap;

public class CoLinAdaptWithR2OverWeights extends CoLinAdapt {

	// The only difference between this method and CoLinAdapt is the R2 regularization is over weights.
	public CoLinAdaptWithR2OverWeights(int classNo, int featureSize,
			HashMap<String, Integer> featureMap, int topK, String globalModel,
			String featureGroupMap) {
		super(classNo, featureSize, featureMap, topK, globalModel, featureGroupMap);
	}
	
	
	public void 

}
